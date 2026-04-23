import re, os
import tempfile
import numpy as np
import soundfile as sf
import editdistance
import parselmouth
from faster_whisper import WhisperModel


# ─── 模型懒加载 ────────────────────────────────────

_MODEL: WhisperModel | None = None

def _get_model() -> WhisperModel:
    """首次调用时才加载 Whisper；支持通过环境变量配置"""
    global _MODEL
    if _MODEL is None:
        size    = os.getenv("WHISPER_MODEL",   "base.en")
        device  = os.getenv("WHISPER_DEVICE",  "cpu")
        compute = os.getenv("WHISPER_COMPUTE", "int8")
        _MODEL = WhisperModel(size, device=device, compute_type=compute)
    return _MODEL


# ─── 工具函数 ────────────────────────────────────────────

def _normalize(word: str) -> str:
    """去掉标点、转小写"""
    return re.sub(r"[^\w]", "", word).lower()

def _normalize_text(text: str) -> str:
    """对整句做 normalize，用于 WER 计算"""
    return " ".join(_normalize(w) for w in text.split())

def _estimate_syllables(word: str) -> int:
    """
    粗略估算英文音节数：连续元音记为 1 个音节；
    末尾静音 e 不计入。用于判断词时长是否合理。
    """
    if not word:
        return 1
    vowels = set("aeiouy")
    count, prev = 0, False
    for c in word.lower():
        cur = c in vowels
        if cur and not prev:
            count += 1
        prev = cur
    if word.lower().endswith("e") and count > 1:
        count -= 1
    return max(1, count)


# ─── 音频预处理 ────────────────────────────────────

def preprocess_audio(audio_path: str) -> str:
    """
    标准化为 16kHz 单声道 PCM_16 WAV；
    写入独立临时文件避免并发评测时相互覆盖。
    """
    import librosa
    audio, _ = librosa.load(audio_path, sr=16000, mono=True)
    fd, out_path = tempfile.mkstemp(suffix="_16k.wav", prefix="eval_")
    os.close(fd)
    sf.write(out_path, audio, 16000, subtype="PCM_16")
    return out_path


# ─── 转录 ────────────────────────────────────────────────

def transcribe(audio_path: str) -> dict:
    """转录并返回词级时间戳，word 已 normalize"""
    segments, _ = _get_model().transcribe(
        audio_path,
        word_timestamps=True,
        language="en",
        beam_size=5,
    )

    words, texts = [], []
    for segment in segments:
        texts.append(segment.text)
        for word in segment.words:
            raw = word.word.strip()
            words.append({
                "word":      raw,
                "word_norm": _normalize(raw),
                "start":     word.start,
                "end":       word.end,
                "prob":      word.probability,
            })

    return {"text": "".join(texts).strip(), "words": words}


# ─── 声学特征分析器（全局缓存，避免重复计算）──────────

class _AudioAnalyzer:
    """一次性提取 pitch，后续切片查询，避免反复调用 parselmouth"""

    def __init__(self, audio_path: str):
        self.snd      = parselmouth.Sound(audio_path)
        self.duration = self.snd.duration
        pitch         = self.snd.to_pitch()
        self._f0      = pitch.selected_array["frequency"]
        self._times   = pitch.xs()

    def voicing_ratio(self, start: float, end: float) -> float:
        """指定区间内含基频 (voiced) 的帧占比"""
        mask = (self._times >= start) & (self._times < end)
        if not mask.any():
            return 0.5  # 采样过少，返回中性值
        return float((self._f0[mask] > 0).mean())

    def pitch_std_semitones(self, ref_hz: float = 100.0) -> float:
        """
        将 f0 转成半音后取 std，消除说话人基频差异（男/女声对齐）。
        正常朗读 2–4 st，更有表现力的语调 >4 st。
        """
        voiced = self._f0[self._f0 > 0]
        if len(voiced) < 5:
            return 0.0
        st = 12 * np.log2(voiced / ref_hz)
        return float(np.std(st))


# ─── 发音分 · 方案 A：启发式（默认，无需额外模型）────────

def _pronunciation_score(word_info: dict, analyzer: _AudioAnalyzer) -> float:
    """
    复合发音分，弥补单用 Whisper prob 的偏差：

      prob            —— ASR 声学+语言综合置信度（与清晰度弱相关）
      dur_deviation   —— 词时长与音节数期望偏差（含糊一带而过 / 过度拖沓）
      voicing_penalty —— 浊音比偏低（含糊、气声、漏读元音）

    这三项均可从现有音频/转录结果免费获得，不增加模型推理开销。
    """
    prob      = word_info["prob"]
    duration  = max(word_info["end"] - word_info["start"], 0.01)
    syllables = _estimate_syllables(word_info["word_norm"])
    expected  = syllables * 0.22  # 英语正常语速约 220ms/音节

    # 时长偏离：过短（<50% 期望）或过长（>220% 期望）才扣分
    dur_deviation = 0.0
    if duration < expected * 0.5:
        dur_deviation = (expected * 0.5 - duration) / (expected * 0.5)
    elif duration > expected * 2.2:
        dur_deviation = min(1.0, (duration - expected * 2.2) / (expected * 2.0))

    voicing = analyzer.voicing_ratio(word_info["start"], word_info["end"])
    # 典型清晰发音的浊音比应 > 0.35；低于此阈值按比例扣分
    voicing_penalty = max(0.0, (0.35 - voicing) / 0.35)

    score  = prob * 100.0
    score -= dur_deviation   * 20.0
    score -= voicing_penalty * 15.0
    return max(0.0, min(100.0, score))


# ─── 发音分 · 方案 B：wav2vec2 CTC 后验（可选）────────────
#
# 原理（经典 GOP-lite）：
#   1. 整段音频过一次 wav2vec2-base-960h，取每帧对 29 个字符的后验概率
#   2. 用 Whisper 给出的词级时间戳限定每个词的帧窗口
#   3. 对该窗口内每个目标字符取 max 后验（GOP 经典做法）
#   4. 词内所有字符的后验均值 ×100 即为发音分
#
# 模型大小 ~95M，CPU 上对 5–10s 的句子整段推理约 1–3 秒，每句只跑一次。
# 懒加载：只有首次调用 wav2vec2 方案时才加载模型，不影响启发式方案的启动。

_WAV2VEC2_HOLDER = None

class _Wav2Vec2Holder:
    def __init__(self):
        import torch
        import torchaudio
        from torchaudio.pipelines import WAV2VEC2_ASR_BASE_960H
        self._torch     = torch
        self._ta        = torchaudio
        bundle          = WAV2VEC2_ASR_BASE_960H
        self.sample_rate = bundle.sample_rate
        self.model       = bundle.get_model()
        self.model.eval()
        self.labels      = bundle.get_labels()
        self.label_to_id = {c: i for i, c in enumerate(self.labels)}

def _get_wav2vec2() -> "_Wav2Vec2Holder":
    global _WAV2VEC2_HOLDER
    if _WAV2VEC2_HOLDER is None:
        try:
            _WAV2VEC2_HOLDER = _Wav2Vec2Holder()
        except ImportError as e:
            raise ImportError(
                "wav2vec2 评分需要 torch + torchaudio，请先安装：\n"
                "    pip install torch torchaudio"
            ) from e
    return _WAV2VEC2_HOLDER

def _wav2vec2_pronunciation_scores(audio_path: str, hyp_words: list) -> list[float]:
    """对 hyp 的每个词返回 [0, 100] 的发音分"""
    holder = _get_wav2vec2()
    torch  = holder._torch
    ta     = holder._ta

    # 用 soundfile 读取而非 torchaudio.load：后者在 torchaudio>=2.1 依赖 torchcodec
    # + FFmpeg DLL，Windows 上常因缺 DLL 报错。我们的输入统一已是 16kHz 单声道 WAV
    # (经 preprocess_audio 预处理)，用 soundfile 即可，轻量且无额外原生依赖。
    audio_np, sr = sf.read(audio_path, dtype="float32", always_2d=False)
    if audio_np.ndim > 1:
        audio_np = audio_np.mean(axis=1)
    waveform = torch.from_numpy(audio_np).unsqueeze(0)  # [1, T]
    if sr != holder.sample_rate:
        waveform = ta.functional.resample(waveform, sr, holder.sample_rate)

    duration_s = waveform.shape[1] / holder.sample_rate
    with torch.inference_mode():
        emissions, _ = holder.model(waveform)
        probs = torch.softmax(emissions, dim=-1)[0]  # [T_frames, C]

    total_frames = probs.shape[0]
    frame_rate   = total_frames / max(duration_s, 1e-6)

    scores = []
    for w in hyp_words:
        start_f = max(0, int(w["start"] * frame_rate))
        end_f   = min(total_frames, int(w["end"] * frame_rate) + 1)
        if end_f <= start_f:
            scores.append(50.0)
            continue

        word   = w["word_norm"].upper()
        window = probs[start_f:end_f]  # [N_frames, C]
        char_posteriors = []
        for c in word:
            cid = holder.label_to_id.get(c)
            if cid is None:
                continue  # 数字/未知字符，跳过
            char_posteriors.append(window[:, cid].max().item())

        if not char_posteriors:
            scores.append(50.0)
            continue
        avg = sum(char_posteriors) / len(char_posteriors)
        scores.append(max(0.0, min(100.0, avg * 100.0)))

    return scores


# ─── 发音分调度入口 ──────────────────────────────────────

SCORER_HEURISTIC = "heuristic"
SCORER_WAV2VEC2  = "wav2vec2"

def _compute_pron_scores(hyp_words: list, analyzer: _AudioAnalyzer,
                         audio_path: str, scorer: str) -> list[float]:
    """
    根据 scorer 选择发音评分方案：
      - "heuristic" : Whisper prob + 声学启发式（零额外模型，快）
      - "wav2vec2"  : wav2vec2-base-960h CTC 后验 GOP（更贴近真实发音，较慢）
    """
    if scorer == SCORER_WAV2VEC2:
        return _wav2vec2_pronunciation_scores(audio_path, hyp_words)
    # 兜底：未知 scorer 名 → 启发式
    return [_pronunciation_score(w, analyzer) for w in hyp_words]


# ─── 准确度评分 ──────────────────────────────────────────

def score_accuracy(ref_norm_words: list[str], hyp_result: dict) -> dict:
    hyp_norm_words = [w["word_norm"] for w in hyp_result["words"]]
    distance = editdistance.eval(ref_norm_words, hyp_norm_words)
    wer      = distance / max(len(ref_norm_words), 1)
    accuracy = max(0.0, round((1 - wer) * 100, 1))
    return {
        "accuracy_score": accuracy,
        "wer":            round(wer, 3),
    }


# ─── 流利度评分 ──────────────────────────

PAUSE_THRESHOLD = 0.4  # 秒；低于此值视为正常连读/呼吸，不计停顿

def score_fluency(analyzer: _AudioAnalyzer, word_timestamps: list) -> dict:
    total_duration = analyzer.duration
    total_words    = len(word_timestamps)

    # WPM 用首尾跨度，而不是"词时长之和"
    if total_words >= 2:
        span = word_timestamps[-1]["end"] - word_timestamps[0]["start"]
    elif total_words == 1:
        span = total_duration
    else:
        span = 0.0
    wpm = (total_words / span * 60) if span > 0 else 0.0

    # 停顿阈值 0.25 → 0.4
    pauses = [
        word_timestamps[i]["start"] - word_timestamps[i-1]["end"]
        for i in range(1, total_words)
        if word_timestamps[i]["start"] - word_timestamps[i-1]["end"] > PAUSE_THRESHOLD
    ]
    pause_count = len(pauses)
    avg_pause   = float(np.mean(pauses)) if pauses else 0.0

    # 语调变化改用半音（对男/女声鲁棒）
    pitch_std_st = analyzer.pitch_std_semitones()

    return {
        "fluency_score":      _calc_fluency_score(wpm, pause_count, avg_pause, total_words),
        "words_per_minute":   round(wpm, 1),
        "pause_count":        pause_count,
        "avg_pause_duration": round(avg_pause, 2),
        "pitch_variation_st": round(pitch_std_st, 2),
    }

def _speed_to_score(wpm: float) -> float:
    """
    分段线性，连续、无跳变：
      0→20, 80→60, 120→100, 150→100, 200→40, 300→20
    """
    if wpm <= 0:          return 0.0
    if wpm < 80:          return 20 + wpm * 0.5
    if wpm < 120:         return 60 + (wpm - 80) * 1.0
    if wpm <= 150:        return 100.0
    if wpm <= 200:        return 100 - (wpm - 150) * 1.2
    return max(20.0, 40 - (wpm - 200) * 0.2)

def _pause_to_score(pause_count: int, avg_pause: float, total_words: int) -> float:
    """
    用"每词平均停顿次数"+"平均停顿时长"统一打分，避免重复扣分。
      ≤1/4 词一次停顿：不扣
      每超出 0.1，扣 12 分（上限 30）
      平均停顿 >0.6s，每超 0.1s 扣 2.5 分（上限 20）
    """
    if total_words == 0:
        return 0.0
    pause_per_word     = pause_count / total_words
    rate_penalty       = min(30.0, max(0.0, pause_per_word - 0.25) * 120)
    len_penalty        = min(20.0, max(0.0, avg_pause - 0.6)       * 25)
    return max(0.0, 100.0 - rate_penalty - len_penalty)

def _calc_fluency_score(wpm: float, pause_count: int,
                        avg_pause: float, total_words: int) -> float:
    """流利度 = 语速分 × 0.5 + 停顿分 × 0.5；不再做叠加式硬阈值扣分"""
    speed = _speed_to_score(wpm)
    pause = _pause_to_score(pause_count, avg_pause, total_words)
    return round(speed * 0.5 + pause * 0.5, 1)


# ─── 词级评分 ─────────────────────────────────

def score_words(ref_norm_words: list[str], hyp_result: dict,
                pron_scores: list[float]) -> list:
    """
    pron_scores: 与 hyp_result["words"] 一一对应的发音分（已预计算，
                 可能来自启发式或 wav2vec2 方案）
    """
    aligned = _align_words(ref_norm_words, hyp_result["words"])
    hyp_to_score = {id(w): s for w, s in zip(hyp_result["words"], pron_scores)}

    result = []
    for ref_w, hyp_w in aligned:
        if hyp_w is None:
            result.append({"word": ref_w, "score": 0, "tag": "missing"})
        elif ref_w is None:
            result.append({"word": hyp_w["word_norm"], "score": 0, "tag": "inserted"})
        else:
            pron  = hyp_to_score.get(id(hyp_w), 0.0)
            exact = (ref_w == hyp_w["word_norm"])
            if not exact:
                # #8 读错的词最多给 ~15 分（而不是之前的 30 分底）
                score, tag = round(pron * 0.15), "substituted"
            elif pron >= 85:
                score, tag = round(pron), "good"
            elif pron >= 70:
                score, tag = round(pron), "ok"
            else:
                score, tag = round(pron), "poor"
            result.append({
                "word":     ref_w,
                "hyp_word": hyp_w["word_norm"],
                "score":    score,
                "tag":      tag,
            })
    return result

def _similar(a: str, b: str) -> bool:
    """#6 相对编辑距离：最多 1/3 字符差异才算同一个词"""
    if a == b:
        return True
    if not a or not b:
        return False
    return editdistance.eval(a, b) / max(len(a), len(b)) <= 0.34

def _align_words(ref_norm: list[str], hyp: list) -> list:
    hyp_norm = [w["word_norm"] for w in hyp]
    m, n = len(ref_norm), len(hyp_norm)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1): dp[i][0] = i
    for j in range(n + 1): dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if _similar(ref_norm[i-1], hyp_norm[j-1]):
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])

    aligned, i, j = [], m, n
    while i > 0 or j > 0:
        if i > 0 and j > 0 and _similar(ref_norm[i-1], hyp_norm[j-1]):
            aligned.append((ref_norm[i-1], hyp[j-1]))
            i -= 1; j -= 1
        elif i > 0 and (j == 0 or dp[i-1][j] <= dp[i][j-1]):
            aligned.append((ref_norm[i-1], None))
            i -= 1
        else:
            aligned.append((None, hyp[j-1]))
            j -= 1
    return list(reversed(aligned))


# ─── 反馈生成 ────────────────────────────────────────

def _collect_weak_words(word_scores: list, limit: int = 5) -> list:
    """从词级评分中提取真正有问题的词（去重），与 accuracy 口径一致"""
    seen, out = set(), []
    for w in word_scores:
        if w["tag"] in ("poor", "substituted", "missing") and w["word"] not in seen:
            seen.add(w["word"])
            out.append(w["word"])
            if len(out) >= limit:
                break
    return out

def _generate_feedback(fluency: dict, weak_words: list) -> list[str]:
    tips = []
    if weak_words:
        tips.append(f"发音需改进的词：{', '.join(weak_words[:3])}")
    if fluency["words_per_minute"] < 100:
        tips.append("语速偏慢，尝试减少停顿")
    elif fluency["words_per_minute"] > 180:
        tips.append("语速偏快，注意咬字清晰")
    if fluency["pause_count"] > 5:
        tips.append(f"停顿次数较多（{fluency['pause_count']}次），注意句子连贯性")
    if fluency["pitch_variation_st"] < 2.0:
        tips.append("语调变化较少，尝试更有表现力地朗读")
    return tips if tips else ["发音流畅，继续保持！"]


# ─── 主入口 ──────────────────────────────────────────────

def evaluate(ref_text: str, audio_path: str, *, scorer: str | None = None) -> dict:
    """
    scorer: 发音打分方案
      - None (默认)：读取环境变量 PRONUNCIATION_SCORER，未设置则用 "heuristic"
      - "heuristic"：零额外模型，Whisper prob + 声学特征；快，适合 CPU 默认
      - "wav2vec2" ：加载 wav2vec2-base-960h (~95M)，基于 CTC 后验的 GOP-lite；
                     更贴近真实"发音是否标准"，但 CPU 上每句多 1–3 秒
    """
    if scorer is None:
        scorer = os.getenv("PRONUNCIATION_SCORER", SCORER_HEURISTIC)

    ref_norm_words = [_normalize(w) for w in ref_text.split()]
    clean_path     = preprocess_audio(audio_path)

    try:
        transcript = transcribe(clean_path)

        # 空语音兜底
        if not transcript["words"]:
            return {
                "overall_score":    0.0,
                "accuracy_score":   0.0,
                "fluency_score":    0.0,
                "transcript":       "",
                "words_per_minute": 0.0,
                "weak_words":       [],
                "word_scores":      [{"word": w, "score": 0, "tag": "missing"}
                                     for w in ref_norm_words],
                "feedback":         ["未检测到有效语音，请重录"],
                "scorer":           scorer,
                "error":            "no_speech_detected",
            }

        analyzer    = _AudioAnalyzer(clean_path)
        accuracy    = score_accuracy(ref_norm_words, transcript)
        fluency     = score_fluency(analyzer, transcript["words"])
        pron_scores = _compute_pron_scores(
            transcript["words"], analyzer, clean_path, scorer
        )
        word_scores = score_words(ref_norm_words, transcript, pron_scores)
    finally:
        if os.path.exists(clean_path):
            os.remove(clean_path)

    weak_words = _collect_weak_words(word_scores)
    overall    = accuracy["accuracy_score"] * 0.6 + fluency["fluency_score"] * 0.4

    return {
        "overall_score":    round(overall, 1),
        "accuracy_score":   accuracy["accuracy_score"],
        "fluency_score":    fluency["fluency_score"],
        "transcript":       transcript["text"],
        "words_per_minute": fluency["words_per_minute"],
        "weak_words":       weak_words,
        "word_scores":      word_scores,
        "feedback":         _generate_feedback(fluency, weak_words),
        "scorer":           scorer,
    }


if __name__ == "__main__":
    ref = "Hello! This is Kokoro TTS running on Windows."

    # 默认启发式方案
    report = evaluate(ref, "kokoro.wav")
    print("[heuristic]", report)

    # wav2vec2 方案
    report = evaluate(ref, "kokoro.wav", scorer="wav2vec2")
    print("[wav2vec2]", report)
