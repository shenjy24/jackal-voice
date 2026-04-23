import re, os
import numpy as np
import soundfile as sf
import editdistance
import parselmouth
from faster_whisper import WhisperModel

model = WhisperModel("base.en", device="cpu", compute_type="int8")


# ─── 工具函数 ────────────────────────────────────────────

def _normalize(word: str) -> str:
    """去掉标点、转小写"""
    return re.sub(r"[^\w]", "", word).lower()

def _normalize_text(text: str) -> str:
    """对整句做 normalize，用于 WER 计算"""
    return " ".join(_normalize(w) for w in text.split())


# ─── 音频预处理 ──────────────────────────────────────────

def preprocess_audio(audio_path: str) -> str:
    """
    标准化为 16kHz 单声道 PCM_16 WAV
    返回处理后的文件路径（供 Whisper 和 Praat 共用）
    """
    import librosa
    audio, sr = librosa.load(audio_path, sr=16000, mono=True)
    out_path = audio_path.rsplit(".", 1)[0] + "_16k.wav"
    sf.write(out_path, audio, 16000, subtype="PCM_16")
    return out_path


# ─── 转录 ────────────────────────────────────────────────

def transcribe(audio_path: str) -> dict:
    """转录并返回词级时间戳，word 已 normalize"""
    segments, _ = model.transcribe(
        audio_path,
        word_timestamps=True,
        language="en",
        beam_size=5,
    )

    words = []
    full_text = ""
    for segment in segments:
        for word in segment.words:
            raw = word.word.strip()
            words.append({
                "word":      raw,
                "word_norm": _normalize(raw),   # ← 提前 normalize
                "start":     word.start,
                "end":       word.end,
                "prob":      word.probability,
            })
            full_text += word.word

    return {"text": full_text.strip(), "words": words}


# ─── 准确度评分 ──────────────────────────────────────────

def score_accuracy(ref_norm_words: list[str], hyp_result: dict) -> dict:
    """
    入参 ref_norm_words：已 normalize 的参考词列表
    hyp 侧直接用 word_norm 字段
    """
    hyp_norm_words = [w["word_norm"] for w in hyp_result["words"]]

    distance = editdistance.eval(ref_norm_words, hyp_norm_words)
    wer      = distance / max(len(ref_norm_words), 1)
    accuracy = max(0, round((1 - wer) * 100, 1))

    # 发音弱的词直接用 normalize 后的词展示
    weak_words = [
        w["word_norm"] for w in hyp_result["words"]
        if w["prob"] < 0.8
    ]

    return {
        "accuracy_score": accuracy,
        "wer":            round(wer, 3),
        "weak_words":     weak_words,
    }


# ─── 流利度评分 ──────────────────────────────────────────

def score_fluency(audio_path: str, word_timestamps: list) -> dict:
    snd            = parselmouth.Sound(audio_path)
    total_duration = snd.duration

    # 语速
    total_words     = len(word_timestamps)
    speech_duration = sum(w["end"] - w["start"] for w in word_timestamps)
    wpm             = (total_words / speech_duration * 60) if speech_duration > 0 else 0

    # 停顿
    pauses = [
        word_timestamps[i]["start"] - word_timestamps[i-1]["end"]
        for i in range(1, len(word_timestamps))
        if word_timestamps[i]["start"] - word_timestamps[i-1]["end"] > 0.25
    ]
    pause_count = len(pauses)
    avg_pause   = float(np.mean(pauses)) if pauses else 0.0

    # 音调
    pitch_values = snd.to_pitch().selected_array["frequency"]
    pitch_values = pitch_values[pitch_values > 0]
    pitch_std    = float(np.std(pitch_values)) if len(pitch_values) > 0 else 0.0

    return {
        "fluency_score":      _calc_fluency_score(wpm, pause_count, avg_pause, total_duration),
        "words_per_minute":   round(wpm, 1),
        "pause_count":        pause_count,
        "avg_pause_duration": round(avg_pause, 2),
        "pitch_variation":    round(pitch_std, 1),
    }

def _speed_to_score(wpm: float) -> float:
    if 120 <= wpm <= 150:   return 100.0
    if wpm < 80 or wpm > 200: return 40.0
    if wpm < 120:           return 40 + (wpm - 80) * (60 / 40)
    return 100 - (wpm - 150) * (60 / 50)

def _calc_fluency_score(wpm, pause_count, avg_pause, duration) -> float:
    score = 100.0
    pause_rate = pause_count / max(duration, 1)
    if pause_rate > 2:  score -= min(30, (pause_rate - 2) * 10)
    if avg_pause > 1.0: score -= min(20, (avg_pause - 1.0) * 15)
    if wpm < 100 or wpm > 180: score -= 15
    return max(0, round(score, 1))


# ─── 词级评分 ────────────────────────────────────────────

def score_words(ref_norm_words: list[str], hyp_result: dict) -> list:
    """入参直接使用已 normalize 的词列表"""
    aligned = _align_words(ref_norm_words, hyp_result["words"])

    result = []
    for ref_w, hyp_w in aligned:
        if hyp_w is None:
            result.append({"word": ref_w, "score": 0, "tag": "missing"})
        elif ref_w is None:
            result.append({"word": hyp_w["word_norm"], "score": 0, "tag": "inserted"})
        else:
            prob  = hyp_w.get("prob", 1.0)
            match = ref_w == hyp_w["word_norm"]   # 两侧都已 normalize，直接比较
            if not match:
                score, tag = round(prob * 100 * 0.3), "wrong"
            elif prob >= 0.9:
                score, tag = 100, "good"
            elif prob >= 0.7:
                score, tag = round(prob * 100), "ok"
            else:
                score, tag = round(prob * 100), "poor"
            result.append({"word": ref_w, "hyp_word": hyp_w["word_norm"], "score": score, "tag": tag})
    return result

def _similar(a: str, b: str) -> bool:
    """两个词足够相似则视为同一个词（只是读错了）"""
    if a == b:
        return True
    # 编辑距离 ≤ 2，且长度差不超过一半
    if abs(len(a) - len(b)) > max(len(a), len(b)) // 2:
        return False
    return editdistance.eval(a, b) <= 2

def _align_words(ref_norm: list[str], hyp: list) -> list:
    """ref 侧传 normalize 后的词，hyp 侧用 word_norm 字段"""
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


# ─── 反馈生成 ────────────────────────────────────────────

def _generate_feedback(accuracy: dict, fluency: dict) -> list[str]:
    tips = []
    if accuracy["accuracy_score"] < 70:
        tips.append(f"发音需改进的词：{', '.join(accuracy['weak_words'][:3])}")
    if fluency["words_per_minute"] < 100:
        tips.append("语速偏慢，尝试减少停顿")
    if fluency["pause_count"] > 5:
        tips.append(f"停顿次数较多（{fluency['pause_count']}次），注意句子连贯性")
    if fluency["pitch_variation"] < 20:
        tips.append("语调变化较少，尝试更有表现力地朗读")
    return tips if tips else ["发音流畅，继续保持！"]


# ─── 主入口 ──────────────────────────────────────────────

def evaluate(ref_text: str, audio_path: str) -> dict:
    # 统一在入口做 normalize，后续所有方法共用
    ref_norm_words = [_normalize(w) for w in ref_text.split()]

    # 音频预处理（统一格式，Whisper 和 Praat 共用同一文件）
    clean_path = preprocess_audio(audio_path)

    try:
        # 转录（hyp 词已含 word_norm 字段）
        transcript = transcribe(clean_path)

        # 各维度评分，均使用 normalize 后的数据
        accuracy    = score_accuracy(ref_norm_words, transcript)
        fluency     = score_fluency(clean_path, transcript["words"])
        word_scores = score_words(ref_norm_words, transcript)
    finally:
        if os.path.exists(clean_path):
            os.remove(clean_path)

    overall = accuracy["accuracy_score"] * 0.6 + fluency["fluency_score"] * 0.4

    return {
        "overall_score":    round(overall, 1),
        "accuracy_score":   accuracy["accuracy_score"],
        "fluency_score":    fluency["fluency_score"],
        "transcript":       transcript["text"],
        "words_per_minute": fluency["words_per_minute"],
        "weak_words":       accuracy["weak_words"],
        "word_scores":      word_scores,
        "feedback":         _generate_feedback(accuracy, fluency),
    }


if __name__ == "__main__":
    ref = "Hello! This is Kokoro TTS running on Windows."
    report = evaluate(ref, "kokoro.wav")
    print(report)