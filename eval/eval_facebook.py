"""
Wav2Vec2 英语口语评测模块

功能模式：
  模式一（转写）  : 将音频识别为 IPA 音素序列并输出
  模式二（评测）  : 参考文本 + 音频 → 音素级评分 + 韵律分析 + 反馈

合并自：facebook_asr.py + eval_facebook_phoneme.py
"""

import io
import json
import os
import re
import sys
import time
from argparse import ArgumentParser

import numpy as np
import requests
import soundfile as sf
import torch

# ─── 项目路径引导（直接运行脚本时可用） ──────────────
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PARENT_DIR = os.path.dirname(_SCRIPT_DIR)
if _PARENT_DIR not in sys.path:
    sys.path.insert(0, _PARENT_DIR)

# ─── espeak-ng DLL 路径（在 phonemizer/transformers 之前设置） ──
_ESPEAK_DLL = r"C:\Program Files\eSpeak NG\libespeak-ng.dll"
if os.path.exists(_ESPEAK_DLL):
    os.environ["PHONEMIZER_ESPEAK_LIBRARY"] = _ESPEAK_DLL
elif "PHONEMIZER_ESPEAK_LIBRARY" not in os.environ:
    for p in [
        r"C:\Program Files\eSpeak NG\libespeak-ng.dll",
        r"C:\Program Files (x86)\eSpeak NG\libespeak-ng.dll",
        "/usr/lib/libespeak-ng.so",
        "/usr/local/lib/libespeak-ng.so",
    ]:
        if os.path.exists(p):
            os.environ["PHONEMIZER_ESPEAK_LIBRARY"] = p
            break

# 设置 Hugging Face 镜像源（如需）
# os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from phonemizer import phonemize
from phonemizer.separator import Separator
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC

from eval.eval_sentence import (
    AudioAnalyzer,
    PAUSE_THRESHOLD,
    preprocess_audio,
    score_fluency,
    tokenize,
    transcribe as whisper_transcribe,
)


# ══════════════════════════════════════════════════════════
# Part 1 — Wav2Vec2 IPA 音素转写
# ══════════════════════════════════════════════════════════

_MODEL: Wav2Vec2ForCTC | None = None
_PROCESSOR: Wav2Vec2Processor | None = None
_MODEL_NAME = "facebook/wav2vec2-xlsr-53-espeak-cv-ft"
_TARGET_SR = 16000


def _get_model():
    """首次调用时加载模型，后续复用（全局缓存）"""
    global _MODEL, _PROCESSOR
    if _MODEL is None:
        print(f"正在加载 Wav2Vec2 模型 ({_MODEL_NAME})，请稍候...")
        _PROCESSOR = Wav2Vec2Processor.from_pretrained(_MODEL_NAME)
        _MODEL = Wav2Vec2ForCTC.from_pretrained(_MODEL_NAME)
        _MODEL.eval()
        print("模型加载完成")
    return _MODEL, _PROCESSOR


def _decode_phonemes(waveform: np.ndarray, sample_rate: int) -> str:
    """
    对波形进行 Wav2Vec2 推理，返回空格分隔的 IPA 音素字符串。

    Args:
        waveform: 1D numpy array, 取值范围 [-1, 1]
        sample_rate: 采样率（会被重采样到 16kHz）

    Returns:
        IPA 音素字符串，如 "h ˈɛ l oʊ w ˈɜː l d"
    """
    model, processor = _get_model()

    if sample_rate != _TARGET_SR:
        import torchaudio
        wav_t = torch.from_numpy(waveform).unsqueeze(0).float()
        resampler = torchaudio.transforms.Resample(
            orig_freq=sample_rate, new_freq=_TARGET_SR
        )
        waveform = resampler(wav_t).squeeze(0).numpy()
        sample_rate = _TARGET_SR

    input_values = processor(
        waveform, return_tensors="pt", sampling_rate=sample_rate,
    ).input_values

    with torch.no_grad():
        logits = model(input_values).logits

    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.batch_decode(predicted_ids)[0]
    return transcription.strip()


def transcribe_from_url(audio_url: str) -> str:
    """从远程 URL 下载音频并转写为 IPA 音素。"""
    print(f"正在下载音频: {audio_url}")
    response = requests.get(audio_url)
    response.raise_for_status()
    audio_bytes = io.BytesIO(response.content)
    waveform, sample_rate = sf.read(audio_bytes)
    if waveform.ndim > 1:
        waveform = waveform.mean(axis=1)
    return _decode_phonemes(waveform, sample_rate)


def transcribe_from_file(audio_path: str) -> str:
    """从本地文件读取音频并转写为 IPA 音素。"""
    waveform, sample_rate = sf.read(audio_path)
    if waveform.ndim > 1:
        waveform = waveform.mean(axis=1)
    return _decode_phonemes(waveform, sample_rate)


def transcribe_from_bytes(audio_bytes: bytes, sample_rate: int = _TARGET_SR) -> str:
    """从原始音频 bytes 转写为 IPA 音素。"""
    buf = io.BytesIO(audio_bytes)
    waveform, sr = sf.read(buf)
    if waveform.ndim > 1:
        waveform = waveform.mean(axis=1)
    return _decode_phonemes(waveform, sr)


# ══════════════════════════════════════════════════════════
# Part 2 — 音素评测引擎
# ══════════════════════════════════════════════════════════

# 中文学习者常见音素混淆：{参考音素: [(错误音素, 描述), ...]}
_COMMON_CONFUSIONS = {
    "θ": [("s", "/θ/ → /s/（如 'think' → 'sink'）"),
          ("f", "/θ/ → /f/（如 'three' → 'free'）")],
    "ð": [("z", "/ð/ → /z/（如 'this' → 'zis'）"),
          ("d", "/ð/ → /d/（如 'that' → 'dat'）")],
    "r": [("l", "/r/ → /l/（如 'right' → 'light'）"),
          ("w", "/r/ → /w/（如 'read' → 'weed'）")],
    "v": [("w", "/v/ → /w/（如 'very' → 'wery'）"),
          ("f", "/v/ → /f/（清音化，如 'live' → 'lif'）")],
    "ŋ": [("n", "/ŋ/ → /n/（如 'sing' → 'sin'）"),
          ("ŋg", "/ŋ/ → /ŋg/（如 'sing' → 'sing-g'）")],
    "ʃ": [("s", "/ʃ/ → /s/（如 'ship' → 'sip'）")],
    "æ": [("ɛ", "/æ/ → /ɛ/（如 'bad' → 'bed'）"),
          ("e", "/æ/ → /e/（如 'cat' → 'cet'）")],
    "ʌ": [("ɑ", "/ʌ/ → /ɑ/（如 'cup' → 'cop'）")],
    "ɪ": [("i", "/ɪ/ → /iː/（如 'ship' → 'sheep'）")],
    "b": [("p", "/b/ → /p/（清音化，如 'cab' → 'cap'）")],
    "d": [("t", "/d/ → /t/（清音化，如 'bad' → 'bat'）")],
    "g": [("k", "/g/ → /k/（清音化，如 'bag' → 'back'）")],
    "z": [("s", "/z/ → /s/（清音化，如 'zoo' → 'sue'）")],
    "tʃ": [("ʃ", "/tʃ/ → /ʃ/（如 'check' → 'sheck'）"),
           ("ts", "/tʃ/ → /ts/（如 'cheese' → 'tseese'）")],
    "dʒ": [("ʒ", "/dʒ/ → /ʒ/（如 'jump' → 'ʒump'）")],
}


def strip_stress(ph: str) -> str:
    """移除 IPA 重音/声调标记，保留纯音素用于比对。"""
    return re.sub(r"[ˈˌˌ0-9]", "", ph).strip()


def tokenize_phonemes(phoneme_str: str) -> list[str]:
    """将空格分隔的 IPA 音素字符串拆分为列表。"""
    return phoneme_str.strip().split()


def text_to_phonemes(text: str) -> dict:
    """
    将参考文本转为 IPA 音素序列，保留词边界。

    Returns:
        {"flat": [...], "words": [...], "boundaries": [(s,e), ...]}
    """
    words = tokenize(text)
    word_phonemes = []
    for w in words:
        try:
            ph = phonemize(w, language="en-us",
                           separator=Separator(phone=" ", word=" | "))
            ph_list = [p for p in ph.strip().split() if p != "|"]
            if not ph_list:
                ph_list = [" "]
            word_phonemes.append((w, ph_list))
        except Exception:
            word_phonemes.append((w, [" "]))

    flat = []
    boundaries = []
    offset = 0
    for _, phs in word_phonemes:
        flat.extend(phs)
        boundaries.append((offset, offset + len(phs)))
        offset += len(phs)

    return {"flat": flat, "words": [w for w, _ in word_phonemes],
            "boundaries": boundaries}


def align_phonemes(ref: list[str], hyp: list[str]) -> list[tuple]:
    """
    编辑距离（Levenshtein）逐音素对齐。

    Returns:
        [(ref_ph, hyp_ph, status), ...]
        status ∈ {"correct", "substitution", "insertion", "deletion"}
    """
    m, n = len(ref), len(hyp)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1): dp[i][0] = i
    for j in range(n + 1): dp[0][j] = j

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if strip_stress(ref[i - 1]) == strip_stress(hyp[j - 1]) else 1
            dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1,
                           dp[i - 1][j - 1] + cost)

    aligned = []
    i, j = m, n
    while i > 0 or j > 0:
        if i > 0 and j > 0:
            cost = 0 if strip_stress(ref[i - 1]) == strip_stress(hyp[j - 1]) else 1
            if dp[i][j] == dp[i - 1][j - 1] + cost:
                status = "correct" if cost == 0 else "substitution"
                aligned.append((ref[i - 1], hyp[j - 1], status))
                i -= 1; j -= 1
            elif dp[i][j] == dp[i - 1][j] + 1:
                aligned.append((ref[i - 1], None, "deletion"))
                i -= 1
            else:
                aligned.append((None, hyp[j - 1], "insertion"))
                j -= 1
        elif i > 0:
            aligned.append((ref[i - 1], None, "deletion"))
            i -= 1
        else:
            aligned.append((None, hyp[j - 1], "insertion"))
            j -= 1
    return list(reversed(aligned))


def score_phonemes(alignment: list[tuple]) -> dict:
    """
    从对齐结果计算音素级指标：accuracy / precision / recall / F1 / 混淆矩阵。
    """
    correct = substitutions = insertions = deletions = 0
    confusion = {}

    for ref_ph, hyp_ph, status in alignment:
        if status == "correct":
            correct += 1
        elif status == "substitution":
            substitutions += 1
            rn = strip_stress(ref_ph) if ref_ph else "∅"
            hn = strip_stress(hyp_ph) if hyp_ph else "∅"
            confusion.setdefault(rn, {}).setdefault(hn, 0)
            confusion[rn][hn] += 1
        elif status == "insertion":
            insertions += 1
            hn = strip_stress(hyp_ph) if hyp_ph else "∅"
            confusion.setdefault("_INS_", {}).setdefault(hn, 0)
            confusion["_INS_"][hn] += 1
        elif status == "deletion":
            deletions += 1
            rn = strip_stress(ref_ph) if ref_ph else "∅"
            confusion.setdefault(rn, {}).setdefault("_DEL_", 0)
            confusion[rn]["_DEL_"] += 1

    total_ref = correct + substitutions + deletions
    total_hyp = correct + substitutions + insertions
    acc = correct / max(total_ref, 1)
    prec = correct / max(total_hyp, 1)
    rec = correct / max(total_ref, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-6)

    return dict(accuracy=round(acc, 4), precision=round(prec, 4),
                recall=round(rec, 4), f1=round(f1, 4),
                correct=correct, substitutions=substitutions,
                insertions=insertions, deletions=deletions,
                confusion_matrix=confusion)


def detect_error_patterns(confusion_matrix: dict, top_n: int = 5) -> list[str]:
    """从混淆矩阵中识别常见中文学习者发音错误模式。"""
    patterns = []
    for ref_norm, hyp_dict in confusion_matrix.items():
        for hyp_norm, count in hyp_dict.items():
            if ref_norm in ("_INS_", "∅") or hyp_norm in ("_DEL_", "∅"):
                continue
            desc = None
            if ref_norm in _COMMON_CONFUSIONS:
                for err_ph, d in _COMMON_CONFUSIONS[ref_norm]:
                    if err_ph == hyp_norm:
                        desc = d; break
            if desc:
                patterns.append((count, f"{desc}，共出现 {count} 次"))
            else:
                patterns.append((count,
                    f"/{ref_norm}/ → /{hyp_norm}/ 替换，共出现 {count} 次"))
    patterns.sort(key=lambda x: -x[0])
    return [d for _, d in patterns[:top_n]]


def _score_word_from_phonemes(seg: list[tuple]) -> tuple:
    """单个词在音素对齐片段上的得分 (score_0_100, tag)。"""
    if not seg:
        return 0.0, "missing"
    correct = sum(1 for _, _, s in seg if s == "correct")
    ref_cnt = sum(1 for r, _, _ in seg if r is not None)
    if ref_cnt == 0:
        return 0.0, "inserted" if all(r is None for r, _, _ in seg) else 0.0
    if all(h is None for _, h, _ in seg):
        return 0.0, "missing"
    accuracy = correct / ref_cnt
    score = accuracy ** 0.5 * 100.0
    tag = "good" if score >= 85 else "ok" if score >= 70 else "substituted" \
          if sum(1 for _, _, s in seg if s == "substitution") > 0 else "poor"
    return round(score, 1), tag


def map_alignment_to_words(ref_text: str, alignment: list[tuple]) -> list[dict]:
    """将音素对齐映射到词级评分。"""
    ph_info = text_to_phonemes(ref_text)
    word_list = ph_info["words"]
    boundaries = ph_info["boundaries"]

    ref_idx_to_aligned = []
    ref_cnt = 0
    for entry in alignment:
        if entry[0] is not None:
            ref_idx_to_aligned.append((ref_cnt, entry))
            ref_cnt += 1
        else:
            ref_idx_to_aligned.append((None, entry))

    word_scores = []
    for word, (start, end) in zip(word_list, boundaries):
        seg = [e for idx, e in ref_idx_to_aligned
               if idx is not None and start <= idx < end]
        score_val, tag_val = _score_word_from_phonemes(seg) if seg else (0.0, "missing")
        word_scores.append({
            "word": word, "hyp_word": word,
            "score": score_val, "tag": tag_val,
            "phoneme_detail": [(r or "∅", h or "∅", s) for r, h, s in seg],
        })
    return word_scores


def _collect_weak_words(word_scores: list, limit: int = 5) -> list[str]:
    """提取低分词。"""
    seen, out = set(), []
    for w in word_scores:
        if w["tag"] in ("poor", "substituted", "missing") and w["word"] not in seen:
            seen.add(w["word"])
            out.append(w["word"])
            if len(out) >= limit:
                break
    return out


def _generate_phoneme_feedback(weak_words: list[str], error_patterns: list[str],
                                fluency: dict) -> list[str]:
    """生成中文音素级反馈。"""
    tips = []
    if weak_words:
        tips.append(f"发音需改进的词：{', '.join(weak_words[:3])}")
    if error_patterns:
        tips.extend(error_patterns[:3])
    wpm = fluency["words_per_minute"]
    if wpm < 100:
        tips.append("语速偏慢，尝试减少停顿")
    elif wpm > 180:
        tips.append("语速偏快，注意咬字清晰")
    if fluency.get("pause_count", 0) > 5:
        tips.append(f"停顿次数较多（{fluency['pause_count']}次），注意句子连贯性")
    if fluency.get("pitch_variation_st", 10) < 2.0:
        tips.append("语调变化较少，尝试更有表现力地朗读")
    return tips if tips else ["发音流畅，继续保持！"]


def evaluate(ref_text: str, audio_path: str, *, scorer: str | None = None) -> dict:
    """
    音素级英语口语评测主入口。
    返回值与 eval_sentence.evaluate() 完全一致，方便交叉对比。

    Returns:
        包含 overall/accuracy/pronunciation/fluency 等字段的 dict
    """
    clean_path = preprocess_audio(audio_path)
    try:
        hyp_phonemes_str = transcribe_from_file(clean_path)
        hyp_phonemes = tokenize_phonemes(hyp_phonemes_str)
        ph_info = text_to_phonemes(ref_text)
        ref_phonemes = ph_info["flat"]

        alignment = align_phonemes(ref_phonemes, hyp_phonemes)
        scores = score_phonemes(alignment)
        error_patterns = detect_error_patterns(scores["confusion_matrix"])
        word_scores = map_alignment_to_words(ref_text, alignment)

        whisper_result = whisper_transcribe(clean_path)
        analyzer = AudioAnalyzer(clean_path)
        fluency = score_fluency(analyzer, whisper_result["words"])

        matched = [w["score"] for w in word_scores
                   if w["tag"] not in ("missing", "inserted")]
        pronunciation_score = round(sum(matched) / len(matched), 1) if matched else 0.0
        accuracy_score = round(scores["accuracy"] * 100.0, 1)
        weak_words = _collect_weak_words(word_scores)
        feedback = _generate_phoneme_feedback(weak_words, error_patterns, fluency)
    finally:
        if os.path.exists(clean_path):
            os.remove(clean_path)

    overall = accuracy_score * 0.35 + pronunciation_score * 0.35 + fluency["fluency_score"] * 0.30
    return dict(overall_score=round(overall, 1),
                accuracy_score=accuracy_score,
                pronunciation_score=pronunciation_score,
                fluency_score=fluency["fluency_score"],
                transcript=hyp_phonemes_str,
                words_per_minute=fluency["words_per_minute"],
                weak_words=weak_words, word_scores=word_scores,
                feedback=feedback, scorer="phoneme_wav2vec2",
                _phoneme_scores=scores, _error_patterns=error_patterns,
                _alignment=alignment)


# ══════════════════════════════════════════════════════════
# CLI — 两种模式
# ══════════════════════════════════════════════════════════

if __name__ == "__main__":
    if "--ref" in sys.argv:
        # ── 评测模式 ──
        p = ArgumentParser(description="音素级英语口语评测 (Wav2Vec2)")
        p.add_argument("--ref", required=True, help="参考文本")
        p.add_argument("--audio", required=True, help="音频文件路径")
        p.add_argument("--verbose", action="store_true", help="输出音素对齐详情")
        args = p.parse_args()

        start = time.time()
        report = evaluate(args.ref, args.audio)
        elapsed = time.time() - start

        out = {k: v for k, v in report.items() if not k.startswith("_")}
        print(json.dumps(out, ensure_ascii=False, indent=2))

        if args.verbose:
            ps = report["_phoneme_scores"]
            print(f"\n── 音素详情 ──")
            print(f"参考音素: {ps['correct']+ps['substitutions']+ps['deletions']} 个")
            print(f"正确: {ps['correct']}, 替换: {ps['substitutions']}, "
                  f"插入: {ps['insertions']}, 删除: {ps['deletions']}")
            print(f"准确率: {ps['accuracy']:.2%}, F1: {ps['f1']:.2%}")
            if report["_error_patterns"]:
                print("\n错误模式:")
                for p in report["_error_patterns"]:
                    print(f"  • {p}")
            if report["_alignment"]:
                print(f"\n音素对齐 (前 50 项):")
                for i, (r, h, s) in enumerate(report["_alignment"][:50]):
                    print(f"  {i:3d}. ref={r or '∅':>6s}  hyp={h or '∅':>6s}  [{s}]")
        print(f"\n耗时: {elapsed:.2f}s")

    elif len(sys.argv) > 1:
        # ── 转写模式 ──
        arg = sys.argv[1]
        if arg.startswith("http://") or arg.startswith("https://"):
            result = transcribe_from_url(arg)
        else:
            result = transcribe_from_file(arg)
        print("-" * 50)
        print("识别结果 (IPA 音素):")
        print(result)
        print("-" * 50)

    else:
        import tempfile
        import requests

        # ── 可替换配置 ──
        ref = "What dishes do you plan to have for Dinner."
        audio_url = "https://ielts-prod.oss-cn-hangzhou.aliyuncs.com/audio/2b47783f50eb44a2ba55a877fd9ffae9.wav"

        # 下载音频到临时文件
        resp = requests.get(audio_url)
        resp.raise_for_status()
        fd, audio = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        with open(audio, "wb") as f:
            f.write(resp.content)

        start_time = time.time()
        report = evaluate(ref, audio)
        end_time = time.time()
        os.remove(audio)

        out = {k: v for k, v in report.items() if not k.startswith("_")}
        print(json.dumps(out, ensure_ascii=False, indent=2))
        print(f"\n耗时: {end_time - start_time:.2f}s")