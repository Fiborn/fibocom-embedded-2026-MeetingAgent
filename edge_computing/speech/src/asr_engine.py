# -*- coding: utf-8 -*-
"""中文语音识别（ASR）"""
import json
import re

import numpy as np
import sherpa_onnx

import config
from audio_utils import enhance_for_asr, normalize_for_asr, resample, TARGET_SR
from asr_postprocess import post_correct_transcript


def _parse_asr_result(result):
    if result is None:
        return ""
    for attr in ("text", "asr_text", "sentence"):
        val = getattr(result, attr, None)
        if val is not None:
            text = str(val).strip()
            if text and text.lower() not in ("none", "null"):
                return text
    raw = str(result).strip()
    if not raw or raw.lower() in ("none", ""):
        return ""
    if raw.startswith("{"):
        try:
            obj = json.loads(raw)
            if isinstance(obj, dict):
                text = str(obj.get("text", "")).strip()
                return text
        except (json.JSONDecodeError, TypeError, ValueError):
            return ""
    return raw


def is_valid_transcript(text):
    text = (text or "").strip()
    if not text or text.startswith("{"):
        return False
    # 单字中文也有效（“好/是/行”），勿因 len<2 丢掉
    if re.fullmatch(r"[\W\d_]+", text, flags=re.UNICODE):
        return False
    return bool(re.search(r"[\u4e00-\u9fffA-Za-z]", text))


class AsrEngine:
    def __init__(self):
        model = config.ASR_MODEL
        tokens = config.ASR_TOKENS
        missing = [str(p) for p in (model, tokens) if not p.exists()]
        if missing:
            raise FileNotFoundError(
                "ASR 模型缺失:\n  {}\n"
                "请运行: python3 scripts/download_model.py".format(
                    "\n  ".join(missing)
                )
            )
        self._model_path = str(config.ASR_MODEL)
        self._tokens_path = str(config.ASR_TOKENS)
        self._call_count = 0
        self._recognizer = None
        self._recreate_recognizer()

    def _recreate_recognizer(self):
        if self._recognizer is not None:
            del self._recognizer
        import gc
        gc.collect()
        self._recognizer = sherpa_onnx.OfflineRecognizer.from_paraformer(
            paraformer=self._model_path,
            tokens=self._tokens_path,
            num_threads=config.ASR_NUM_THREADS,
            sample_rate=TARGET_SR,
            feature_dim=80,
            decoding_method="greedy_search",
            debug=False,
        )

    @property
    def recognizer(self):
        return self._recognizer

    def transcribe(self, samples, sample_rate=16000, already_normalized=False):
        """音频 float32 数组 → 文字"""
        if len(samples) == 0:
            return ""

        samples = np.ascontiguousarray(samples, dtype=np.float32)
        if sample_rate != TARGET_SR:
            samples = resample(samples, sample_rate, TARGET_SR)
        # 上游已 enhance+normalize 时不要再做一遍，避免双重去直流/再放大
        if not already_normalized:
            samples = enhance_for_asr(samples, TARGET_SR)
            target_rms = getattr(config, "ASR_TARGET_RMS", 0.10)
            max_gain = float(getattr(config, "ASR_MAX_GAIN", 12.0))
            samples = normalize_for_asr(
                samples, target_rms=target_rms, max_gain=max_gain
            )

        min_sec = float(getattr(config, "ASR_MIN_SEC", 0.40))
        if len(samples) < int(min_sec * TARGET_SR):
            return ""

        stream = self._recognizer.create_stream()
        stream.accept_waveform(TARGET_SR, samples)
        if hasattr(stream, "input_finished"):
            stream.input_finished()
        self._recognizer.decode_stream(stream)

        text = _parse_asr_result(stream.result)
        del stream

        self._call_count += 1
        recreate_every = int(getattr(config, "ASR_RECREATE_EVERY", 120) or 0)
        if recreate_every > 0 and self._call_count % recreate_every == 0:
            self._recreate_recognizer()

        if getattr(config, "ASR_POST_CORRECT", True):
            text = post_correct_transcript(text)

        if not is_valid_transcript(text):
            if config.DEBUG and text:
                print("[DEBUG] ASR 无效结果已丢弃: {!r}".format(text[:80]))
            return ""

        if config.DEBUG and not text:
            print("[DEBUG] ASR 返回空，片段时长 {:.2f}s".format(
                len(samples) / TARGET_SR
            ))
        return text
