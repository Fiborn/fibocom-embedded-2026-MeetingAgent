# -*- coding: utf-8 -*-
"""将独立 doa/ 引擎接入会议主链路：段级方位估计 + 实时兜底。"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

import config

_DOA_ROOT = Path(__file__).resolve().parent.parent / "doa"
if str(_DOA_ROOT) not in sys.path:
    sys.path.insert(0, str(_DOA_ROOT))


class DoaBridge:
    """封装 DoaEngine；无多通道或禁用时安全降级。"""

    def __init__(self):
        self.enabled = bool(getattr(config, "SPEAKER_USE_DOA", True))
        self.min_conf = float(getattr(config, "SPEAKER_DOA_MIN_CONF", 0.6))
        self.min_sec = float(getattr(config, "SPEAKER_DOA_MIN_SEC", 0.35))
        self._engine = None
        self._last = None  # (azimuth, confidence)
        if not self.enabled:
            return
        try:
            # 显式加载 doa/engine.py，避免与其它名为 engine 的模块冲突
            import importlib.util

            engine_path = _DOA_ROOT / "engine.py"
            spec = importlib.util.spec_from_file_location(
                "sc171_doa_engine", engine_path
            )
            if spec is None or spec.loader is None:
                raise ImportError("无法加载 {}".format(engine_path))
            mod = importlib.util.module_from_spec(spec)
            # 保证 doa 包内相对依赖可解析
            if str(_DOA_ROOT) not in sys.path:
                sys.path.insert(0, str(_DOA_ROOT))
            spec.loader.exec_module(mod)
            self._engine = mod.DoaEngine(
                sample_rate=int(getattr(config, "SAMPLE_RATE", 16000))
            )
            print("[DOA] 已接入主链路 (GCC-PHAT)", flush=True)
        except Exception as exc:
            self.enabled = False
            self._engine = None
            print("[DOA] 初始化失败，已降级为纯声纹: {}".format(exc), flush=True)

    @property
    def last_azimuth(self):
        return self._last

    def feed_block(self, multi):
        """实时喂入 (num_mics, samples)，更新最近方位兜底。"""
        if not self.enabled or self._engine is None or multi is None:
            return None
        multi = np.asarray(multi, dtype=np.float32)
        if multi.ndim != 2 or multi.shape[0] < 2 or multi.shape[1] < 8:
            return None
        try:
            result = self._engine.feed(multi)
        except Exception:
            return None
        return self._consume_result(result)

    def analyze_segment(self, multi):
        """
        对 VAD 段对应的多通道切片做方位估计。
        返回 (azimuth_deg, confidence) 或 None（低置信/不可用）。
        """
        if not self.enabled or self._engine is None or multi is None:
            return self._last if self._last and self._last[1] >= self.min_conf else None
        multi = np.asarray(multi, dtype=np.float32)
        if multi.ndim != 2 or multi.shape[0] < 2:
            return None
        duration = multi.shape[1] / float(getattr(config, "SAMPLE_RATE", 16000))
        if duration < self.min_sec:
            return None
        try:
            result = self._engine.analyze(multi)
        except Exception:
            return self._fallback()
        parsed = self._consume_result(result)
        if parsed is not None:
            return parsed
        return self._fallback()

    def _fallback(self):
        if self._last is not None and self._last[1] >= self.min_conf:
            return self._last
        return None

    def _consume_result(self, result):
        if result is None or not getattr(result, "vad_active", False):
            return None
        sd = getattr(result, "source_data", None)
        if sd is None:
            return None
        conf = float(getattr(sd, "confidence", 0.0) or 0.0)
        az = float(getattr(sd, "azimuth", 0.0) or 0.0)
        self._last = (az, conf)
        if conf < self.min_conf:
            return None
        return (az, conf)

    def reset(self):
        self._last = None
        if self._engine is not None:
            self._engine._buffer = np.zeros((0, 0), dtype=np.float32)
