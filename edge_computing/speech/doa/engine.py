# -*- coding: utf-8 -*-
"""DOA 引擎：GCC-PHAT → 角度解算 → PDF 标准 JSON 结构"""
import time

import numpy as np

from array_geometry import (
    azimuth_to_beam_index,
    estimate_azimuth_elevation_from_tdoas,
    quantize_azimuth,
)
from config_doa import (
    ANALYSIS_WINDOW_SEC,
    CONFIDENCE_MIN_FOR_ACTION,
    DEBUG,
    SAMPLE_RATE,
)
from gcc_phat import pairwise_tdoas
from doa_types import DoaResult, SourceData
from vad_simple import is_speech_active


class DoaEngine:
    def __init__(self, sample_rate=SAMPLE_RATE, window_sec=ANALYSIS_WINDOW_SEC):
        self.sample_rate = sample_rate
        self.window_samples = int(window_sec * sample_rate)
        self._buffer = np.zeros((0, 0), dtype=np.float32)

    def feed(self, block):
        """
        block: (num_mics, samples)
        缓冲够一个分析窗口后返回 DoaResult，否则返回 None。
        """
        if block.ndim != 2:
            raise ValueError("feed() 需要 shape (num_mics, samples)")

        if self._buffer.size == 0:
            self._buffer = block.copy()
        else:
            self._buffer = np.concatenate([self._buffer, block], axis=1)

        if self._buffer.shape[1] < self.window_samples:
            return None

        window = self._buffer[:, -self.window_samples :]
        self._buffer = self._buffer[:, -self.window_samples // 2 :]

        return self.analyze(window)

    def analyze(self, channels):
        """对固定窗口做多通道 TDOA + 角度估计"""
        timestamp_ms = int(time.time() * 1000)
        vad_active = is_speech_active(channels)

        if not vad_active:
            return DoaResult.silent(timestamp_ms)

        max_tau = 0.0015
        tdoas, gcc_conf = pairwise_tdoas(channels, self.sample_rate, max_tau_sec=max_tau)
        az_raw, el_raw, geo_score = estimate_azimuth_elevation_from_tdoas(tdoas)

        confidence = float(np.clip(0.5 * gcc_conf + 0.5 * geo_score, 0.0, 1.0))
        azimuth = quantize_azimuth(az_raw)
        elevation = float(el_raw)
        beam_index = azimuth_to_beam_index(azimuth)

        if confidence < CONFIDENCE_MIN_FOR_ACTION:
            if DEBUG:
                print(
                    "[DOA DEBUG] 置信度过低 {:.2f} < {:.2f}".format(
                        confidence, CONFIDENCE_MIN_FOR_ACTION
                    )
                )

        result = DoaResult(
            timestamp=timestamp_ms,
            vad_active=True,
            source_data=SourceData(
                azimuth=round(azimuth, 1),
                elevation=round(elevation, 1),
                confidence=round(confidence, 2),
                beam_index=beam_index,
            ),
        )
        return result
