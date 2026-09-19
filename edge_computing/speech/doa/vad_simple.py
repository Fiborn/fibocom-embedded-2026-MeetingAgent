# -*- coding: utf-8 -*-
"""简单能量 VAD —— 对应 PDF 中的 vad_active 开关"""
import numpy as np

from config_doa import VAD_RMS_THRESHOLD


def rms(samples):
    if len(samples) == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.asarray(samples, dtype=np.float64) ** 2)))


def is_speech_active(channels, threshold=VAD_RMS_THRESHOLD):
    """
    channels: (num_mics, samples) float32
    任一通道超过阈值即认为有人说话。
    """
    for ch in channels:
        if rms(ch) >= threshold:
            return True
    return False
