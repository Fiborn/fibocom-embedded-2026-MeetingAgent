# -*- coding: utf-8 -*-
"""GCC-PHAT 广义互相关 —— PDF 核心 TDOA 算法"""
import numpy as np


def gcc_phat(sig_ref, sig_other, sample_rate, max_tau_sec=None):
    """
    计算 sig_other 相对 sig_ref 的时延（秒）及归一化峰值置信度。

    Returns:
        tau_sec: 时延（正数表示 other 比 ref 晚到）
        confidence: 主峰锐度 [0, 1]
    """
    sig_ref = np.asarray(sig_ref, dtype=np.float64)
    sig_other = np.asarray(sig_other, dtype=np.float64)
    n = len(sig_ref) + len(sig_other)

    SIG1 = np.fft.rfft(sig_ref, n=n)
    SIG2 = np.fft.rfft(sig_other, n=n)
    cross = SIG1 * np.conj(SIG2)
    cross /= np.abs(cross) + 1e-12

    cc = np.fft.irfft(cross, n=n)
    max_shift = n // 2
    cc = np.concatenate((cc[-max_shift:], cc[: max_shift + 1]))

    if max_tau_sec is not None:
        limit = int(max_tau_sec * sample_rate)
        center = max_shift
        cc_window = cc[center - limit : center + limit + 1]
        peak_idx = int(np.argmax(np.abs(cc_window)))
        shift = peak_idx - limit
    else:
        shift = int(np.argmax(np.abs(cc))) - max_shift

    tau_sec = shift / float(sample_rate)

    peak = float(np.max(np.abs(cc)))
    mean = float(np.mean(np.abs(cc))) + 1e-12
    confidence = float(np.clip(peak / (mean * 8.0), 0.0, 1.0))

    return tau_sec, confidence


def pairwise_tdoas(channels, sample_rate, max_tau_sec=0.001):
    """
    多通道音频 (num_mics, samples) → 相对 Mic0 的 TDOA 列表与平均置信度。
    """
    ref = channels[0]
    taus = []
    confidences = []
    for i in range(1, channels.shape[0]):
        tau, conf = gcc_phat(ref, channels[i], sample_rate, max_tau_sec=max_tau_sec)
        taus.append(tau)
        confidences.append(conf)
    return np.array(taus, dtype=np.float64), float(np.mean(confidences))
