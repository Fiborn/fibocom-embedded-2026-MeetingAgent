# -*- coding: utf-8 -*-
"""音频预处理：重采样、归一化"""
import numpy as np

TARGET_SR = 16000


def int16_bytes_to_float32(pcm_bytes, channels=1, primary_channel=0):
    samples = np.frombuffer(pcm_bytes, dtype=np.int16)
    if channels > 1:
        samples = samples.reshape(-1, channels)[:, primary_channel]
    return samples.astype(np.float32) / 32768.0


def int16_bytes_to_multichannel_float32(pcm_bytes, channels=1):
    """解析交错 int16 PCM → shape (num_channels, samples) float32。"""
    samples = np.frombuffer(pcm_bytes, dtype=np.int16)
    if channels == 1:
        return samples.astype(np.float32).reshape(1, -1) / 32768.0
    return samples.reshape(-1, channels).T.astype(np.float32) / 32768.0


# 计划/文档别名
int16_bytes_to_float32_multichannel = int16_bytes_to_multichannel_float32


def resample(samples, orig_sr, target_sr=TARGET_SR):
    if orig_sr == target_sr or len(samples) == 0:
        return samples
    import librosa
    return librosa.resample(
        np.ascontiguousarray(samples, dtype=np.float32),
        orig_sr=orig_sr,
        target_sr=target_sr,
    )


def resample_multichannel(multi, orig_sr, target_sr=TARGET_SR):
    """对 (num_channels, samples) 逐通道重采样。"""
    if multi is None or getattr(multi, "size", 0) == 0:
        return multi
    multi = np.asarray(multi, dtype=np.float32)
    if multi.ndim != 2:
        raise ValueError("multi 需要 shape (num_channels, samples)")
    if orig_sr == target_sr:
        return multi
    channels = [resample(multi[i], orig_sr, target_sr) for i in range(multi.shape[0])]
    # 逐通道 resample 长度偶发差 1，对齐到最短，避免 mono/multi 漂移
    min_len = min(len(c) for c in channels)
    if min_len <= 0:
        return np.zeros((multi.shape[0], 0), dtype=np.float32)
    return np.stack([c[:min_len] for c in channels], axis=0)


def align_multi_to_mono(mono, multi):
    """保证 multi 第二维与 mono 长度一致（截断或尾部零填充）。"""
    mono = np.asarray(mono, dtype=np.float32)
    if multi is None:
        return mono, None
    multi = np.asarray(multi, dtype=np.float32)
    if multi.ndim != 2:
        return mono, None
    n = len(mono)
    if multi.shape[1] == n:
        return mono, multi
    if multi.shape[1] > n:
        return mono, multi[:, :n]
    if multi.shape[1] == 0:
        return mono, None
    pad = np.zeros((multi.shape[0], n - multi.shape[1]), dtype=np.float32)
    return mono, np.concatenate([multi, pad], axis=1)


def unpack_read_16k(result):
    """兼容 (mono, level) 与 (mono, level, multi) 返回值。"""
    if result is None:
        return np.array([], dtype=np.float32), 0.0, None
    if not isinstance(result, (tuple, list)):
        raise TypeError("read_16k 应返回 tuple，收到: {}".format(type(result)))
    if len(result) == 3:
        mono, level, multi = result
        if multi is not None and getattr(multi, "size", 0) == 0:
            multi = None
        mono, multi = align_multi_to_mono(mono, multi)
        return mono, float(level), multi
    if len(result) == 2:
        mono, level = result
        return np.asarray(mono, dtype=np.float32), float(level), None
    raise ValueError("read_16k 返回值长度异常: {}".format(len(result)))


def unpack_flush_16k(result):
    """兼容 flush_16k 的 mono 或 (mono, multi)。"""
    if result is None:
        return np.array([], dtype=np.float32), None
    if isinstance(result, tuple):
        if len(result) >= 2:
            mono, multi = result[0], result[1]
            if multi is not None and getattr(multi, "size", 0) == 0:
                multi = None
            mono, multi = align_multi_to_mono(mono, multi)
            return np.asarray(mono, dtype=np.float32), multi
        mono = result[0]
        return np.asarray(mono, dtype=np.float32), None
    return np.asarray(result, dtype=np.float32), None


def normalize(samples, target_peak=0.9, min_peak=0.002):
    """峰值归一化，避免阵列麦输出过小。

    注意：peak 极低时（USB 麦挂死吐近零）禁止放大，否则会把底噪
    抬成“假语音”，VAD 切段后 ASR 只会返回空串。
    """
    if len(samples) == 0:
        return samples
    samples = np.ascontiguousarray(samples, dtype=np.float32)
    peak = float(np.max(np.abs(samples)))
    if peak < 1e-6:
        return samples
    if peak < float(min_peak):
        # 近似静音：原样返回，交给上层 RMS 门控跳过
        return samples
    if peak < 0.05:
        samples = samples / peak * target_peak
    return samples


def prepare_for_embedding(samples, target_peak=0.9, target_rms=0.06, max_gain=20.0):
    """说话人嵌入统一预处理：峰值 + 适度 RMS 增益"""
    samples = normalize(samples, target_peak=target_peak)
    return normalize_for_asr(samples, target_rms=target_rms, max_gain=max_gain)


def normalize_for_asr(samples, target_rms=0.08, max_gain=25.0):
    """按 RMS 增益，让 ASR 输入音量更稳定"""
    if len(samples) == 0:
        return samples
    samples = np.ascontiguousarray(samples, dtype=np.float32)
    level = rms(samples)
    if level < 1e-6:
        return samples
    gain = min(max_gain, target_rms / level)
    if gain <= 1.0:
        # 过响时轻限幅，避免削波失真
        peak = float(np.max(np.abs(samples)))
        if peak > 0.98:
            return samples / peak * 0.95
        return samples
    boosted = samples * gain
    peak = float(np.max(np.abs(boosted)))
    if peak > 0.98:
        boosted = boosted / peak * 0.95
    return boosted


def enhance_for_asr(samples, sample_rate=TARGET_SR):
    """ASR 前端：去直流 + 可选预加重，改善清晰度。"""
    if samples is None or len(samples) == 0:
        return samples
    x = np.ascontiguousarray(samples, dtype=np.float32)
    try:
        import config as _cfg  # noqa: WPS433
    except Exception:
        _cfg = None
    if _cfg is None or bool(getattr(_cfg, "ASR_REMOVE_DC", True)):
        x = x - float(np.mean(x))
    pre = 0.0
    if _cfg is not None:
        pre = float(getattr(_cfg, "ASR_PREEMPHASIS", 0.0) or 0.0)
    if pre > 0.0 and len(x) > 1:
        y = np.empty_like(x)
        y[0] = x[0]
        y[1:] = x[1:] - pre * x[:-1]
        x = y
    peak = float(np.max(np.abs(x))) if len(x) else 0.0
    if peak > 0.99:
        x = x * (0.95 / peak)
    return x


def select_asr_mono(mono, multi=None):
    """从多通道中选能量最佳通道，或对活跃通道做能量加权混合。

    固定 PRIMARY_CHANNEL=0 时，若 ch0 偏弱/静音会严重拖垮识别。
    """
    mono = np.asarray(mono, dtype=np.float32).reshape(-1)
    if multi is None:
        return mono
    multi = np.asarray(multi, dtype=np.float32)
    if multi.ndim != 2 or multi.shape[0] < 2 or multi.shape[1] < 16:
        return mono
    n = min(len(mono), multi.shape[1])
    if n < 16:
        return mono
    multi = multi[:, :n]
    mono = mono[:n]
    try:
        import config as _cfg  # noqa: WPS433
    except Exception:
        _cfg = None
    use_best = True if _cfg is None else bool(getattr(_cfg, "ASR_USE_BEST_CHANNEL", True))
    use_mix = True if _cfg is None else bool(getattr(_cfg, "ASR_CHANNEL_MIX", True))
    if not use_best and not use_mix:
        return mono

    rms_ch = np.sqrt(np.mean(np.square(multi, dtype=np.float64), axis=1))
    best_i = int(np.argmax(rms_ch))
    best_r = float(rms_ch[best_i])
    if best_r < 1e-6:
        return mono

    # 峰值过低的通道视为“增益被关/硬件静音”，绝不参与混合
    peak_ch = np.max(np.abs(multi), axis=1).astype(np.float64)
    alive = (rms_ch >= best_r * 0.35) & (peak_ch >= max(1e-3, float(peak_ch[best_i]) * 0.25))
    if use_mix:
        idx = np.where(alive)[0]
        if idx.size >= 2:
            w = np.square(rms_ch[idx])
            w = w / (np.sum(w) + 1e-12)
            out = np.zeros(n, dtype=np.float64)
            for wi, ci in zip(w, idx):
                out += float(wi) * multi[int(ci)]
            peak = float(np.max(np.abs(out)))
            if peak > 0.98:
                out *= 0.95 / peak
            return out.astype(np.float32)
    if use_best:
        return np.ascontiguousarray(multi[best_i], dtype=np.float32)
    return mono


def rms(samples):
    if len(samples) == 0:
        return 0.0
    return float(np.sqrt(np.mean(samples ** 2)))


def _estimate_formants_lpc(clip, sample_rate, n_formants=2):
    """LPC 共振峰估计，返回最多 n_formants 个频率(Hz)。"""
    try:
        import librosa
    except Exception:
        return []
    if len(clip) < int(0.25 * sample_rate):
        return []
    # 预加重 + 取中段，降低端点效应
    x = np.ascontiguousarray(clip, dtype=np.float32)
    x = np.append(x[0], x[1:] - 0.97 * x[:-1])
    max_len = int(1.2 * sample_rate)
    if len(x) > max_len:
        mid = len(x) // 2
        half = max_len // 2
        x = x[mid - half : mid + half]
    order = max(8, min(16, int(sample_rate / 1000) + 2))
    try:
        a = librosa.lpc(x, order=order)
    except Exception:
        return []
    roots = np.roots(a)
    roots = roots[np.imag(roots) >= 0.01]
    angs = np.arctan2(np.imag(roots), np.real(roots))
    freqs = angs * (sample_rate / (2.0 * np.pi))
    mags = np.abs(roots)
    # 稳定极点 + 人声共振峰合理范围
    cands = []
    for f, m in zip(freqs, mags):
        if 90.0 <= f <= 4500.0 and m >= 0.7:
            cands.append((float(f), float(m)))
    cands.sort(key=lambda t: t[0])
    # 去重：相邻过近只留幅值更大者
    merged = []
    for f, m in cands:
        if not merged or abs(f - merged[-1][0]) > 150.0:
            merged.append((f, m))
        elif m > merged[-1][1]:
            merged[-1] = (f, m)
    return [f for f, _ in merged[:n_formants]]


def extract_voice_freq_features(samples, sample_rate=16000):
    """
    频率侧辅助特征（说话人区分增强）：
      - pitch / pitch_p25 / pitch_p75 / pitch_iqr : 基频分布
      - f1 / f2 : 共振峰
      - centroid : 频谱质心
    失败字段为 None，调用方需容错。
    """
    feats = {
        "pitch": None,
        "pitch_p25": None,
        "pitch_p75": None,
        "pitch_iqr": None,
        "f1": None,
        "f2": None,
        "centroid": None,
    }
    if samples is None or len(samples) < int(0.35 * sample_rate):
        return feats
    try:
        import librosa

        clip = np.ascontiguousarray(samples, dtype=np.float32)
        max_len = int(2.0 * sample_rate)
        if len(clip) > max_len:
            clip = clip[:max_len]

        f0 = librosa.yin(
            clip, fmin=80, fmax=350, sr=sample_rate, frame_length=1024
        )
        f0 = f0[np.isfinite(f0) & (f0 > 0)]
        if len(f0) >= 5:
            p25, p50, p75 = np.percentile(f0, [25, 50, 75])
            feats["pitch"] = float(p50)
            feats["pitch_p25"] = float(p25)
            feats["pitch_p75"] = float(p75)
            feats["pitch_iqr"] = float(max(0.0, p75 - p25))

        formants = _estimate_formants_lpc(clip, sample_rate, n_formants=2)
        if len(formants) >= 1:
            feats["f1"] = float(formants[0])
        if len(formants) >= 2:
            feats["f2"] = float(formants[1])

        cen = librosa.feature.spectral_centroid(y=clip, sr=sample_rate)
        if cen is not None and cen.size > 0:
            feats["centroid"] = float(np.median(cen))
    except Exception:
        pass
    return feats


def estimate_pitch_median(samples, sample_rate=16000):
    """估计基频中位数(Hz)，兼容旧接口。"""
    return extract_voice_freq_features(samples, sample_rate).get("pitch")
