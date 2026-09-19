# -*- coding: utf-8 -*-
"""WebRTC VAD 备用切分（Silero 无输出时使用）"""
import webrtcvad

import config
from audio_utils import TARGET_SR


def webrtc_available():
    try:
        import webrtcvad
        return True
    except ImportError:
        return False


def webrtc_segments(samples, sample_rate=16000, aggressiveness=1):
    """返回 [(start_sample, end_sample), ...]"""
    vad = webrtcvad.Vad(aggressiveness)
    frame_ms = 30
    frame_len = int(sample_rate * frame_ms / 1000)
    bytes_per_frame = frame_len * 2

    pcm = (samples * 32768.0).clip(-32768, 32767).astype("int16").tobytes()
    n_frames = len(pcm) // bytes_per_frame
    if n_frames == 0:
        return []

    flags = []
    for i in range(n_frames):
        frame = pcm[i * bytes_per_frame : (i + 1) * bytes_per_frame]
        flags.append(vad.is_speech(frame, sample_rate))

    min_frames = max(1, int(config.MIN_SPEECH_DURATION / (frame_ms / 1000.0)))
    segments, i = [], 0
    while i < len(flags):
        if not flags[i]:
            i += 1
            continue
        start = i
        while i < len(flags) and flags[i]:
            i += 1
        end = i
        if end - start >= min_frames:
            s = start * frame_len
            e = end * frame_len
            segments.append((s, min(e, len(samples))))
    return segments
