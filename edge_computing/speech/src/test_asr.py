# -*- coding: utf-8 -*-
"""
ASR 诊断：录 5 秒 → 直接转文字
用法:
  python3 src/test_asr.py
  python3 src/test_asr.py data/recordings/test.wav
"""
import sys
import time
import wave
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config
from asr_engine import AsrEngine
try:
    from audio_capture import create_microphone_reader
except ImportError:
    from audio_capture import MicrophoneReader
    def create_microphone_reader():
        return MicrophoneReader()
from audio_utils import TARGET_SR, normalize, resample, rms


def read_wav(path):
    with wave.open(str(path), "rb") as wf:
        sr = wf.getframerate()
        raw = wf.readframes(wf.getnframes())
        ch = wf.getnchannels()
    samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    if ch > 1:
        samples = samples.reshape(-1, ch)[:, 0]
    return samples, sr


def record_seconds(seconds=5):
    mic = create_microphone_reader()
    print("请对着麦克风说一句话（{} 秒）…".format(seconds))
    t0 = time.time()
    from audio_utils import unpack_flush_16k

    while time.time() - t0 < seconds:
        mic.read_16k()
    samples, _multi = unpack_flush_16k(mic.flush_16k())
    mic.close()
    return samples


def main():
    print("=" * 50)
    print("ASR 诊断工具")
    print("=" * 50)
    print("ASR 模型: {}  存在: {}".format(
        config.ASR_MODEL, config.ASR_MODEL.exists()
    ))

    asr = AsrEngine()

    if len(sys.argv) > 1:
        samples, sr = read_wav(sys.argv[1])
        print("文件: {}  sr={}  时长={:.1f}s".format(
            sys.argv[1], sr, len(samples) / sr
        ))
    else:
        samples = record_seconds(5)
        sr = TARGET_SR

    samples = normalize(samples)
    print("RMS: {:.4f}  峰值: {:.4f}".format(
        rms(samples), float(np.max(np.abs(samples)))
    ))

    if rms(samples) < 0.001:
        print("\n[错误] 录音几乎静音！先运行: python3 src/mic_test.py")
        sys.exit(1)

    text = asr.transcribe(samples, sr)
    print("\n识别结果: [{}]".format(text if text else "（空）"))

    if not text:
        print("\n可能原因: ASR 模型问题或录音质量差")
        sys.exit(1)

    print("\nASR 正常！")


if __name__ == "__main__":
    main()
