# -*- coding: utf-8 -*-
"""
扫描可用麦克风（ALSA + PyAudio）
用法: python3 src/find_mic.py
测试时请对着麦克风说话！
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config
from audio_capture import find_best_alsa_device, list_input_devices, parse_arecord_devices, test_alsa_device

import pyaudio
import numpy as np


def test_pyaudio_device(idx, seconds=2):
    pa = pyaudio.PyAudio()
    info = pa.get_device_info_by_index(idx)
    best_peak = 0.0
    best_cfg = None
    for ch in [1, 2]:
        for rate in [16000, int(info["defaultSampleRate"]), 48000, 44100]:
            try:
                stream = pa.open(
                    format=pyaudio.paInt16, channels=ch, rate=rate,
                    input=True, input_device_index=idx, frames_per_buffer=1024,
                )
                frames = []
                n = int(rate / 1024 * seconds)
                for _ in range(n):
                    frames.append(stream.read(1024, exception_on_overflow=False))
                stream.stop_stream()
                stream.close()
                pcm = b"".join(frames)
                samples = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
                if ch > 1:
                    samples = samples.reshape(-1, ch)[:, 0]
                peak = float(np.max(np.abs(samples))) if len(samples) else 0.0
                if peak > best_peak:
                    best_peak = peak
                    best_cfg = (rate, ch)
            except Exception:
                continue
    pa.terminate()
    return best_peak, best_cfg


def main():
    print("=" * 60)
    print("麦克风扫描（请对着麦克风说话）")
    print("=" * 60)

    print("\n【1】ALSA 设备 (arecord -l)")
    alsa_list = parse_arecord_devices()
    if not alsa_list:
        print("  未解析到设备，尝试 hw:0,0 ~ hw:3,0")
    else:
        for d in alsa_list:
            print("  {}  {}".format(d["alsa"], d["name"]))

    print("\n  正在测试 ALSA（约 {} 秒/设备）…".format(2 * max(1, len(alsa_list))))
    dev, rate, ch, peak = find_best_alsa_device()
    if peak >= 0.005:
        print("\n  [推荐 ALSA]")
        print("    ALSA_DEVICE = \"{}\"".format(dev))
        print("    ALSA_RATE = {}".format(rate))
        print("    ALSA_CHANNELS = {}".format(ch))
        print("    CAPTURE_BACKEND = \"alsa\"")
        print("    峰值: {:.4f}".format(peak))
    else:
        print("\n  ALSA 全部静音")

    print("\n【2】PyAudio 设备")
    pa = pyaudio.PyAudio()
    best_pa = (0.0, None, None)
    for d in list_input_devices(pa):
        peak, cfg = test_pyaudio_device(d["index"])
        mark = " <-- 默认" if d["default"] else ""
        print("  [{}] {}  peak={:.4f}{}".format(d["index"], d["name"], peak, mark))
        if peak > best_pa[0]:
            best_pa = (peak, d["index"], cfg)
    pa.terminate()

    if best_pa[0] >= 0.005:
        rate, ch = best_pa[2]
        print("\n  [PyAudio 可用]")
        print("    CAPTURE_BACKEND = \"pyaudio\"")
        print("    INPUT_DEVICE_INDEX = {}".format(best_pa[1]))
        print("    INPUT_CHANNELS = {}".format(ch))
        print("    峰值: {:.4f}".format(best_pa[0]))
    else:
        print("\n  PyAudio 全部静音（你的情况很可能如此）")

    print("\n" + "=" * 60)
    print("请把上面 [推荐 ALSA] 的配置写入 config.py")
    print("然后: python3 src/mic_test.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
