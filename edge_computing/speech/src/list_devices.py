# -*- coding: utf-8 -*-
"""列出 PyAudio 输入设备"""
import sys
from pathlib import Path

import pyaudio

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from audio_capture import list_input_devices

pa = pyaudio.PyAudio()
print("=" * 60)
print("输入设备列表（把阵列麦 index 填入 config.py）")
print("=" * 60)
for d in list_input_devices(pa):
    mark = " <-- 系统默认" if d["default"] else ""
    print("  [{}] {}".format(d["index"], d["name"]))
    print("       channels={}  rate={}{}".format(
        d["channels"], d["rate"], mark
    ))
print("=" * 60)
print("下一步:")
print("  1. 编辑 config.py → INPUT_DEVICE_INDEX = 你的设备号")
print("  2. python3 src/mic_test.py")
print("  3. python3 src/meeting_realtime.py")
pa.terminate()
