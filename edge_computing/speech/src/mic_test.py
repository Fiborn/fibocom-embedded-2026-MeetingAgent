# -*- coding: utf-8 -*-
"""麦克风测试"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config

from audio_capture import MicrophoneReader, print_level_meter


def main():
    duration = int(sys.argv[1]) if len(sys.argv) > 1 else 15
    print("=" * 50)
    print("麦克风测试 {} 秒".format(duration))
    print("INPUT_DEVICE_INDEX = {}".format(config.INPUT_DEVICE_INDEX))
    print("INPUT_CHANNELS = {}".format(config.INPUT_CHANNELS))
    print("=" * 50)

    mic = MicrophoneReader()
    max_rms = 0.0
    t0 = time.time()
    try:
        while time.time() - t0 < duration:
            from audio_utils import unpack_read_16k
            chunk, level, _multi = unpack_read_16k(mic.read_16k())
            max_rms = max(max_rms, level)
            print_level_meter(level)
            time.sleep(0.05)
    except KeyboardInterrupt:
        pass
    finally:
        mic.close()

    print("\n\n最大 RMS: {:.4f}".format(max_rms))
    if max_rms < 0.005:
        print("\n[失败] 无声音！请运行: python3 src/find_mic.py")
        print("  然后修改 config.py → INPUT_DEVICE_INDEX")
        sys.exit(1)
    print("\n[通过] 可运行 meeting_realtime.py")


if __name__ == "__main__":
    main()
