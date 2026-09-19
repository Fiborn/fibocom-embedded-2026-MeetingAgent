"""
录音脚本（录会议用，无需预注册）
用法:
  python3 src/record.py data/recordings/test.wav 5
  python3 src/record.py data/recordings/meeting.wav 60
"""
import sys
from pathlib import Path

import numpy as np
import pyaudio
import wave

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config

FORMAT = pyaudio.paInt16


def pick_channel(raw, channels, pick=0):
    if channels == 1:
        return raw
    samples = np.frombuffer(raw, dtype=np.int16).reshape(-1, channels)
    return samples[:, pick].astype(np.int16).tobytes()


def record(filename, duration):
    config.DATA_MEETINGS.mkdir(parents=True, exist_ok=True)

    pa = pyaudio.PyAudio()
    stream = pa.open(
        format=FORMAT,
        channels=config.INPUT_CHANNELS,
        rate=config.SAMPLE_RATE,
        input=True,
        input_device_index=config.INPUT_DEVICE_INDEX,
        frames_per_buffer=config.CHUNK,
    )
    frames = []
    n = int(config.SAMPLE_RATE / config.CHUNK * duration)
    print("开始录音 {} 秒（请多人轮流发言）…".format(duration))
    for _ in range(n):
        raw = stream.read(config.CHUNK, exception_on_overflow=False)
        frames.append(pick_channel(raw, config.INPUT_CHANNELS, config.PRIMARY_CHANNEL))
    stream.stop_stream()
    stream.close()
    pa.terminate()

    out = Path(filename)
    out.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(out), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(config.SAMPLE_RATE)
        wf.writeframes(b"".join(frames))
    print("已保存: {}".format(out.resolve()))


if __name__ == "__main__":
    default = str(config.DATA_MEETINGS / "test.wav")
    out = sys.argv[1] if len(sys.argv) > 1 else default
    dur = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    record(out, dur)
