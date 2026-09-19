# -*- coding: utf-8 -*-
"""
分析已有会议录音（输出说话人+内容，无时间戳）
用法: python3 src/analyze_file.py data/recordings/demo.wav
"""
import sys
import wave
from pathlib import Path

import numpy as np
import sherpa_onnx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config
from asr_engine import AsrEngine
from audio_utils import TARGET_SR, normalize, resample
from speaker_tracker import SpeakerTracker


def read_wav(path):
    with wave.open(str(path), "rb") as wf:
        ch = wf.getnchannels()
        sr = wf.getframerate()
        raw = wf.readframes(wf.getnframes())
    samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    if ch > 1:
        samples = samples.reshape(-1, ch)[:, 0]
    return samples, sr


def analyze_wav(wav_path):
    if not config.VAD_MODEL.exists():
        raise FileNotFoundError("请先运行 download_model.py")

    samples, sr = read_wav(wav_path)
    if sr != TARGET_SR:
        samples = resample(samples, sr, TARGET_SR)
        sr = TARGET_SR

    asr = AsrEngine()
    tracker = SpeakerTracker()

    vad_cfg = sherpa_onnx.VadModelConfig()
    vad_cfg.silero_vad.model = str(config.VAD_MODEL)
    vad_cfg.silero_vad.min_silence_duration = config.MIN_SILENCE_DURATION
    vad_cfg.silero_vad.min_speech_duration = 0.12
    vad_cfg.silero_vad.threshold = 0.35
    vad_cfg.sample_rate = sr
    window = vad_cfg.silero_vad.window_size
    vad = sherpa_onnx.VoiceActivityDetector(vad_cfg, buffer_size_in_seconds=600)

    min_samples = int(config.MIN_SPEECH_DURATION * sr)
    idx = 0
    lines = []

    while idx + window <= len(samples):
        vad.accept_waveform(samples[idx : idx + window])
        idx += window
        while not vad.empty():
            seg = normalize(np.array(vad.front.samples, dtype=np.float32))
            vad.pop()
            if len(seg) < min_samples:
                continue
            text = asr.transcribe(seg, sr)
            if not text:
                continue
            speaker = tracker.assign(seg, sr)
            line = "{}: {}".format(speaker, text)
            lines.append(line)
            print(line, flush=True)

    if hasattr(vad, "flush"):
        vad.flush()
        while not vad.empty():
            seg = normalize(np.array(vad.front.samples, dtype=np.float32))
            vad.pop()
            if len(seg) < min_samples:
                continue
            text = asr.transcribe(seg, sr)
            if not text:
                continue
            speaker = tracker.assign(seg, sr)
            line = "{}: {}".format(speaker, text)
            lines.append(line)
            print(line, flush=True)

    return lines


def main():
    if len(sys.argv) < 2:
        print("用法: python3 src/analyze_file.py <wav文件>")
        sys.exit(1)
    wav = sys.argv[1]
    print("分析: {}".format(wav))
    lines = analyze_wav(wav)
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out = config.OUTPUT_DIR / (Path(wav).stem + "_transcript.txt")
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
        if lines:
            f.write("\n")
    print("\n已保存: {}（共 {} 条）".format(out, len(lines)))


if __name__ == "__main__":
    main()
