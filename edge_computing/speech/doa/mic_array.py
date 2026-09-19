# -*- coding: utf-8 -*-
"""
4 通道麦克风阵列采集 —— 独立于 src/audio_capture.py 的读取接口，专供 DOA 使用。
内部复用 src/audio_capture.open_best_input_stream，与会议识别使用相同的设备匹配逻辑。
"""
import sys
from pathlib import Path

import numpy as np

DOA_ROOT = Path(__file__).resolve().parent
SPEAKER_ROOT = DOA_ROOT.parent
SPEAKER_SRC = SPEAKER_ROOT / "src"
sys.path.insert(0, str(SPEAKER_SRC))
sys.path.insert(0, str(SPEAKER_ROOT))
sys.path.insert(0, str(DOA_ROOT))

import config as speaker_config  # noqa: E402
import config_doa as cfg  # noqa: E402
from audio_capture import open_best_input_stream  # noqa: E402
from audio_utils import resample  # noqa: E402


def _require_pyaudio():
    try:
        import pyaudio
    except ImportError as exc:
        raise ImportError(
            "实时 DOA 需要 pyaudio。请安装: pip install pyaudio\n"
            "或使用模拟模式: python doa/run.py --simulate"
        ) from exc
    return pyaudio


class MicArrayReader:
    """读取 N 通道 PCM，输出 shape (num_mics, samples) float32 @ 16 kHz"""

    def __init__(self, device_index=None, channels=None, sample_rate=None, chunk=None):
        pyaudio = _require_pyaudio()
        self._pyaudio = pyaudio
        self.device_index = (
            cfg.INPUT_DEVICE_INDEX if device_index is None else device_index
        )
        self.target_channels = cfg.INPUT_CHANNELS if channels is None else channels
        self.sample_rate = cfg.SAMPLE_RATE if sample_rate is None else sample_rate
        self.requested_chunk = cfg.CHUNK if chunk is None else chunk

        speaker_config.INPUT_DEVICE_INDEX = self.device_index
        speaker_config.INPUT_CHANNELS = self.target_channels
        if getattr(speaker_config, "PRIMARY_CHANNEL", None) is None:
            speaker_config.PRIMARY_CHANNEL = 0

        self.pa = self._pyaudio.PyAudio()
        self.stream = None
        self.capture_rate = self.sample_rate
        self.channels = self.target_channels
        self.chunk = self.requested_chunk
        self._open_stream()

    def _open_stream(self):
        pyaudio = self._pyaudio
        if self.stream is not None:
            self._close_stream()
        (
            self.stream,
            self.capture_rate,
            self.channels,
            self.device_index,
            self.chunk,
        ) = open_best_input_stream(self.pa, self.device_index)

    def _close_stream(self):
        if self.stream is None:
            return
        try:
            if self.stream.is_active():
                self.stream.stop_stream()
        except Exception:
            pass
        try:
            self.stream.close()
        except Exception:
            pass
        self.stream = None

    def _decode_block(self, pcm_bytes):
        samples = np.frombuffer(pcm_bytes, dtype=np.int16)
        if self.channels == 1:
            block = samples.astype(np.float32).reshape(1, -1) / 32768.0
        else:
            block = samples.reshape(-1, self.channels).T.astype(np.float32) / 32768.0
        if self.capture_rate == self.sample_rate:
            return block
        return np.stack([
            resample(block[c], self.capture_rate, self.sample_rate)
            for c in range(block.shape[0])
        ], axis=0)

    def read_block(self):
        """返回 (channels, samples) float32"""
        raw = self.stream.read(self.chunk, exception_on_overflow=False)
        return self._decode_block(raw)

    def close(self):
        self._close_stream()
        if self.pa is not None:
            try:
                self.pa.terminate()
            except Exception:
                pass
            self.pa = None


class SimulatedMicArrayReader:
    """无硬件时生成带方向性时延的模拟多通道信号"""

    def __init__(self, azimuth_deg=125.5, elevation_deg=15.0, sample_rate=None, chunk=None):
        from array_geometry import mic_positions, predict_tdoa_matrix

        self.sample_rate = cfg.SAMPLE_RATE if sample_rate is None else sample_rate
        self.chunk = cfg.CHUNK if chunk is None else chunk
        self.channels = cfg.NUM_MICS
        self.positions = mic_positions()
        self.azimuth_deg = azimuth_deg
        self.elevation_deg = elevation_deg
        self._phase = 0.0
        self._tdoa_full = np.zeros(self.channels, dtype=np.float64)
        pred = predict_tdoa_matrix(azimuth_deg, elevation_deg, self.positions)
        self._tdoa_full[1:] = pred

    def read_block(self):
        t = np.arange(self.chunk, dtype=np.float64) / self.sample_rate
        base = 0.15 * np.sin(2 * np.pi * 220.0 * t + self._phase)
        self._phase += 2 * np.pi * 220.0 * self.chunk / self.sample_rate

        out = np.zeros((self.channels, self.chunk), dtype=np.float32)
        for mic in range(self.channels):
            delay_samples = int(self._tdoa_full[mic] * self.sample_rate)
            if delay_samples == 0:
                out[mic] = base.astype(np.float32)
            elif delay_samples > 0:
                out[mic, delay_samples:] = base[:-delay_samples].astype(np.float32)
            else:
                d = -delay_samples
                out[mic, : self.chunk - d] = base[d:].astype(np.float32)
        return out

    def close(self):
        pass
