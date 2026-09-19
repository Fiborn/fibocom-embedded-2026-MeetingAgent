#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""真机采集 + 引擎短时跑通检查（约 12 秒）。"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
UI = ROOT.parent / "UI"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(SRC))
if UI.exists():
    sys.path.insert(0, str(UI))


def detect_dov_alsa():
    """从 arecord -l 找 DOV / USB Audio 卡号。"""
    import re
    import subprocess

    try:
        out = subprocess.check_output(["arecord", "-l"], text=True, stderr=subprocess.STDOUT)
    except Exception as exc:
        return None, str(exc)
    card = None
    for line in out.splitlines():
        m = re.search(r"^card (\d+):.*DOV", line, re.I)
        if m:
            card = int(m.group(1))
            break
        m = re.search(r"^card (\d+):.*USB-Audio", line, re.I)
        if m and card is None:
            card = int(m.group(1))
    if card is None:
        return None, "未找到 DOV/USB 采集卡"
    return "plughw:{},0".format(card), None


def probe_channels(device):
    """探测可用声道数：优先真实 4，其次 2（拒绝 plughw 假扩成的静音通道）。"""
    import subprocess

    for ch in (4, 2, 1):
        cmd = [
            "arecord", "-D", device, "-f", "S16_LE", "-r", "16000",
            "-c", str(ch), "-d", "1", "-t", "raw", "/tmp/_ch_probe.raw",
        ]
        p = subprocess.run(cmd, capture_output=True, text=True)
        if p.returncode != 0:
            continue
        x = np.fromfile("/tmp/_ch_probe.raw", dtype=np.int16)
        if x.size < ch:
            continue
        frames = x.reshape(-1, ch)
        peaks = [float(np.max(np.abs(frames[:, i]))) for i in range(ch)]
        live = sum(1 for p in peaks if p > 2)
        # plughw 常把 2ch 扩成 4ch，后两路恒 0
        if ch == 4 and live < 3:
            continue
        return ch, max(peaks) if peaks else 0.0, None
    return None, 0.0, "无法打开 {}".format(device)


def main():
    import config
    from audio_capture import create_microphone_reader
    from audio_utils import unpack_read_16k, rms
    from meeting_realtime import RealtimeMeetingEngine

    device, err = detect_dov_alsa()
    if not device:
        print("FAIL: {}".format(err))
        return 2
    print("DETECT_DEVICE", device)

    ch, peak0, err = probe_channels(device)
    if ch is None:
        print("FAIL: {}".format(err))
        return 3
    print("DETECT_CHANNELS", ch, "probe_peak", peak0)

    # 应用 UI 同款覆盖
    if UI.exists():
        from paths import apply_runtime_paths, load_config

        rt = apply_runtime_paths(load_config())
        mic_cfg = rt.get("mic") or {}
        # 真机探测结果优先于可能过期的 ui_config
        config.CAPTURE_BACKEND = "alsa"
        config.ALSA_DEVICE = device
        config.INPUT_CHANNELS = ch
        config.PRIMARY_CHANNEL = int(mic_cfg.get("primary_channel", 0))
        config.SAMPLE_RATE = 16000
        config.CHUNK = int(mic_cfg.get("frames", 2048))
    else:
        config.CAPTURE_BACKEND = "alsa"
        config.ALSA_DEVICE = device
        config.INPUT_CHANNELS = ch

    # 不足 4 通道时关闭 DOA，避免虚假 4ch 零填充污染
    if ch < 4:
        config.SPEAKER_USE_DOA = False
        print("DOA_DISABLED reason=channels<4 actual={}".format(ch))
    else:
        config.SPEAKER_USE_DOA = True
        print("DOA_ENABLED")

    utterances = []

    def on_utt(spk, text):
        utterances.append((spk, text))
        print("UTT", spk, (text or "")[:60], flush=True)

    print("ENGINE_LOAD...", flush=True)
    eng = RealtimeMeetingEngine(on_utterance=on_utt)
    if not eng._asr_ready.wait(timeout=180):
        print("FAIL: ASR timeout")
        return 4
    if eng._asr_error is not None:
        print("FAIL: ASR {}".format(eng._asr_error))
        return 5
    print("ENGINE_READY", flush=True)

    print("MIC_OPEN...", flush=True)
    mic = create_microphone_reader()
    eng.begin_session()

    # 真机验收：略放宽静音切段，便于短时说话出结果
    try:
        eng.vad.config.silero_vad.min_silence_duration = 0.4
    except Exception:
        pass

    duration = 18.0
    t0 = time.time()
    max_peak = 0.0
    n_chunks = 0
    multi_ch = None
    print("RECORDING {:.0f}s — 请连续对着 DOV 说几句话…".format(duration), flush=True)
    try:
        while time.time() - t0 < duration:
            chunk, level, multi = unpack_read_16k(mic.read_16k())
            if len(chunk) == 0:
                continue
            n_chunks += 1
            max_peak = max(max_peak, float(np.max(np.abs(chunk))), level)
            if multi is not None:
                multi_ch = multi.shape[0]
            eng.feed_16k(chunk, multi=multi)
            if n_chunks % 40 == 0:
                print("  …{:.1f}s peak={:.4f}".format(time.time() - t0, max_peak), flush=True)
        # 麦缓冲尾部也喂入，避免句末被截断
        flushed = mic.flush_16k()
        from audio_utils import unpack_flush_16k
        tail, tail_m = unpack_flush_16k(flushed)
        if len(tail):
            eng.feed_16k(tail, multi=tail_m)
    finally:
        mic.close()

    eng.flush()
    # 追加一段静音帮助 VAD 收尾切段
    silence = np.zeros(int(1.2 * 16000), dtype=np.float32)
    eng.feed_16k(silence, multi=None)
    eng.flush()
    for _ in range(60):
        if eng._asr_queue.empty():
            break
        time.sleep(0.1)
    lines = eng.get_transcript_lines()
    eng.end_session()

    print("RESULT chunks={} max_peak={:.5f} multi_ch={} segs={} utts={} lines={}".format(
        n_chunks, max_peak, multi_ch, eng._seg_count, len(utterances), lines
    ))
    print("DOA_LAST", getattr(eng.doa, "last_azimuth", None))

    if n_chunks < 5:
        print("FAIL: 几乎读不到音频块")
        return 6
    if max_peak < 0.001:
        print("WARN: 麦克风电平过低(接近静音)，软件链路已跑通但无有效语音输入")
        print("HINT: 检查 DOV 是否插入/供电；当前 USB 描述为 {}ch；设备 {}".format(ch, device))
        print("SOFT_OK_LOW_LEVEL")
        return 0
    if utterances or lines:
        print("REAL_MIC_PIPELINE_OK")
        return 0
    if eng._seg_count > 0:
        print("WARN: 有 VAD 段但 ASR 无有效文字")
        print("SOFT_OK_SEG_NO_ASR")
        return 0
    print("WARN: 有电平但无 VAD/ASR 段，请更大声连续再说一次")
    print("SOFT_OK_NO_SEG")
    return 0


if __name__ == "__main__":
    sys.exit(main())
