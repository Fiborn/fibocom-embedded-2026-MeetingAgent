#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
端到端 pipeline 冒烟（无需真实麦克风）。

用法:
  python3 scripts/smoke_pipeline.py

覆盖：配置接线 → 引擎加载 → 4ch DOA 喂数 → 单通道降级 → 长度不齐 → end_session
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(SRC))

UI_DIR = ROOT.parent / "UI"
if UI_DIR.exists():
    sys.path.insert(0, str(UI_DIR))


def main():
    import config
    from audio_utils import TARGET_SR, align_multi_to_mono, unpack_read_16k
    from meeting_realtime import RealtimeMeetingEngine
    from speaker_tracker import TRACKER_VERSION

    mono = np.zeros(1000, np.float32)
    multi = np.zeros((4, 900), np.float32)
    _, aligned = align_multi_to_mono(mono, multi)
    assert aligned is not None and aligned.shape == (4, 1000)
    _, _, m = unpack_read_16k((mono, 0.1, multi))
    assert m is not None and m.shape[1] == 1000

    if UI_DIR.exists():
        from paths import apply_runtime_paths, load_config

        rt = apply_runtime_paths(load_config())
        sp = rt.get("speaker") or {}
        if "use_doa" in sp:
            config.SPEAKER_USE_DOA = bool(sp["use_doa"])

    eng = RealtimeMeetingEngine(on_utterance=lambda s, t: None)
    if not eng._asr_ready.wait(timeout=180):
        raise RuntimeError("ASR 加载超时")
    if eng._asr_error is not None:
        raise RuntimeError("ASR 加载失败: {}".format(eng._asr_error))

    sr = TARGET_SR
    n = int(1.2 * sr)
    t = np.arange(n) / sr
    x = (0.2 * np.sin(2 * np.pi * 160 * t)).astype(np.float32)
    chs = [x]
    for dly in (1, 2, 3):
        c = np.zeros(n, np.float32)
        c[dly:] = x[:-dly]
        chs.append(c)
    multi4 = np.stack(chs, axis=0)
    sil = np.zeros(int(0.8 * sr), np.float32)
    sil4 = np.zeros((4, len(sil)), np.float32)

    eng.begin_session()
    for i in range(0, n, 1024):
        eng.feed_16k(x[i : i + 1024], multi=multi4[:, i : i + 1024])
    eng.feed_16k(sil, multi=sil4)
    eng.flush()
    time.sleep(0.5)
    assert eng._multi_ring is not None and eng._multi_ring.shape[0] == 4
    eng.end_session()

    eng.begin_session()
    eng.feed_16k(x, multi=None)
    eng.feed_16k(sil, multi=None)
    eng.flush()
    eng.end_session()

    eng.begin_session()
    eng.feed_16k(x, multi=multi4[:, : n // 2])
    eng.end_session()

    print("SMOKE_OK tracker={} doa_enabled={}".format(
        TRACKER_VERSION, getattr(config, "SPEAKER_USE_DOA", None)
    ))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        print("SMOKE_FAIL: {}".format(exc), file=sys.stderr)
        raise
