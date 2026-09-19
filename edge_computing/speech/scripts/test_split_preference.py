#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""验证：两个明显不同的声纹不应被粘成同一标签。"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(SRC))

import config  # noqa: E402
from speaker_tracker import SpeakerTracker  # noqa: E402


def _unit(vec):
    v = np.asarray(vec, dtype=np.float32)
    return v / (np.linalg.norm(v) + 1e-8)


def main():
    # 避免依赖真实 onnx：直接注入假 embedding / pitch
    tr = SpeakerTracker.__new__(SpeakerTracker)
    tr.extractor = None
    tr.speakers = []
    tr._last_label = None
    tr._self_sims = []
    tr._pending_streak = 0
    tr._pending_merges = []
    tr._unlimited = True
    tr._max_speakers = 0
    tr._use_doa = False

    emb_a = _unit(np.concatenate([np.ones(128), np.zeros(128)]))
    emb_b = _unit(np.concatenate([np.zeros(128), np.ones(128)]))
    # 余弦应接近 0
    assert tr._cosine(emb_a, emb_b) < 0.2

    seq = []

    def fake_extract(samples, sample_rate=16000):
        # samples 用均值编码身份：>0 → A，<0 → B
        return emb_a if float(np.mean(samples)) >= 0 else emb_b

    def fake_feats(samples, sample_rate=16000):
        male = float(np.mean(samples)) >= 0
        return {
            "pitch": 120.0 if male else 210.0,
            "pitch_p25": 110.0 if male else 195.0,
            "pitch_p75": 130.0 if male else 225.0,
            "pitch_iqr": 20.0,
            "f1": 500.0 if male else 700.0,
            "f2": 1200.0 if male else 1700.0,
            "centroid": 1500.0 if male else 2200.0,
        }

    tr._extract = fake_extract
    tr._estimate_voice_feats = fake_feats

    # A 说两句
    for _ in range(2):
        seq.append(tr.assign(np.ones(16000, np.float32) * 0.1, update_profile=True))
    # B 说两句（旧逻辑会因 pending=25 长时间粘成说话人1）
    for _ in range(2):
        seq.append(tr.assign(np.ones(16000, np.float32) * -0.1, update_profile=True))
    # A 再说一句
    seq.append(tr.assign(np.ones(16000, np.float32) * 0.1, update_profile=True))

    print("labels:", seq)
    print("profiles:", tr.summary())
    assert seq[0] == seq[1] == seq[4], "同一人 A 标签应一致"
    assert seq[2] == seq[3], "同一人 B 标签应一致"
    assert seq[0] != seq[2], "异人 A/B 必须不同标签"
    print("OK preference-split")


if __name__ == "__main__":
    main()
