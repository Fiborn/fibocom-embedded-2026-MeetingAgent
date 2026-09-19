#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
检查 USB 4 通道阵列麦是否真的能用于 DOA。

用法:
  python doa/diagnose_channels.py
  python doa/diagnose_channels.py 37
"""
import sys
from pathlib import Path

import numpy as np

DOA_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(DOA_ROOT))

import config_doa as cfg
from gcc_phat import pairwise_tdoas
from mic_array import MicArrayReader

DOA_ENGINE_VERSION = "planar-v2"


def main():
    device = int(sys.argv[1]) if len(sys.argv) > 1 else cfg.INPUT_DEVICE_INDEX
    print("=" * 50)
    print("DOA 通道诊断  version={}".format(DOA_ENGINE_VERSION))
    print("设备 index: {}".format(device))
    print("请对着麦克风不同方向说话，采集约 3 秒…")
    print("=" * 50)

    mic = MicArrayReader(device_index=device)
    blocks = []
    try:
        for _ in range(30):
            blocks.append(mic.read_block())
    finally:
        mic.close()

    data = np.concatenate(blocks, axis=1)
    print("\n各通道 RMS:")
    for i in range(data.shape[0]):
        rms = float(np.sqrt(np.mean(data[i] ** 2)))
        print("  ch{}: {:.5f}".format(i, rms))

    ref = data[0]
    diffs = [float(np.max(np.abs(data[i] - ref))) for i in range(1, data.shape[0])]
    print("\n相对 ch0 的 peak_diff:", ", ".join("{:.6f}".format(d) for d in diffs))
    max_diff = max(diffs) if diffs else 0.0

    if max_diff < 1e-3:
        print("\n【结论】4 通道波形几乎相同，当前设备/通道配置无法做 DOA。")
        print("可能原因:")
        print("  1. PyAudio 4 通道里只有 1 路是真麦克风，其余是重复或静音")
        print("  2. 选错了 INPUT_DEVICE_INDEX")
        print("  3. 需要在 config_doa.py 调整 MIC_CHANNEL_ORDER")
        return 1

    window = data[:, -min(data.shape[1], int(cfg.SAMPLE_RATE * 0.25)) :]
    tdoas, gcc_conf = pairwise_tdoas(window, cfg.SAMPLE_RATE)
    print("\nTDOA (微秒, 相对 Mic0):", ", ".join("{:.1f}".format(t * 1e6) for t in tdoas))
    print("GCC 平均置信度: {:.3f}".format(gcc_conf))

    if float(np.max(np.abs(tdoas))) < cfg.MIN_TDOA_ABS_SEC:
        print("\n【结论】通道有差异，但时延过小 (< {:.1f} us)，角度估计不可靠。".format(
            cfg.MIN_TDOA_ABS_SEC * 1e6
        ))
        print("可尝试: ARRAY_GEOMETRY = \"linear\" 或增大 ANALYSIS_WINDOW_SEC")
        return 2

    print("\n【结论】硬件通道正常，可以跑 DOA。")
    print("请确认 run.py 启动时显示: DOA engine version={}".format(DOA_ENGINE_VERSION))
    print("若仍是 俯仰 45°，说明 engine.py 未同步到板子。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
