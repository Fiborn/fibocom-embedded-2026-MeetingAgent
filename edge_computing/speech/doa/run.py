# -*- coding: utf-8 -*-
"""
声源定位主程序 —— 100 ms 周期广播 PDF 标准 JSON。

用法:
  python doa/run.py
  python doa/run.py --simulate
  python doa/run.py --tcp 9900
  python doa/run.py --json-only
"""
import argparse
import sys
import time
from pathlib import Path

DOA_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(DOA_ROOT))

import config_doa as cfg
from broadcaster import DoaBroadcaster
from engine import DoaEngine
from mic_array import MicArrayReader, SimulatedMicArrayReader
from doa_types import DoaResult


def parse_args():
    parser = argparse.ArgumentParser(description="SC171 声源定位 (DOA) 独立模块")
    parser.add_argument(
        "--simulate",
        action="store_true",
        help="模拟模式（无麦克风硬件，用于联调 UI/接口）",
    )
    parser.add_argument(
        "--azimuth",
        type=float,
        default=cfg.SIM_AZIMUTH_DEG,
        help="模拟声源方位角（度）",
    )
    parser.add_argument(
        "--device",
        type=int,
        default=cfg.INPUT_DEVICE_INDEX,
        help="PyAudio 输入设备编号",
    )
    parser.add_argument(
        "--tcp",
        type=int,
        default=None,
        help="额外开启 TCP JSON 广播端口（供 UI/摄像头进程订阅）",
    )
    parser.add_argument(
        "--json-only",
        action="store_true",
        help="仅输出完整 JSON 行（适合管道/日志采集）",
    )
    parser.add_argument(
        "--no-log",
        action="store_true",
        help="不写入 output/doa/doa_stream.jsonl",
    )
    return parser.parse_args()


class JsonOnlyBroadcaster(DoaBroadcaster):
    def _print_console(self, result: DoaResult):
        print(result.to_json())


def run():
    args = parse_args()
    simulate = args.simulate or cfg.SIMULATION_MODE

    if simulate:
        print("=" * 50)
        print("DOA 模拟模式 | 方位角 {}°".format(args.azimuth))
        mic = SimulatedMicArrayReader(azimuth_deg=args.azimuth)
    else:
        print("=" * 50)
        print("DOA 实时模式 | 4 通道阵列麦")
        print("按 Ctrl+C 结束")
        mic = MicArrayReader(device_index=args.device)

    engine = DoaEngine()
    broadcaster = (
        JsonOnlyBroadcaster(console=True, log_file=not args.no_log, tcp_port=args.tcp)
        if args.json_only
        else DoaBroadcaster(console=True, log_file=not args.no_log, tcp_port=args.tcp)
    )

    interval_sec = cfg.BROADCAST_INTERVAL_MS / 1000.0
    last_publish = 0.0
    last_result = DoaResult.silent()

    try:
        while True:
            block = mic.read_block()
            result = engine.feed(block)

            now = time.time()
            if result is not None:
                last_result = result

            if now - last_publish >= interval_sec:
                broadcaster.publish(last_result)
                last_publish = now

    except KeyboardInterrupt:
        print("\n\nDOA 已停止")
    finally:
        mic.close()
        broadcaster.close()
        if not args.no_log:
            print("JSONL 日志: {}".format(cfg.JSON_LOG))


if __name__ == "__main__":
    run()
