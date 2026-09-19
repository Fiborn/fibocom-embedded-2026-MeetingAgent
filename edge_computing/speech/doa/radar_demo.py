# -*- coding: utf-8 -*-
"""
ASCII 声场雷达图 Demo —— 订阅 DOA JSON 输出，验证 UI 交互逻辑。

用法（两个终端）:
  终端1: python doa/run.py --simulate --tcp 9900
  终端2: python doa/radar_demo.py --port 9900
"""
import argparse
import json
import math
import socket
import sys
from pathlib import Path

DOA_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(DOA_ROOT))

from config_doa import CONFIDENCE_MIN_FOR_ACTION


def draw_radar(azimuth, elevation, confidence, vad_active, width=41, height=21):
    cx, cy = width // 2, height // 2
    radius = min(cx, cy) - 2
    grid = [[" " for _ in range(width)] for _ in range(height)]

    for r in (radius // 2, radius):
        for deg in range(0, 360, 3):
            rad = math.radians(deg - 90)
            x = int(cx + r * math.cos(rad))
            y = int(cy + r * math.sin(rad))
            if 0 <= x < width and 0 <= y < height:
                grid[y][x] = "."

    for deg, label in ((0, "0"), (90, "90"), (180, "180"), (270, "270")):
        rad = math.radians(deg - 90)
        x = int(cx + (radius + 1) * math.cos(rad))
        y = int(cy + (radius + 1) * math.sin(rad))
        if 0 <= x < width and 0 <= y < height:
            grid[y][x] = label

    if vad_active and confidence >= CONFIDENCE_MIN_FOR_ACTION:
        rad = math.radians(azimuth - 90)
        x = int(cx + (radius - 2) * math.cos(rad))
        y = int(cy + (radius - 2) * math.sin(rad))
        if 0 <= x < width and 0 <= y < height:
            grid[y][x] = "*"

    lines = ["".join(row) for row in grid]
    header = "声场雷达 | vad={} conf={:.2f} az={:.0f}° el={:.0f}°".format(
        vad_active, confidence, azimuth, elevation
    )
    return header + "\n" + "\n".join(lines)


def listen_tcp(port):
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect(("127.0.0.1", port))
    file = sock.makefile("r", encoding="utf-8")
    print("已连接 DOA TCP 127.0.0.1:{}".format(port))
    try:
        for line in file:
            data = json.loads(line)
            sd = data.get("source_data", {})
            screen = draw_radar(
                sd.get("azimuth", 0),
                sd.get("elevation", 0),
                sd.get("confidence", 0),
                data.get("vad_active", False),
            )
            sys.stdout.write("\033[2J\033[H" + screen + "\n")
            sys.stdout.flush()
    finally:
        sock.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=9900)
    args = parser.parse_args()
    listen_tcp(args.port)


if __name__ == "__main__":
    main()
