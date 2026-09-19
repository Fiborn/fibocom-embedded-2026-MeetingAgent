#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""一键 Demo：启动实时会议识别 30 秒"""
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main():
    subprocess.call([sys.executable, "src/check_env.py"], cwd=str(ROOT))
    print("\n>>> 30 秒后自动结束，请 2~3 人轮流说话")
    print(">>> 输出格式: 说话人1: 具体内容\n")
    proc = subprocess.Popen(
        [sys.executable, "src/meeting_realtime.py"],
        cwd=str(ROOT),
    )
    try:
        time.sleep(30)
        proc.send_signal(2)  # Ctrl+C
        proc.wait(timeout=5)
    except Exception:
        proc.kill()
    print("\nDemo 结束，查看 output/meeting_transcript.txt")


if __name__ == "__main__":
    main()
