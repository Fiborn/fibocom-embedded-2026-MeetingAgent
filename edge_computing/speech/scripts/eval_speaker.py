#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
离线评估说话人分离效果（竞赛调参 / 答辩材料）

用法:
  python3 scripts/eval_speaker.py data/recordings/test.wav labels.txt

labels.txt 每行格式（无时间戳，按 VAD 段顺序）:
  说话人1
  说话人2
  说话人1
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import config
from analyze_file import analyze_wav


def load_labels(path):
    lines = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                lines.append(line)
    return lines


def main():
    if len(sys.argv) < 3:
        print("用法: python3 scripts/eval_speaker.py <wav> <labels.txt>")
        sys.exit(1)

    wav = Path(sys.argv[1])
    labels_path = Path(sys.argv[2])
    if not wav.exists():
        print("找不到 WAV: {}".format(wav))
        sys.exit(1)
    if not labels_path.exists():
        print("找不到标签: {}".format(labels_path))
        sys.exit(1)

    gold = load_labels(labels_path)
    print("评估: {}".format(wav))
    print("标注段数: {}".format(len(gold)))

    import io
    from contextlib import redirect_stdout

    buf = io.StringIO()
    with redirect_stdout(buf):
        lines = analyze_wav(str(wav))

    pred = []
    for line in lines:
        pred.append(line.split(":", 1)[0].strip())

    n = min(len(gold), len(pred))
    if n == 0:
        print("无有效段，无法评估")
        sys.exit(1)

    correct = sum(1 for i in range(n) if gold[i] == pred[i])
    acc = correct / n
    print("\n预测段数: {}".format(len(pred)))
    print("对齐段数: {}".format(n))
    print("段级标签准确率: {:.1f}% ({}/{})".format(acc * 100, correct, n))

    if len(gold) != len(pred):
        print("注意: 段数不一致，请检查 VAD 切分或标注是否对齐")

    print("\n逐段对比:")
    for i in range(n):
        mark = "OK" if gold[i] == pred[i] else "X"
        print("  [{}] 标注={}  预测={}".format(mark, gold[i], pred[i]))


if __name__ == "__main__":
    main()
