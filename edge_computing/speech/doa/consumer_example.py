# -*- coding: utf-8 -*-
"""
DOA 消费端示例 —— 演示 UI / 摄像头控制层如何解析 PDF JSON。

业务规则（与 PDF 一致）:
  - vad_active == false → 不驱动摄像头、不高亮雷达
  - confidence > 0.6    → 才执行跟踪动作
"""
import json
import sys
from pathlib import Path

DOA_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(DOA_ROOT))

from config_doa import CONFIDENCE_MIN_FOR_ACTION
from doa_types import DoaResult


def handle_doa_packet(raw_json: str):
    data = json.loads(raw_json)
    result = DoaResult.from_json_dict(data)

    if not result.vad_active:
        return {"action": "idle", "reason": "静音/底噪，不响应"}

    sd = result.source_data
    if sd.confidence < CONFIDENCE_MIN_FOR_ACTION:
        return {
            "action": "ignore",
            "reason": "置信度 {:.2f} 低于阈值 {:.2f}".format(
                sd.confidence, CONFIDENCE_MIN_FOR_ACTION
            ),
        }

    return {
        "action": "track",
        "ptz": {"pan": sd.azimuth, "tilt": sd.elevation},
        "beam_index": sd.beam_index,
        "ui_radar_highlight_deg": sd.azimuth,
    }


def demo():
    samples = [
        '{"timestamp":1,"vad_active":false,"source_data":{"azimuth":0,"elevation":0,"confidence":0,"beam_index":0}}',
        '{"timestamp":2,"vad_active":true,"source_data":{"azimuth":125.5,"elevation":15.0,"confidence":0.85,"beam_index":3}}',
        '{"timestamp":3,"vad_active":true,"source_data":{"azimuth":200.0,"elevation":10.0,"confidence":0.35,"beam_index":4}}',
    ]
    for s in samples:
        action = handle_doa_packet(s)
        print("输入:", s[:60], "...")
        print("输出:", action)
        print("-" * 40)


if __name__ == "__main__":
    demo()
