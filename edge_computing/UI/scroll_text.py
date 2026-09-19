"""从录音记录 TXT 中提取转写正文，供桌面 / Web 界面显示。"""
from __future__ import annotations

import re
from typing import List


def extract_display_lines(content: str, *, skip_ai_blocks: bool = True) -> List[str]:
    """从 record TXT 提取转写正文（跳过 AI 插入块与分隔线），保持原文不转义。

    历史说明：原实现会把 "," 替换为 ";"、'"' 替换为 "'"，目的是让文本能安全
    下发给陶晶驰串口屏。串口屏 UI 已下线，此处不再转义，以免污染界面显示与
    送入 LLM 的会议正文。
    """
    lines: List[str] = []
    skipping_block = False
    for raw in content.splitlines():
        stripped = raw.strip()
        if not stripped:
            continue
        if stripped.startswith("录音开始") or stripped.startswith("录音结束"):
            continue
        if stripped.startswith("---") or re.fullmatch(r"[-=*═]{10,}", stripped):
            continue
        if stripped.startswith("[会议总结]") or stripped.startswith("[会议日程]"):
            if skip_ai_blocks:
                skipping_block = True
            continue
        if skipping_block:
            if re.match(r"^\[\d{2}:\d{2}:\d{2}\]", stripped):
                skipping_block = False
            else:
                continue
        match = re.match(r"^\[\d{2}:\d{2}:\d{2}\]\s*(.+)$", stripped)
        if match:
            lines.append(match.group(1).strip())
        elif not skipping_block:
            lines.append(stripped)
    return lines
