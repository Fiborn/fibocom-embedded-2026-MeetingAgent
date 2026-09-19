# -*- coding: utf-8 -*-
"""会议场景 ASR 后处理：常见同音/错词纠错（轻量、无额外模型）。"""
from __future__ import annotations

import re

# 左：模型常错；右：会议语境更合理。仅做整词/短语替换，避免误伤。
_MEETING_REPLACEMENTS = (
    ("会意", "会议"),
    ("会意纪要", "会议纪要"),
    ("会意总结", "会议总结"),
    ("行动想", "行动项"),
    ("行动象", "行动项"),
    ("待办想", "待办项"),
    ("说活人", "说话人"),
    ("端测", "端侧"),
    ("端测智能", "端侧智能"),
    ("大摸型", "大模型"),
    ("大模形", "大模型"),
    ("麦克风风", "麦克风"),
    ("麦克疯", "麦克风"),
    ("OKR ", "OKR"),
    ("kpi", "KPI"),
    ("Kpi", "KPI"),
    ("下周吧", "下周把"),
)

# 正则类：数字/日期常见粘连
_REGEX_FIXES = (
    (re.compile(r"(\d)\s*点\s*(\d{1,2})\s*分"), r"\1点\2分"),
    (re.compile(r"百分之\s*(\d+)"), r"百分之\1"),
)


def post_correct_transcript(text: str) -> str:
    text = (text or "").strip()
    if not text:
        return text
    # 去掉 Paraformer 偶发的多余空格（中文间）
    text = re.sub(r"(?<=[\u4e00-\u9fff])\s+(?=[\u4e00-\u9fff])", "", text)
    text = re.sub(r"\s{2,}", " ", text)
    for src, dst in _MEETING_REPLACEMENTS:
        if src in text:
            text = text.replace(src, dst)
    for pat, repl in _REGEX_FIXES:
        text = pat.sub(repl, text)
    return text.strip()
