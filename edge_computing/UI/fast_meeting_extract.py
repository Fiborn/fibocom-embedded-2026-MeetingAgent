"""规则引擎：从会议转写快速提取结构化草稿（毫秒级）。"""
from __future__ import annotations

import json
import re
from typing import Any

FUNCTION_NAME = "create_meeting_task_list"

_SPEAKER_LINE = re.compile(
    r"^(说话人\d+|Speaker\d+|[\u4e00-\u9fff]{1,4}|[A-Za-z][\w\-]{0,31})\s*:\s*(.+)$",
    re.I,
)
_ROLE_VERBS = "负责|跟进|完成|处理|整理|提交|实现|开发|测试|部署|上线|修复|优化|确认|安排"
_TASK_HINT = re.compile(
    rf"(?:{_ROLE_VERBS}|需要|安排)"
)
_DECISION_HINT = re.compile(r"(决定|确认|统一|采用|同意|通过|定为|方案是|结论是)")
_DUE_DATE = re.compile(
    r"(\d{4}-\d{1,2}-\d{1,2}|\d{1,2}月\d{1,2}[日号]|今天|明天|后天|本周[一二三四五六日天]|下周[一二三四五六日天])"
)
_TOPIC_HINT = re.compile(r"(?:会议|讨论|关于|主题)[是为：:]\s*([^\s，,。；;]{2,30})")
_FILLER = re.compile(r"^(嗯+|啊+|哦+|ok|OK|好的|可以|喂+|那个|然后|就是)\s*$", re.I)
_META_LINE = re.compile(
    r"^(录音开始|录音结束|---\s*(暂停|恢复)|-{3,}|={3,})",
    re.I,
)
_INVALID_ASSIGNEE = frozenset(
    {
        "其实", "那个", "这个", "就是", "然后", "我们", "你们", "他们", "一下",
        "今天", "明天", "后天", "接口", "模型", "会议", "说话", "有没有", "不太",
        "清楚", "已经", "开始", "等他", "转写", "出来", "再跟", "跟他说", "等他等",
        "第一句", "使用方法", "数据库", "大模型", "学习", "构建", "处理", "文本",
        "开始转", "开始了", "已经开", "已经可以", "可以把它", "这边已经",
        "据库", "库构", "模型使", "用方法",
    }
)
_COMMON_SURNAMES = frozenset(
    "张王李赵刘陈杨黄周吴徐孙马朱胡郭何高林罗郑梁宋谢唐韩曹许邓萧冯曾程蔡彭潘"
    "袁于董余苏叶吕魏蒋田杜丁沈姜范江傅钟卢汪戴崔任陆廖姚方金邱夏谭韦贾邹石熊孟"
    "秦阎薛侯雷白龙段郝孔邵史毛常万顾赖武康贺严尹钱施牛洪龚汤陶黎易常文樊兰殷"
    "施陶洪翟安颜倪严牛温芦季俞章鲁葛伍申尤毕聂丛焦向柳邢路岳齐沿梅莫庄辛管祝左"
    "涂谷祁时舒耿牟卜路詹关苗凌费纪靳盛童欧甄项曲成游阳裴席卫查屈鲍位覃霍翁隋"
    "植甘景薄单包司柏宁柯阮桂闵解强柴华车冉房边辜吉饶刁瞿戚丘古米池滕晋苑邬臧"
    "畅宫来嵺苟全褚廉简娄盖符奚木穆党燕郎邸冀谈姬屠连郜晏栾暴甘钭厉戎祖武符"
)
_SURNAME_RE = "[" + "".join(sorted(_COMMON_SURNAMES)) + "]"
_ASSIGN_MARKER = re.compile(
    rf"({_SURNAME_RE}[\u4e00-\u9fff]{{0,3}})\s*(?:来)?(负责|跟进)"
)


def _strip_speaker_prefix(line: str) -> tuple[str, str]:
    line = line.strip()
    bracket = re.match(r"^\[\d{2}:\d{2}:\d{2}\]\s*(.+)$", line)
    if bracket:
        line = bracket.group(1).strip()
    if not line or _META_LINE.match(line):
        return "", ""
    match = _SPEAKER_LINE.match(line)
    if match:
        return match.group(1), match.group(2).strip()
    return "", line


def _is_valid_assignee(name: str) -> bool:
    name = (name or "").strip()
    if not re.fullmatch(r"[\u4e00-\u9fff]{2,4}", name):
        return False
    if _FILLER.fullmatch(name):
        return False
    if name in _INVALID_ASSIGNEE:
        return False
    if any(ch in name for ch in "的了吗呢吧啊呀嘛着过"):
        return False
    if name[0] not in _COMMON_SURNAMES:
        return False
    return True


def _clean_task_text(text: str) -> str:
    text = (text or "").strip()
    text = re.sub(r"^(?:来)?(?:负责|跟进)\s*", "", text)
    text = text.strip(" ：:，,。；;、")
    return text


def _extract_due(text: str) -> str | None:
    match = _DUE_DATE.search(text)
    return match.group(1) if match else None


def _extract_tasks_from_text(text: str) -> list[tuple[str | None, str]]:
    """从一句转写中提取 (assignee, task) 列表，支持多人多任务。"""
    valid_markers = [m for m in _ASSIGN_MARKER.finditer(text) if _is_valid_assignee(m.group(1))]

    found: list[tuple[str | None, str]] = []
    for i, match in enumerate(valid_markers):
        assignee = match.group(1)
        task_start = match.end()
        task_end = valid_markers[i + 1].start() if i + 1 < len(valid_markers) else len(text)
        task = text[task_start:task_end].strip(" ，,。；;、")
        task = _clean_task_text(task)
        if task and len(task) >= 2:
            found.append((assignee, task))

    if found:
        return found

    single = _ASSIGN_MARKER.search(text)
    if single and _is_valid_assignee(single.group(1)):
        task = _clean_task_text(text[single.end() :])
        if task:
            return [(single.group(1), task)]

    if _TASK_HINT.search(text):
        task = _clean_task_text(text)
        if task and len(task) >= 4 and not _FILLER.fullmatch(task):
            return [(None, task)]

    return []


def _dedupe_tasks(tasks: list[dict]) -> list[dict]:
    seen: set[tuple[str, str]] = set()
    out: list[dict] = []
    for item in tasks:
        assignee = (item.get("assignee") or "").strip()
        task = (item.get("task") or "").strip()
        if not task:
            continue
        key = (assignee, task)
        if key in seen:
            continue
        seen.add(key)
        out.append(item)
    return out


def _dedupe_decisions(decisions: list[dict]) -> list[dict]:
    seen: set[str] = set()
    out: list[dict] = []
    for item in decisions:
        val = (item.get("decision") or "").strip()
        if not val or val in seen:
            continue
        seen.add(val)
        out.append(item)
    return out


def _guess_title(lines: list[tuple[str, str]], tasks: list[dict]) -> str:
    for _speaker, text in lines:
        topic = _TOPIC_HINT.search(text)
        if topic:
            return topic.group(1).strip()
    if tasks:
        first = tasks[0]
        assignee = first.get("assignee") or ""
        task = first.get("task") or ""
        if assignee and task:
            return f"{assignee}：{task[:16]}"
        if task:
            return task[:24] + ("…" if len(task) > 24 else "")
    for _speaker, text in lines:
        if len(text) >= 8 and not _FILLER.fullmatch(text):
            cleaned = re.sub(r"[\u4e00-\u9fff]{2,4}\s*(?:来)?(?:负责|跟进)\S+", "", text)
            cleaned = cleaned.strip(" ，,。；;")
            if len(cleaned) >= 6:
                return cleaned[:24] + ("…" if len(cleaned) > 24 else "")
    return "未命名会议"


def _build_summary(lines: list[tuple[str, str]], tasks: list[dict], max_sentences: int = 4) -> str:
    parts: list[str] = []
    for item in tasks:
        assignee = item.get("assignee") or "待定"
        task = item.get("task") or ""
        if task:
            parts.append(f"{assignee}负责{task}")
    if parts:
        return "。".join(parts[:max_sentences]) + "。"

    picked: list[str] = []
    for _speaker, text in lines:
        text = text.strip()
        if len(text) < 8 or _FILLER.fullmatch(text):
            continue
        if text not in picked:
            picked.append(text)
        if len(picked) >= max_sentences:
            break
    if not picked:
        return "本次会议暂无有效摘要内容。"
    return "。".join(picked) + ("。" if not picked[-1].endswith(("。", "！", "？")) else "")


def extract_meeting_rules(meeting_text: str) -> dict[str, Any]:
    """规则解析转写 → 与 LLM 相同结构的 task_list 草稿。"""
    raw_lines = [ln.strip() for ln in meeting_text.splitlines() if ln.strip()]
    parsed: list[tuple[str, str]] = []
    for line in raw_lines:
        speaker, text = _strip_speaker_prefix(line)
        if text:
            parsed.append((speaker, text))

    participants = sorted({sp for sp, _ in parsed if sp})
    name_participants: set[str] = set()
    decisions: list[dict] = []
    tasks: list[dict] = []

    for speaker, text in parsed:
        due = _extract_due(text)

        if _DECISION_HINT.search(text):
            owner = None
            for match in _ASSIGN_MARKER.finditer(text):
                if _is_valid_assignee(match.group(1)):
                    owner = match.group(1)
                    break
            decisions.append(
                {
                    "decision": text,
                    "owner": owner,
                    "due_date": due,
                    "evidence": text,
                }
            )

        for assignee, task_text in _extract_tasks_from_text(text):
            if assignee:
                name_participants.add(assignee)
            tasks.append(
                {
                    "task": task_text,
                    "assignee": assignee,
                    "due_date": due,
                    "priority": "medium",
                    "status": "todo",
                    "source_decision": None,
                }
            )

    decisions = _dedupe_decisions(decisions)
    tasks = _dedupe_tasks(tasks)

    all_participants = sorted(set(participants) | name_participants)

    task_list = {
        "function_call": {
            "name": FUNCTION_NAME,
            "arguments": {
                "meeting_title": _guess_title(parsed, tasks),
                "summary": _build_summary(parsed, tasks),
                "participants": all_participants,
                "decisions": decisions,
                "tasks": tasks,
            },
        }
    }
    return task_list


def draft_to_refine_text(task_list: dict) -> str:
    """压缩草稿 JSON，供 LLM 精炼。"""
    args = task_list["function_call"]["arguments"]
    compact = {
        "meeting_title": args.get("meeting_title"),
        "summary": args.get("summary"),
        "participants": args.get("participants"),
        "decisions": args.get("decisions"),
        "tasks": args.get("tasks"),
    }
    return json.dumps(compact, ensure_ascii=False, separators=(",", ":"))
