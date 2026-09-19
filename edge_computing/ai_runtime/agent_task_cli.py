#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import time
from datetime import datetime
from pathlib import Path


BASE = Path("/home/fibo")
PROJECT_ROOT = BASE / "MeetingAgent"
LICENSE_DIR = BASE / "qcom_6490_license"
MODEL_ROOT = PROJECT_ROOT / "ai_models" / "llm_models"
OUT_DIR = PROJECT_ROOT / "data" / "agent_tasks"

MODELS = {
    "qwen": MODEL_ROOT / "Qwen3-0.6B" / "qwen3-0.6b_1.0.0_qcom_6490_qnn_2.28_dsp_537988b3eb4c6960d47c433c63fd32c2.fmodel",
    "deepseek": MODEL_ROOT / "deepseek-r1-distill-qwen-1.5b" / "deepseek-r1-qwen-1.5b_1.0.0_all_all_mnn_3.0.5_cpu_206fb9a374c94c4d4d52bf0e873c739c.fmodel",
}

FUNCTION_NAME = "create_meeting_task_list"


def configure_agent_paths(
    *,
    base: Path | None = None,
    license_dir: Path | None = None,
    model_root: Path | None = None,
    out_dir: Path | None = None,
) -> None:
    """供 UI / worker 注入运行时路径。"""
    global BASE, PROJECT_ROOT, LICENSE_DIR, MODEL_ROOT, OUT_DIR, MODELS
    if base is not None:
        BASE = Path(base)
        PROJECT_ROOT = BASE
    if license_dir is not None:
        LICENSE_DIR = Path(license_dir)
    if model_root is not None:
        MODEL_ROOT = Path(model_root)
    if out_dir is not None:
        OUT_DIR = Path(out_dir)
    MODELS = {
        "qwen": MODEL_ROOT
        / "Qwen3-0.6B"
        / "qwen3-0.6b_1.0.0_qcom_6490_qnn_2.28_dsp_537988b3eb4c6960d47c433c63fd32c2.fmodel",
        "deepseek": MODEL_ROOT
        / "deepseek-r1-distill-qwen-1.5b"
        / "deepseek-r1-qwen-1.5b_1.0.0_all_all_mnn_3.0.5_cpu_206fb9a374c94c4d4d52bf0e873c739c.fmodel",
    }


class LlmSession:
    """常驻 LLM 会话：License + 模型只初始化一次。"""

    def __init__(self):
        self._api = None
        self._model_name = None
        self._licensed = False
        self._init_sec = 0.0

    @property
    def model_name(self):
        return self._model_name

    @property
    def ready(self) -> bool:
        return self._api is not None and self._model_name is not None

    def ensure_license(self) -> None:
        if self._licensed:
            return
        init_license()
        self._licensed = True

    def load_model(self, model_name: str) -> dict:
        from fiboaisdk.api_aisdk_py import api_nlp_py as nlp_api

        if self.ready and self._model_name == model_name:
            return {
                "model": model_name,
                "init_sec": 0.0,
                "reused": True,
            }

        self.release()
        model_path = MODELS[model_name]
        if not model_path.exists():
            raise FileNotFoundError(f"model not found: {model_path}")

        self.ensure_license()
        api = nlp_api.NLPAPI()
        start = time.perf_counter()
        ret = api.Init(str(model_path), "")
        init_sec = time.perf_counter() - start
        print(f"[llm] model={model_name} Init => {ret} ({init_sec:.3f}s)", flush=True)
        if ret != 0:
            api.Release()
            raise RuntimeError(f"LLM init failed: {ret}")

        self._api = api
        self._model_name = model_name
        self._init_sec = init_sec
        return {
            "model": model_name,
            "init_sec": round(init_sec, 4),
            "reused": False,
        }

    def generate(self, model_name: str, meeting_text: str) -> tuple[str, dict]:
        from fiboaisdk.api_aisdk_py import api_nlp_py as nlp_api

        load_meta = self.load_model(model_name)
        result = nlp_api.ResultNlpText()
        prompt = build_prompt(model_name, meeting_text)
        start = time.perf_counter()
        ret = self._api.GenerateSync(prompt, result)
        generate_sec = time.perf_counter() - start
        text = clean_llm_text(getattr(result, "text", ""))
        print(f"[llm] GenerateSync => {ret} ({generate_sec:.3f}s)", flush=True)
        if ret != 0:
            raise RuntimeError(f"LLM generate failed: {ret}")

        meta = {
            "model": model_name,
            "init_sec": load_meta.get("init_sec", 0.0),
            "generate_sec": round(generate_sec, 4),
            "reused_model": bool(load_meta.get("reused")),
            "prompt_chars": len(prompt),
        }
        return text, meta

    def release(self) -> None:
        if self._api is not None:
            try:
                self._api.Release()
            except Exception:
                pass
        self._api = None
        self._model_name = None
        self._init_sec = 0.0


def qwen_prompt(meeting_text):
    system = (
        "你是端侧会议 Agent。把会议转写整理为 JSON 函数调用，"
        "只输出 JSON，不要 Markdown，不要解释，不要思考过程。"
    )
    user = (
        "函数名: create_meeting_task_list\n"
        "字段: meeting_title, summary, participants[], "
        "decisions[{decision,owner,due_date,evidence}], "
        "tasks[{task,assignee,due_date,priority,status,source_decision}]\n"
        "规则: decisions=会议决策; tasks=行动项且尽量含 assignee; "
        "due_date 无则 null; priority 默认 medium; status 固定 todo。\n\n"
        f"会议内容：\n{meeting_text}\n/no_think"
    )
    return (
        f"<|im_start|>system\n{system}<|im_end|>\n"
        f"<|im_start|>user\n{user}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def plain_prompt(meeting_text):
    return (
        "你是端侧会议 Agent。只输出 JSON，不要解释。\n"
        '结构: {"function_call":{"name":"create_meeting_task_list","arguments":'
        '{"meeting_title":"","summary":"","participants":[],"decisions":[],"tasks":[]}}}\n'
        "从下面会议内容提取决策、执行人和任务：\n"
        f"{meeting_text}\nJSON:"
    )


def read_license_file(path):
    mode = "r" if str(path).endswith(".pem") else "rb"
    with open(path, mode) as f:
        return f.read()


def init_license():
    from fiboaisdk.api_aisdk_py import license_py as license_api

    ret = license_api.Init(
        read_license_file(LICENSE_DIR / "key1.pem"),
        read_license_file(LICENSE_DIR / "key2.pem"),
        read_license_file(LICENSE_DIR / "key3.pem"),
        read_license_file(LICENSE_DIR / "license.bin"),
    )
    print(f"[license] Init => {ret}")
    if ret != 0:
        raise RuntimeError(f"license init failed: {ret}")


def clean_llm_text(text):
    text = text or ""
    while "<think>" in text and "</think>" in text:
        start = text.find("<think>")
        end = text.find("</think>", start) + len("</think>")
        text = text[:start] + text[end:]
    for marker in ["<think>", "</think>", "<|im_end|>", "<|im_start|>", "<|endoftext|>"]:
        text = text.replace(marker, "")
    return text.strip()


def build_prompt(model_name, meeting_text):
    if model_name == "qwen":
        return qwen_prompt(meeting_text)
    return plain_prompt(meeting_text)


def build_refine_prompt(
    model_name,
    draft_json: str,
    meeting_text: str,
    *,
    snippet_max_chars: int = 600,
) -> str:
    """LLM 精炼：在规则草稿基础上纠错、去重、补全。"""
    snippet = meeting_text.strip()
    if len(snippet) > snippet_max_chars:
        snippet = snippet[-snippet_max_chars:]

    if model_name == "qwen":
        system = (
            "你是会议记录质检员。输入包含规则引擎草稿和原始转写。"
            "请修正错误、合并重复、补全 assignee，输出标准 JSON 函数调用。"
            "只输出 JSON，不要解释，不要 Markdown，不要思考过程。"
        )
        user = (
            "函数名必须是 create_meeting_task_list。\n"
            "字段: meeting_title, summary, participants, decisions, tasks。\n"
            "要求:\n"
            "1. 以规则草稿为主，仅依据转写修正明显错误。\n"
            "2. 删除重复任务/决策；tasks 必须尽量保留 assignee。\n"
            "3. summary 用 2-4 句中文概括，不要逐句复制转写。\n"
            "4. due_date 无明确日期填 null；priority 默认 medium；status 固定 todo。\n"
            "5. 输出尽量简短，JSON 控制在 800 字以内。\n\n"
            f"规则草稿 JSON:\n{draft_json}\n\n"
            f"原始转写(节选):\n{snippet}\n/no_think"
        )
        return (
            f"<|im_start|>system\n{system}<|im_end|>\n"
            f"<|im_start|>user\n{user}<|im_end|>\n"
            "<|im_start|>assistant\n"
        )

    return (
        "你是会议记录质检员。根据规则草稿和转写，输出修正后的 JSON。\n"
        '结构: {"function_call":{"name":"create_meeting_task_list","arguments":{...}}}\n'
        f"规则草稿:\n{draft_json}\n\n转写节选:\n{snippet}\nJSON:"
    )


def refine_meeting_tasks(
    draft_task_list: dict,
    meeting_text: str,
    *,
    model_name: str = "qwen",
    out_name: str | None = None,
    allow_fallback: bool = True,
    session: LlmSession | None = None,
    metadata_extra: dict | None = None,
    snippet_max_chars: int = 400,
) -> tuple[dict, dict, str]:
    """规则草稿 → LLM 精炼 → 最终 task_list。"""
    owned_session = session is None
    session = session or LlmSession()
    metadata: dict = {
        "mode": "rules_llm_refine",
        "model": model_name,
        "pipeline": "rules_llm",
    }
    if metadata_extra:
        metadata.update(metadata_extra)

    draft_args = draft_task_list["function_call"]["arguments"]
    draft_json = json.dumps(
        {
            "meeting_title": draft_args.get("meeting_title"),
            "summary": draft_args.get("summary"),
            "participants": draft_args.get("participants"),
            "decisions": draft_args.get("decisions"),
            "tasks": draft_args.get("tasks"),
        },
        ensure_ascii=False,
        separators=(",", ":"),
    )

    try:
        session.ensure_license()
        session.load_model(model_name)
        from fiboaisdk.api_aisdk_py import api_nlp_py as nlp_api

        prompt = build_refine_prompt(
            model_name, draft_json, meeting_text, snippet_max_chars=snippet_max_chars
        )
        result = nlp_api.ResultNlpText()
        print(
            f"[llm] Refine 开始 prompt={len(prompt)}字 tasks={len(draft_args.get('tasks') or [])}",
            flush=True,
        )
        start = time.perf_counter()
        ret = session._api.GenerateSync(prompt, result)
        generate_sec = time.perf_counter() - start
        raw_output = clean_llm_text(getattr(result, "text", ""))
        print(f"[llm] Refine GenerateSync => {ret} ({generate_sec:.3f}s)", flush=True)
        if ret != 0:
            raise RuntimeError(f"LLM refine failed: {ret}")

        metadata.update(
            {
                "init_sec": 0.0,
                "generate_sec": round(generate_sec, 4),
                "reused_model": True,
                "prompt_chars": len(prompt),
                "rules_draft_tasks": len(draft_args.get("tasks") or []),
                "rules_draft_decisions": len(draft_args.get("decisions") or []),
            }
        )
        try:
            task_list = parse_function_call(raw_output)
            metadata["parse_status"] = "refined"
        except Exception as exc:
            if not allow_fallback:
                raise
            metadata["parse_status"] = "refine_fallback_to_rules"
            metadata["parse_error"] = str(exc)
            task_list = normalize_task_list(draft_task_list)

        stem = out_name or datetime.now().strftime("%Y%m%d_%H%M%S")
        save_outputs(task_list, raw_output, metadata, stem)
        return task_list, metadata, raw_output
    finally:
        if owned_session:
            session.release()


def call_llm(model_name, meeting_text):
    session = LlmSession()
    try:
        return session.generate(model_name, meeting_text)
    finally:
        session.release()


def extract_meeting_tasks(
    meeting_text: str,
    *,
    model_name: str = "qwen",
    out_name: str | None = None,
    allow_fallback: bool = True,
    session: LlmSession | None = None,
    metadata_extra: dict | None = None,
) -> tuple[dict, dict, str]:
    """提取会议 JSON；可复用 LlmSession 以避免重复加载模型。"""
    owned_session = session is None
    session = session or LlmSession()
    metadata: dict = {"mode": "extract", "model": model_name}
    if metadata_extra:
        metadata.update(metadata_extra)

    try:
        raw_output, llm_meta = session.generate(model_name, meeting_text)
        metadata.update(llm_meta)
        try:
            task_list = parse_function_call(raw_output)
            metadata["parse_status"] = "ok"
        except Exception as exc:
            if not allow_fallback:
                raise
            metadata["parse_status"] = "fallback"
            metadata["parse_error"] = str(exc)
            task_list = fallback_task_list(meeting_text, str(exc))

        stem = out_name or datetime.now().strftime("%Y%m%d_%H%M%S")
        save_outputs(task_list, raw_output, metadata, stem)
        return task_list, metadata, raw_output
    finally:
        if owned_session:
            session.release()


def extract_json_object(text):
    text = clean_llm_text(text)
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.S)
    if fenced:
        text = fenced.group(1)
    start = text.find("{")
    if start < 0:
        raise ValueError("model output does not contain a JSON object")

    depth = 0
    in_string = False
    escape = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : index + 1]
    raise ValueError("model output JSON object is incomplete")


def normalize_priority(value):
    value = (value or "medium").lower()
    if value in {"high", "medium", "low"}:
        return value
    if value in {"高", "紧急", "重要"}:
        return "high"
    if value in {"低", "不急"}:
        return "low"
    return "medium"


def normalize_due_date(value):
    if value in ("", "null", "None", None):
        return None
    value = str(value).strip()
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", value):
        return value
    return value


def normalize_task_list(data):
    if "function_call" not in data:
        data = {"function_call": {"name": FUNCTION_NAME, "arguments": data}}

    call = data.get("function_call") or {}
    args = call.get("arguments") or {}
    if isinstance(args, str):
        args = json.loads(args)

    normalized = {
        "function_call": {
            "name": call.get("name") or FUNCTION_NAME,
            "arguments": {
                "meeting_title": args.get("meeting_title") or "未命名会议",
                "summary": args.get("summary") or "",
                "participants": args.get("participants") or [],
                "decisions": [],
                "tasks": [],
            },
        }
    }

    if normalized["function_call"]["name"] != FUNCTION_NAME:
        normalized["function_call"]["name"] = FUNCTION_NAME

    for decision in args.get("decisions") or []:
        if not isinstance(decision, dict):
            continue
        text = str(decision.get("decision") or "").strip()
        if not text:
            continue
        normalized["function_call"]["arguments"]["decisions"].append(
            {
                "decision": text,
                "owner": decision.get("owner") or None,
                "due_date": normalize_due_date(decision.get("due_date")),
                "evidence": decision.get("evidence") or "",
            }
        )

    for task in args.get("tasks") or []:
        if not isinstance(task, dict):
            continue
        text = str(task.get("task") or "").strip()
        if not text:
            continue
        normalized["function_call"]["arguments"]["tasks"].append(
            {
                "task": text,
                "assignee": task.get("assignee") or None,
                "due_date": normalize_due_date(task.get("due_date")),
                "priority": normalize_priority(task.get("priority")),
                "status": "todo",
                "source_decision": task.get("source_decision") or None,
            }
        )
    return normalized


def parse_function_call(text):
    json_text = extract_json_object(text)
    return normalize_task_list(json.loads(json_text))


def fallback_task_list(meeting_text, error_message):
    lines = [line.strip(" -\t") for line in meeting_text.splitlines() if line.strip()]
    tasks = []
    decisions = []

    def extract_owner(line):
        for keyword in ["来负责", "负责", "跟进", "完成"]:
            if keyword not in line:
                continue
            before = line.split(keyword, 1)[0]
            before = re.split(r"[，,。；;、\s]", before)[-1]
            if 1 <= len(before) <= 8:
                return before
        return None

    def extract_due(line):
        match = re.search(r"(\d{4}-\d{1,2}-\d{1,2}|\d{1,2}月\d{1,2}日|今天|明天|后天|本周[一二三四五六日天]|下周[一二三四五六日天])", line)
        return match.group(1) if match else None

    for line in lines:
        if any(word in line for word in ["决定", "确认", "统一", "采用", "上线", "部署"]):
            decisions.append(
                {
                    "decision": line,
                    "owner": extract_owner(line),
                    "due_date": extract_due(line),
                    "evidence": line,
                }
            )
        if any(word in line for word in ["负责", "跟进", "完成", "测试", "整理", "提交", "实现"]):
            tasks.append(
                {
                    "task": line,
                    "assignee": extract_owner(line),
                    "due_date": extract_due(line),
                    "priority": "medium",
                    "status": "todo",
                    "source_decision": None,
                }
            )

    if not tasks:
        tasks.append(
            {
                "task": "人工复核会议内容并补充任务清单",
                "assignee": None,
                "due_date": None,
                "priority": "medium",
                "status": "todo",
                "source_decision": None,
            }
        )

    return {
        "function_call": {
            "name": FUNCTION_NAME,
            "arguments": {
                "meeting_title": "会议任务清单",
                "summary": f"模型 JSON 解析失败，已使用本地兜底规则生成。原因：{error_message}",
                "participants": [],
                "decisions": decisions,
                "tasks": tasks,
            },
        }
    }


def save_outputs(task_list, raw_output, metadata, out_name=None):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stem = out_name or datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = OUT_DIR / f"{stem}.json"
    raw_path = OUT_DIR / f"{stem}_raw.txt"
    latest_path = OUT_DIR / "latest_tasks.json"

    payload = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "metadata": metadata,
        "task_list": task_list,
    }
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    latest_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    raw_path.write_text(raw_output or "", encoding="utf-8")
    return json_path, raw_path, latest_path


def read_input_text(args):
    if args.file:
        return Path(args.file).read_text(encoding="utf-8")
    if args.text:
        return args.text
    if args.demo:
        return (
            "会议主题：端侧智能体联调。\n"
            "参会人：张三、李四、王五。\n"
            "会议决定：采用 Qwen3 作为端侧 Agent 的任务提取模型。\n"
            "张三负责明天完成 Function Calling prompt 模板。\n"
            "李四负责测试从会议纪要中提取决策和执行人。\n"
            "王五下周五前整理 JSON 任务清单并保存到本地。"
        )
    raise ValueError("please provide --text, --file, or --demo")


def cmd_extract(args):
    meeting_text = read_input_text(args)
    metadata = {"mode": "extract", "model": args.model}

    if args.mock_output:
        raw_output = Path(args.mock_output).read_text(encoding="utf-8")
        metadata["mock_output"] = args.mock_output
        try:
            task_list = parse_function_call(raw_output)
            metadata["parse_status"] = "ok"
        except Exception as exc:
            if not args.allow_fallback:
                raise
            metadata["parse_status"] = "fallback"
            metadata["parse_error"] = str(exc)
            task_list = fallback_task_list(meeting_text, str(exc))
    else:
        task_list, metadata, raw_output = extract_meeting_tasks(
            meeting_text,
            model_name=args.model,
            out_name=args.out_name,
            allow_fallback=args.allow_fallback,
            metadata_extra=metadata,
        )
        json_path = OUT_DIR / f"{args.out_name or datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        raw_path = OUT_DIR / f"{json_path.stem}_raw.txt"
        latest_path = OUT_DIR / "latest_tasks.json"
        args_obj = task_list["function_call"]["arguments"]
        print(json.dumps(task_list, ensure_ascii=False, indent=2))
        print(f"[agent] decisions={len(args_obj['decisions'])}, tasks={len(args_obj['tasks'])}")
        print(f"[agent] json => {json_path}")
        print(f"[agent] raw => {raw_path}")
        print(f"[agent] latest => {latest_path}")
        return

    json_path, raw_path, latest_path = save_outputs(task_list, raw_output, metadata, args.out_name)
    args_obj = task_list["function_call"]["arguments"]
    print(json.dumps(task_list, ensure_ascii=False, indent=2))
    print(f"[agent] decisions={len(args_obj['decisions'])}, tasks={len(args_obj['tasks'])}")
    print(f"[agent] json => {json_path}")
    print(f"[agent] raw => {raw_path}")
    print(f"[agent] latest => {latest_path}")


def cmd_validate(args):
    raw = Path(args.json_file).read_text(encoding="utf-8")
    data = json.loads(raw)
    if "task_list" in data:
        data = data["task_list"]
    task_list = normalize_task_list(data)
    args_obj = task_list["function_call"]["arguments"]
    print(f"[agent] valid: decisions={len(args_obj['decisions'])}, tasks={len(args_obj['tasks'])}")


def build_parser():
    parser = argparse.ArgumentParser(description="SC171V3 local Agent task extraction with function-call style JSON.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    extract = sub.add_parser("extract")
    extract.add_argument("--model", choices=["qwen", "deepseek"], default="qwen")
    extract.add_argument("--text")
    extract.add_argument("--file")
    extract.add_argument("--demo", action="store_true")
    extract.add_argument("--out-name")
    extract.add_argument("--mock-output")
    extract.add_argument("--allow-fallback", action="store_true", default=True)
    extract.set_defaults(func=cmd_extract)

    validate = sub.add_parser("validate")
    validate.add_argument("json_file")
    validate.set_defaults(func=cmd_validate)

    return parser


def main():
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
