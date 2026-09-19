"""Harness / 长期记忆：供 UI 检索增强与会议结束后写回。"""
from __future__ import annotations

import json
import subprocess
import sys
import threading
from pathlib import Path
from typing import Any

from paths import PROJECT_ROOT, UI_DIR, load_config

MEMORY_CLI = PROJECT_ROOT / "ai_runtime" / "memory_cli.py"
_WRITE_LOCK = threading.Lock()
_SCHEDULED_WRITEBACK_STEMS: set[str] = set()
_SCHEDULED_LOCK = threading.Lock()


def harness_settings() -> dict:
    cfg = load_config()
    harness_cfg = cfg.get("harness") or {}
    defaults = {
        "enabled": True,
        "enhance_extract": True,
        "write_memory_on_end": True,
        "backend": "auto",
        "top_k": 5,
        "query_chars": 800,
        "min_chars": 20,
        "memory_tags": ["meeting", "ui"],
        "search_timeout_sec": 30,
        "write_timeout_sec": 120,
    }
    defaults.update(harness_cfg)
    return defaults


def _run_memory_cli(args: list[str], *, timeout: float) -> tuple[int, str, str]:
    cmd = [sys.executable, "-u", str(MEMORY_CLI), *args]
    proc = subprocess.run(
        cmd,
        text=True,
        capture_output=True,
        timeout=timeout,
        cwd=str(UI_DIR),
    )
    return proc.returncode, proc.stdout or "", proc.stderr or ""


def _parse_json_payload(output: str) -> dict | list | None:
    start = output.find("{")
    end = output.rfind("}")
    if start < 0 or end < start:
        return None
    try:
        return json.loads(output[start : end + 1])
    except json.JSONDecodeError:
        return None


def search_memories(
    query: str,
    *,
    top_k: int | None = None,
    backend: str | None = None,
    timeout: float | None = None,
) -> list[dict[str, Any]]:
    """子进程检索长期记忆，避免与 ASR 进程内 onnx 冲突。"""
    settings = harness_settings()
    if not settings.get("enabled", True):
        return []

    query = (query or "").strip()
    if not query:
        return []

    top_k = int(top_k if top_k is not None else settings["top_k"])
    backend = backend or settings.get("backend", "auto")
    timeout = float(timeout if timeout is not None else settings["search_timeout_sec"])

    if not MEMORY_CLI.exists():
        print(f"[harness] memory_cli 缺失: {MEMORY_CLI}")
        return []

    try:
        code, stdout, stderr = _run_memory_cli(
            ["search", query[: int(settings["query_chars"])], "--top-k", str(top_k), "--backend", backend],
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        print(f"[harness] 记忆检索超时 ({timeout:.0f}s)")
        return []
    except Exception as exc:
        print(f"[harness] 记忆检索失败: {exc}")
        return []

    if code != 0:
        detail = (stderr or stdout).strip()
        print(f"[harness] 记忆检索 exit={code}: {detail[:200]}")
        return []

    payload = _parse_json_payload(stdout)
    if not isinstance(payload, dict):
        return []
    results = payload.get("results") or []
    return results if isinstance(results, list) else []


def build_memory_augmented_text(meeting_text: str, memories: list[dict[str, Any]]) -> str:
    """将检索到的记忆拼入 LLM 输入，参考 post_meeting_harness 格式。"""
    meeting_text = (meeting_text or "").strip()
    if not memories:
        return meeting_text

    memory_lines = []
    for item in memories:
        memory_lines.append(
            "[{id}] score={score:.3f} {text}".format(
                id=item.get("id", "?"),
                score=float(item.get("score", 0.0)),
                text=item.get("text", ""),
            )
        )
    memory_block = "\n".join(memory_lines)
    return (
        "【检索到的长期记忆】\n"
        f"{memory_block}\n\n"
        "【修正与提取要求】\n"
        "1. 参考长期记忆修正转写中的专有名词、设备型号、人名和项目名。\n"
        "2. 不要编造转写和记忆中没有的任务。\n\n"
        "【待分析会议转写】\n"
        f"{meeting_text}\n"
    )


def add_meeting_memory(
    transcript_text: str,
    *,
    stem: str = "",
    tags: list[str] | None = None,
    backend: str | None = None,
    timeout: float | None = None,
) -> dict | None:
    """将会议转写写入长期记忆库并重建向量索引。"""
    settings = harness_settings()
    if not settings.get("enabled", True):
        return None

    transcript_text = (transcript_text or "").strip()
    min_chars = int(settings.get("min_chars", 20))
    if len(transcript_text) < min_chars:
        print(f"[harness] 转写过短，跳过写回 ({len(transcript_text)}<{min_chars})")
        return None

    if not MEMORY_CLI.exists():
        print(f"[harness] memory_cli 缺失: {MEMORY_CLI}")
        return None

    label = stem or "未命名"
    memory_text = f"会议记录({label})：\n{transcript_text}"
    backend = backend or settings.get("backend", "auto")
    timeout = float(timeout if timeout is not None else settings["write_timeout_sec"])
    tags = list(tags if tags is not None else settings.get("memory_tags") or ["meeting", "ui"])

    cmd = ["add", memory_text, "--backend", backend]
    for tag in tags:
        cmd.extend(["--tag", tag])

    with _WRITE_LOCK:
        try:
            code, stdout, stderr = _run_memory_cli(cmd, timeout=timeout)
        except subprocess.TimeoutExpired:
            print(f"[harness] 记忆写回超时 ({timeout:.0f}s)")
            return None
        except Exception as exc:
            print(f"[harness] 记忆写回失败: {exc}")
            return None

    if code != 0:
        detail = (stderr or stdout).strip()
        print(f"[harness] 记忆写回 exit={code}: {detail[:200]}")
        return None

    payload = _parse_json_payload(stdout)
    if isinstance(payload, dict):
        added = payload.get("added") or {}
        print(
            f"[harness] 已写回记忆 id={added.get('id')} "
            f"tfidf={payload.get('tfidf_count')} embedding={payload.get('embedding_count')}",
            flush=True,
        )
        return payload
    print("[harness] 记忆写回完成")
    return {"ok": True}


def writeback_from_record_file(txt_path: str | Path, *, stem: str = "") -> dict | None:
    from agent_bridge import read_transcript_file

    path = Path(txt_path)
    if not path.exists():
        print(f"[harness] 记录不存在，跳过写回: {path}")
        return None
    transcript = read_transcript_file(path)
    record_stem = stem or path.stem
    return add_meeting_memory(transcript, stem=record_stem)


def schedule_memory_writeback(txt_path: str | Path | None, *, stem: str = "") -> None:
    """后台写回记忆，不阻塞串口主循环。"""
    settings = harness_settings()
    if not settings.get("enabled", True) or not settings.get("write_memory_on_end", True):
        return
    if not txt_path:
        return

    path = Path(txt_path)
    record_stem = stem or path.stem
    with _SCHEDULED_LOCK:
        if record_stem in _SCHEDULED_WRITEBACK_STEMS:
            print(f"[harness] 已写过记忆，跳过重复写回: {record_stem}", flush=True)
            return
        _SCHEDULED_WRITEBACK_STEMS.add(record_stem)

    def _worker():
        writeback_from_record_file(path, stem=record_stem)

    threading.Thread(target=_worker, daemon=True, name="harness-writeback").start()
    print(f"[harness] 已调度记忆写回: {path.name}", flush=True)


def reset_writeback_schedule() -> None:
    """新一轮录音开始时清除写回去重标记。"""
    with _SCHEDULED_LOCK:
        _SCHEDULED_WRITEBACK_STEMS.clear()


def augment_for_llm_refine(meeting_text: str) -> tuple[str, list[dict[str, Any]]]:
    """检索记忆并返回 LLM 精炼用增强文本。"""
    settings = harness_settings()
    if not settings.get("enabled", True) or not settings.get("enhance_extract", True):
        return meeting_text, []

    query = meeting_text[: int(settings["query_chars"])]
    memories = search_memories(query)
    if not memories:
        return meeting_text, []
    print(f"[harness] 检索到 {len(memories)} 条相关记忆，注入 LLM 上下文", flush=True)
    return build_memory_augmented_text(meeting_text, memories), memories
