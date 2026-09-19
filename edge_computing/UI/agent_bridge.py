"""会议 Agent：从转写文本提取总结与任务，供 UI 调用。"""
from __future__ import annotations

import json
import os
import queue as queue_mod
import re
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from paths import PROJECT_ROOT, UI_DIR, apply_runtime_paths, load_config

FIBO_SDK_LIB = Path("/usr/local/lib/python3.8/dist-packages/fiboaisdk")
AGENT_WORKER = UI_DIR / "agent_worker.py"

_WORKER_MANAGER: "LlmWorkerManager | None" = None
_WORKER_LOCK = threading.Lock()


def _llm_worker_settings() -> dict:
    cfg = load_config()
    worker_cfg = cfg.get("llm_worker") or {}
    defaults = {
        "warmup_on_start": True,
        "transcript_max_lines": 40,
        "transcript_max_chars": 3500,
        "merge_same_speaker": True,
        "request_timeout_sec": 900,
        "warmup_timeout_sec": 180,
    }
    defaults.update(worker_cfg)
    return defaults


def _meeting_extract_settings() -> dict:
    cfg = load_config()
    ext_cfg = cfg.get("meeting_extract") or {}
    defaults = {
        "mode": "rules_llm",
        "llm_refine": "auto",
        "preview_rules_first": True,
        "refine_snippet_max_chars": 400,
    }
    defaults.update(ext_cfg)
    return defaults


def _should_skip_llm_refine(compressed_text: str, draft: dict, cfg: dict) -> bool:
    """llm_refine: never=跳过, always=必跑, auto=草稿足够好时跳过。"""
    mode = cfg.get("llm_refine", "auto")
    if mode == "never":
        return True
    if mode == "always":
        return False

    args = draft["function_call"]["arguments"]
    tasks = args.get("tasks") or []
    summary = (args.get("summary") or "").strip()
    if not summary and not tasks:
        return True
    if len(compressed_text.strip()) < 120:
        return True

    if summary and len(summary) >= 10 and tasks:
        with_assignee = sum(1 for item in tasks if (item.get("assignee") or "").strip())
        if with_assignee >= max(1, len(tasks) // 2):
            return True
    return False


def _return_rules_draft(
    draft: dict,
    *,
    rules_sec: float,
    compress_meta: dict | None,
    out_name: str | None,
    reason: str,
) -> tuple[dict, dict, str]:
    args = draft["function_call"]["arguments"]
    metadata = {
        "mode": "rules_llm",
        "pipeline": "rules_llm",
        "parse_status": "rules_only_skipped_llm",
        "llm_skipped": True,
        "llm_skip_reason": reason,
        "rules_sec": round(rules_sec, 4),
        "init_sec": 0.0,
        "generate_sec": 0.0,
        "rules_tasks": len(args.get("tasks") or []),
        "rules_decisions": len(args.get("decisions") or []),
    }
    if compress_meta:
        metadata["transcript_compress"] = compress_meta
    _save_task_list_outputs(draft, metadata, "", out_name)
    print(
        f"[规则] 跳过 LLM 精炼 ({reason}) "
        f"tasks={metadata['rules_tasks']} decisions={metadata['rules_decisions']}",
        flush=True,
    )
    return draft, metadata, ""


def _save_task_list_outputs(
    task_list: dict,
    metadata: dict,
    raw_output: str,
    out_name: str | None,
) -> None:
    agent = _configure_agent_module()
    agent.save_outputs(task_list, raw_output, metadata, out_name)


def _run_rules_only(compressed_text: str, compress_meta: dict, out_name: str | None) -> tuple[dict, dict, str]:
    from fast_meeting_extract import extract_meeting_rules

    t0 = time.perf_counter()
    task_list = extract_meeting_rules(compressed_text)
    rules_sec = time.perf_counter() - t0
    args = task_list["function_call"]["arguments"]
    metadata = {
        "mode": "fast_rules",
        "pipeline": "fast",
        "parse_status": "rules_only",
        "rules_sec": round(rules_sec, 4),
        "init_sec": 0.0,
        "generate_sec": 0.0,
        "rules_tasks": len(args.get("tasks") or []),
        "rules_decisions": len(args.get("decisions") or []),
    }
    if compress_meta:
        metadata["transcript_compress"] = compress_meta
    _save_task_list_outputs(task_list, metadata, "", out_name)
    print(
        f"[规则] 解析完成 {rules_sec:.3f}s "
        f"tasks={metadata['rules_tasks']} decisions={metadata['rules_decisions']}",
        flush=True,
    )
    return task_list, metadata, ""


# 说话人标签：说话人N / SpeakerN / 声纹注册姓名（如「徐」）
_SPEAKER_LINE_RE = re.compile(
    r"^(说话人\d+|Speaker\d+|[\u4e00-\u9fff]{1,4}|[A-Za-z][\w\-]{0,31})\s*:\s*(.+)$",
    re.I,
)


def compress_transcript_for_llm(
    text: str,
    *,
    max_lines: int | None = None,
    max_chars: int | None = None,
    merge_same_speaker: bool | None = None,
) -> tuple[str, dict[str, Any]]:
    """压缩转写文本：合并同说话人、保留最近内容，降低 LLM 输入长度。"""
    settings = _llm_worker_settings()
    max_lines = int(max_lines if max_lines is not None else settings["transcript_max_lines"])
    max_chars = int(max_chars if max_chars is not None else settings["transcript_max_chars"])
    merge_same_speaker = (
        settings["merge_same_speaker"]
        if merge_same_speaker is None
        else merge_same_speaker
    )

    raw_lines = [line.strip() for line in text.splitlines() if line.strip()]
    meta: dict[str, Any] = {
        "original_lines": len(raw_lines),
        "original_chars": len(text),
    }

    merged: list[str] = []
    if merge_same_speaker:
        speaker_re = _SPEAKER_LINE_RE
        for line in raw_lines:
            match = speaker_re.match(line)
            if not match:
                merged.append(line)
                continue
            speaker, content = match.group(1), match.group(2).strip()
            if merged:
                prev_match = speaker_re.match(merged[-1])
                if prev_match and prev_match.group(1) == speaker:
                    merged[-1] = f"{speaker}: {prev_match.group(2).strip()} {content}".strip()
                    continue
            merged.append(f"{speaker}: {content}")
    else:
        merged = list(raw_lines)

    if len(merged) > max_lines:
        merged = merged[-max_lines:]
        meta["truncated_by_lines"] = True
    else:
        meta["truncated_by_lines"] = False

    compressed = "\n".join(merged).strip()
    if len(compressed) > max_chars:
        compressed = compressed[-max_chars:].lstrip()
        meta["truncated_by_chars"] = True
    else:
        meta["truncated_by_chars"] = False

    meta["compressed_lines"] = len(merged)
    meta["compressed_chars"] = len(compressed)
    return compressed, meta


HARNESS_TRANSCRIPT_MARKER = "【待分析会议转写】"


def compress_for_llm_refine(text: str) -> tuple[str, dict[str, Any]]:
    """压缩 LLM 输入；若含 Harness 记忆前缀则只压缩转写段，保留记忆上下文。"""
    marker = HARNESS_TRANSCRIPT_MARKER
    if marker in text:
        idx = text.index(marker)
        prefix = text[: idx + len(marker)] + "\n"
        body = text[idx + len(marker) :].strip()
        compressed_body, meta = compress_transcript_for_llm(body)
        meta["harness_prefix"] = True
        return prefix + compressed_body, meta
    return compress_transcript_for_llm(text)


def _configure_agent_module():
    runtime_dir = PROJECT_ROOT / "ai_runtime"
    if str(runtime_dir) not in sys.path:
        sys.path.insert(0, str(runtime_dir))
    import agent_task_cli as agent  # noqa: WPS433

    runtime = apply_runtime_paths()
    agent.configure_agent_paths(
        base=PROJECT_ROOT,
        license_dir=runtime["license_dir"],
        model_root=PROJECT_ROOT / "ai_models" / "llm_models",
        out_dir=runtime["agent_out_dir"],
    )
    return agent


def extract_transcript_text(content: str) -> str:
    """从 record TXT 中提取可用于 LLM 的转写正文。"""
    lines: list[str] = []
    for raw in content.splitlines():
        stripped = raw.strip()
        if not stripped:
            continue
        if stripped.startswith("录音开始") or stripped.startswith("录音结束"):
            continue
        if stripped.startswith("---") or re.fullmatch(r"[-=*═]{10,}", stripped):
            continue
        if stripped.startswith("[会议总结]") or stripped.startswith("[会议日程]"):
            break
        if stripped.startswith("  "):
            continue
        match = re.match(r"^\[\d{2}:\d{2}:\d{2}\]\s*(.+)$", stripped)
        if match:
            lines.append(match.group(1).strip())
        else:
            # 保留说话人N / SpeakerN / 声纹姓名（如「徐:」）及无标签转写行
            lines.append(stripped)
    return "\n".join(lines).strip()


def read_transcript_file(path: Path | str) -> str:
    content = Path(path).read_text(encoding="utf-8")
    return extract_transcript_text(content)


def _needs_isolated_llm() -> bool:
    """ASR(sherpa_onnx) 与 LLM(fiboaisdk) 依赖不同版本 onnxruntime，需隔离。"""
    return "sherpa_onnx" in sys.modules or "onnxruntime" in sys.modules


def _fibo_env() -> dict[str, str]:
    env = os.environ.copy()
    fibo_lib = str(FIBO_SDK_LIB)
    if env.get("LD_LIBRARY_PATH"):
        env["LD_LIBRARY_PATH"] = fibo_lib + ":" + env["LD_LIBRARY_PATH"]
    else:
        env["LD_LIBRARY_PATH"] = fibo_lib
    return env


class LlmWorkerManager:
    """常驻 LLM 子进程：启动时预热模型，sum/agenda 复用。"""

    def __init__(self):
        self._proc: subprocess.Popen | None = None
        self._stderr_thread: threading.Thread | None = None
        self._reader_thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._req_id = 0
        self._ready = threading.Event()
        self._ready_model: str | None = None
        self._start_error: str | None = None
        self._warmup_error: str | None = None
        self._pending: dict[int, queue_mod.Queue] = {}

    @property
    def ready(self) -> bool:
        return self._ready.is_set() and self._proc is not None and self._proc.poll() is None

    def _next_id(self) -> int:
        self._req_id += 1
        return self._req_id

    def _drain_stderr(self) -> None:
        proc = self._proc
        if proc is None or proc.stderr is None:
            return
        for line in proc.stderr:
            line = line.strip()
            if line.startswith("[license]") or line.startswith("[llm]") or line.startswith("[规则]"):
                print(line, flush=True)

    def _reader_loop(self) -> None:
        proc = self._proc
        if proc is None or proc.stdout is None:
            return
        for line in proc.stdout:
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue

            event = payload.get("event")
            if event == "ready" and payload.get("ok"):
                with self._lock:
                    self._ready_model = payload.get("model")
                self._ready.set()
                continue
            if event == "error":
                self._warmup_error = payload.get("error") or "worker warmup failed"
                self._ready.set()
                continue

            req_id = payload.get("id")
            if req_id is not None and req_id in self._pending:
                self._pending[req_id].put(payload)

    def _spawn_worker(self, model_name: str) -> bool:
        if not AGENT_WORKER.exists():
            self._start_error = f"worker script missing: {AGENT_WORKER}"
            print(f"[LLM] {self._start_error}")
            return False

        self._ready.clear()
        self._warmup_error = None
        self._pending.clear()

        try:
            self._proc = subprocess.Popen(
                [sys.executable, str(AGENT_WORKER), "--daemon"],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                env=_fibo_env(),
                cwd=str(UI_DIR),
            )
        except Exception as exc:
            self._start_error = str(exc)
            print(f"[LLM] 启动 worker 失败: {exc}")
            return False

        self._stderr_thread = threading.Thread(target=self._drain_stderr, daemon=True)
        self._stderr_thread.start()
        self._reader_thread = threading.Thread(
            target=self._reader_loop,
            daemon=True,
            name="llm-worker-stdout",
        )
        self._reader_thread.start()
        return True

    def start(self, *, model: str | None = None, block: bool = True) -> bool:
        settings = _llm_worker_settings()
        model_name = model or load_config().get("llm_model", "qwen")

        with self._lock:
            if self.ready:
                return True
            if self._proc is None or self._proc.poll() is not None:
                self.shutdown_unlocked()
                if not self._spawn_worker(model_name):
                    return False

        if not block:
            return True

        if self._ready.wait(timeout=float(settings["warmup_timeout_sec"])):
            if self._warmup_error:
                self._start_error = self._warmup_error
                print(f"[LLM] {self._start_error}")
                self.shutdown()
                return False
            if self.ready:
                meta_model = self._ready_model or model_name
                print(f"[LLM] 常驻 worker 就绪 model={meta_model}", flush=True)
                return True

        self._start_error = f"worker warmup timeout ({settings['warmup_timeout_sec']:.0f}s)"
        print(f"[LLM] {self._start_error}")
        self.shutdown()
        return False

    def _request(self, payload: dict, *, timeout: float | None = None) -> dict:
        settings = _llm_worker_settings()
        cmd = payload.get("cmd", "")
        if timeout is None:
            if cmd == "refine":
                timeout = float(settings.get("refine_timeout_sec", 420))
            else:
                timeout = float(settings["request_timeout_sec"])
        timeout = float(timeout)
        heartbeat = float(settings.get("progress_heartbeat_sec", 30))

        req_id = payload.get("id")
        if req_id is None:
            raise ValueError("request payload missing id")

        response_q: queue_mod.Queue = queue_mod.Queue()
        self._pending[req_id] = response_q

        with self._lock:
            if not self.ready:
                self._pending.pop(req_id, None)
                raise RuntimeError(self._start_error or "LLM worker 未就绪")
            proc = self._proc
            assert proc is not None and proc.stdin is not None
            proc.stdin.write(json.dumps(payload, ensure_ascii=False) + "\n")
            proc.stdin.flush()

        start = time.perf_counter()
        last_heartbeat = start
        try:
            while time.perf_counter() - start < timeout:
                try:
                    return response_q.get(timeout=1.0)
                except queue_mod.Empty:
                    proc = self._proc
                    if proc is None or proc.poll() is not None:
                        raise RuntimeError("LLM worker 意外退出")
                    elapsed = time.perf_counter() - start
                    if cmd == "refine" and (time.perf_counter() - last_heartbeat) >= heartbeat:
                        print(f"[LLM] 精炼进行中… {elapsed:.0f}s", flush=True)
                        last_heartbeat = time.perf_counter()
        finally:
            self._pending.pop(req_id, None)

        raise TimeoutError(f"LLM worker 请求超时 ({timeout:.0f}s)")

    def extract(
        self,
        meeting_text: str,
        *,
        model: str | None = None,
        out_name: str | None = None,
        allow_fallback: bool = True,
        compress_meta: dict | None = None,
        pipeline: str = "rules_llm",
    ) -> tuple[dict, dict, str]:
        if not self.ready and not self.start(block=True):
            raise RuntimeError(self._start_error or "LLM worker 启动失败")

        req_id = self._next_id()
        resp = self._request(
            {
                "id": req_id,
                "cmd": "extract",
                "pipeline": pipeline,
                "meeting_text": meeting_text,
                "model": model or load_config().get("llm_model", "qwen"),
                "out_name": out_name,
                "allow_fallback": allow_fallback,
            }
        )
        if not resp.get("ok"):
            raise RuntimeError(resp.get("error") or "LLM worker extract failed")

        metadata = resp.get("metadata") or {}
        if compress_meta:
            metadata["transcript_compress"] = compress_meta
        return resp["task_list"], metadata, resp.get("raw_output") or ""

    def refine(
        self,
        meeting_text: str,
        draft_task_list: dict,
        *,
        model: str | None = None,
        out_name: str | None = None,
        allow_fallback: bool = True,
        compress_meta: dict | None = None,
    ) -> tuple[dict, dict, str]:
        if not self.ready and not self.start(block=True):
            raise RuntimeError(self._start_error or "LLM worker 启动失败")

        req_id = self._next_id()
        refine_cfg = _meeting_extract_settings()
        resp = self._request(
            {
                "id": req_id,
                "cmd": "refine",
                "meeting_text": meeting_text,
                "draft_task_list": draft_task_list,
                "model": model or load_config().get("llm_model", "qwen"),
                "out_name": out_name,
                "allow_fallback": allow_fallback,
                "snippet_max_chars": int(refine_cfg.get("refine_snippet_max_chars", 400)),
            }
        )
        if not resp.get("ok"):
            raise RuntimeError(resp.get("error") or "LLM worker refine failed")

        metadata = resp.get("metadata") or {}
        if compress_meta:
            metadata["transcript_compress"] = compress_meta
        return resp["task_list"], metadata, resp.get("raw_output") or ""

    def shutdown_unlocked(self) -> None:
        proc = self._proc
        if proc is not None and proc.poll() is None and proc.stdin is not None:
            try:
                req_id = self._next_id()
                proc.stdin.write(
                    json.dumps({"id": req_id, "cmd": "shutdown"}, ensure_ascii=False) + "\n"
                )
                proc.stdin.flush()
            except Exception:
                pass
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
        self._proc = None
        self._ready.clear()
        self._ready_model = None
        self._warmup_error = None
        self._pending.clear()

    def shutdown(self) -> None:
        with self._lock:
            self.shutdown_unlocked()
        print("[LLM] 常驻 worker 已关闭", flush=True)


def get_llm_worker_manager() -> LlmWorkerManager:
    global _WORKER_MANAGER
    with _WORKER_LOCK:
        if _WORKER_MANAGER is None:
            _WORKER_MANAGER = LlmWorkerManager()
        return _WORKER_MANAGER


def warmup_llm_worker(*, model: str | None = None, background: bool = True) -> None:
    settings = _llm_worker_settings()
    if not settings.get("warmup_on_start", True):
        print("[LLM] 已配置跳过启动预热")
        return

    manager = get_llm_worker_manager()

    def _run():
        ok = manager.start(model=model, block=True)
        if not ok:
            print("[LLM] 启动预热失败，sum/agenda 将尝试临时 worker")

    if background:
        threading.Thread(target=_run, daemon=True, name="llm-worker-warmup").start()
        print("[LLM] 正在后台预热会议纪要模型…")
    else:
        _run()


def shutdown_llm_worker() -> None:
    global _WORKER_MANAGER
    with _WORKER_LOCK:
        if _WORKER_MANAGER is not None:
            _WORKER_MANAGER.shutdown()
            _WORKER_MANAGER = None


def _parse_worker_payload(stdout: str, *, out_name: str | None = None) -> dict:
    """解析 worker stdout；若被 SDK 日志污染则尝试末行 JSON 或读取落盘文件。"""
    text = (stdout or "").strip()
    if text:
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            for line in reversed(text.splitlines()):
                line = line.strip()
                if not line.startswith("{"):
                    continue
                try:
                    return json.loads(line)
                except json.JSONDecodeError:
                    continue

    if out_name:
        runtime = apply_runtime_paths()
        saved = runtime["agent_out_dir"] / f"{out_name}.json"
        if saved.exists():
            payload = json.loads(saved.read_text(encoding="utf-8"))
            raw_path = runtime["agent_out_dir"] / f"{out_name}_raw.txt"
            return {
                "task_list": payload["task_list"],
                "metadata": payload.get("metadata", {}),
                "raw_output": raw_path.read_text(encoding="utf-8", errors="ignore")
                if raw_path.exists()
                else "",
            }

    preview = text[:120].replace("\n", " ") if text else "(空)"
    raise RuntimeError(f"无法解析 agent worker 输出: {preview}")


def _run_agent_extract_direct(
    meeting_text: str,
    *,
    model: str | None = None,
    out_name: str | None = None,
    allow_fallback: bool = True,
    compress_meta: dict | None = None,
) -> tuple[dict, dict, str]:
    """在当前进程直接调用 LLM（不可与 sherpa_onnx 共存）。"""
    agent = _configure_agent_module()
    model_name = model or load_config().get("llm_model", "qwen")
    metadata_extra = {"mode": "ui_extract_direct"}
    if compress_meta:
        metadata_extra["transcript_compress"] = compress_meta

    task_list, metadata, raw_output = agent.extract_meeting_tasks(
        meeting_text,
        model_name=model_name,
        out_name=out_name,
        allow_fallback=allow_fallback,
        session=None,
        metadata_extra=metadata_extra,
    )
    return task_list, metadata, raw_output


def _run_agent_extract_subprocess(
    meeting_text: str,
    *,
    model: str | None = None,
    out_name: str | None = None,
    allow_fallback: bool = True,
    compress_meta: dict | None = None,
    pipeline: str = "llm",
    rules_sec: float | None = None,
) -> tuple[dict, dict, str]:
    """一次性子进程（daemon 不可用时的兜底）。"""
    req = {
        "meeting_text": meeting_text,
        "model": model,
        "out_name": out_name,
        "allow_fallback": allow_fallback,
        "pipeline": pipeline,
    }
    proc = subprocess.run(
        [sys.executable, str(AGENT_WORKER)],
        input=json.dumps(req, ensure_ascii=False),
        capture_output=True,
        text=True,
        env=_fibo_env(),
        timeout=_llm_worker_settings()["request_timeout_sec"],
        cwd=str(UI_DIR),
    )
    if proc.stderr:
        for line in proc.stderr.splitlines():
            line = line.strip()
            if line.startswith("[license]") or line.startswith("[llm]"):
                print(line, flush=True)

    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout or "").strip()
        raise RuntimeError(detail or f"agent worker exit {proc.returncode}")

    payload = _parse_worker_payload(proc.stdout, out_name=out_name)
    if payload.get("error"):
        raise RuntimeError(payload["error"])

    metadata = payload.get("metadata") or {}
    if compress_meta:
        metadata["transcript_compress"] = compress_meta
    if rules_sec is not None:
        metadata.setdefault("rules_sec", round(rules_sec, 4))
    return payload["task_list"], metadata, payload.get("raw_output") or ""


def extract_rules_draft(meeting_text: str) -> tuple[dict, dict]:
    """压缩转写并生成规则草稿（供 UI 先屏显）。"""
    compressed_text, compress_meta = compress_transcript_for_llm(meeting_text)
    from fast_meeting_extract import extract_meeting_rules

    t0 = time.perf_counter()
    draft = extract_meeting_rules(compressed_text)
    rules_sec = time.perf_counter() - t0
    meta = {
        "rules_sec": round(rules_sec, 4),
        "transcript_compress": compress_meta,
    }
    return draft, meta


def should_run_llm_refine(meeting_text: str, draft: dict) -> bool:
    """是否应对规则草稿继续跑 LLM 精炼（供 UI 判断是否显示「优化中」）。"""
    extract_cfg = _meeting_extract_settings()
    compressed_text, _ = compress_for_llm_refine(meeting_text)
    return not _should_skip_llm_refine(compressed_text, draft, extract_cfg)


def run_agent_llm_refine(
    meeting_text: str,
    draft_task_list: dict,
    *,
    model: str | None = None,
    out_name: str | None = None,
    allow_fallback: bool = True,
) -> tuple[dict, dict, str]:
    """仅 LLM 精炼已有规则草稿（UI 已屏显草稿后调用）。"""
    extract_cfg = _meeting_extract_settings()
    compressed_text, compress_meta = compress_for_llm_refine(meeting_text)

    if _should_skip_llm_refine(compressed_text, draft_task_list, extract_cfg):
        return _return_rules_draft(
            draft_task_list,
            rules_sec=0.0,
            compress_meta=compress_meta,
            out_name=out_name,
            reason="llm_refine=auto_draft_ok",
        )

    snippet_max = int(extract_cfg.get("refine_snippet_max_chars", 400))
    print(
        "[LLM] 开始后台精炼（端侧约 1–5 分钟，日志每 30s 心跳）",
        flush=True,
    )

    manager = get_llm_worker_manager()
    try:
        if manager.ready or _needs_isolated_llm():
            if not manager.ready:
                manager.start(block=True)
            if manager.ready:
                return manager.refine(
                    compressed_text,
                    draft_task_list,
                    model=model,
                    out_name=out_name,
                    allow_fallback=allow_fallback,
                    compress_meta=compress_meta,
                )
    except TimeoutError as exc:
        print(f"[LLM] 精炼超时，保留规则草稿: {exc}", flush=True)
        try:
            manager.shutdown()
        except Exception:
            pass
        if allow_fallback:
            metadata = {
                "mode": "rules_llm_refine_timeout",
                "pipeline": "rules_llm",
                "parse_status": "refine_timeout_fallback",
                "transcript_compress": compress_meta,
            }
            return draft_task_list, metadata, ""
        raise
    except Exception as exc:
        print(f"[LLM] 常驻 worker 精炼失败: {exc}", flush=True)

    if _needs_isolated_llm() or AGENT_WORKER.exists():
        req = {
            "meeting_text": compressed_text,
            "draft_task_list": draft_task_list,
            "model": model,
            "out_name": out_name,
            "allow_fallback": allow_fallback,
            "pipeline": "rules_llm",
            "snippet_max_chars": snippet_max,
        }
        proc = subprocess.run(
            [sys.executable, str(AGENT_WORKER)],
            input=json.dumps(req, ensure_ascii=False),
            capture_output=True,
            text=True,
            env=_fibo_env(),
            timeout=_llm_worker_settings().get("refine_timeout_sec", 420),
            cwd=str(UI_DIR),
        )
        if proc.stderr:
            for line in proc.stderr.splitlines():
                line = line.strip()
                if line.startswith("[license]") or line.startswith("[llm]"):
                    print(line, flush=True)
        if proc.returncode != 0:
            detail = (proc.stderr or proc.stdout or "").strip()
            raise RuntimeError(detail or f"agent worker exit {proc.returncode}")
        payload = _parse_worker_payload(proc.stdout, out_name=out_name)
        if payload.get("error"):
            raise RuntimeError(payload["error"])
        metadata = payload.get("metadata") or {}
        metadata["transcript_compress"] = compress_meta
        return payload["task_list"], metadata, payload.get("raw_output") or ""

    agent = _configure_agent_module()
    model_name = model or load_config().get("llm_model", "qwen")
    return agent.refine_meeting_tasks(
        draft_task_list,
        compressed_text,
        model_name=model_name,
        out_name=out_name,
        allow_fallback=allow_fallback,
        session=None,
        metadata_extra={"transcript_compress": compress_meta},
        snippet_max_chars=snippet_max,
    )


def run_agent_extract(
    meeting_text: str,
    *,
    model: str | None = None,
    out_name: str | None = None,
    allow_fallback: bool = True,
    pipeline: str | None = None,
    status_cb=None,
    draft_cb=None,
) -> tuple[dict, dict, str]:
    """提取会议 JSON。默认 rules_llm：规则解析 + LLM 精炼。"""
    if not meeting_text.strip():
        raise ValueError("转写内容为空，请先录音")

    extract_cfg = _meeting_extract_settings()
    pipeline = pipeline or extract_cfg.get("mode", "rules_llm")
    rules_sec = None
    draft = None

    compressed_text, compress_meta = compress_transcript_for_llm(meeting_text)
    if compress_meta.get("compressed_chars", 0) < len(meeting_text.strip()):
        print(
            "[LLM] 转写已压缩: "
            f"{compress_meta.get('original_lines')}行/{compress_meta.get('original_chars')}字 -> "
            f"{compress_meta.get('compressed_lines')}行/{compress_meta.get('compressed_chars')}字",
            flush=True,
        )

    if pipeline == "fast":
        return _run_rules_only(compressed_text, compress_meta, out_name)

    if pipeline == "rules_llm":
        from fast_meeting_extract import extract_meeting_rules

        t0 = time.perf_counter()
        draft = extract_meeting_rules(compressed_text)
        rules_sec = time.perf_counter() - t0
        draft_args = draft["function_call"]["arguments"]
        print(
            f"[规则] 草稿完成 {rules_sec:.3f}s "
            f"tasks={len(draft_args.get('tasks') or [])} "
            f"decisions={len(draft_args.get('decisions') or [])}",
            flush=True,
        )

        if _should_skip_llm_refine(compressed_text, draft, extract_cfg):
            return _return_rules_draft(
                draft,
                rules_sec=rules_sec,
                compress_meta=compress_meta,
                out_name=out_name,
                reason="llm_refine=never",
            )

        if extract_cfg.get("preview_rules_first", True) and draft_cb:
            draft_cb(draft)
        print("[规则] LLM 精炼中（端侧约需 1–5 分钟）…", flush=True)
        # 勿用 status_cb 覆盖已显示的规则草稿

    manager = get_llm_worker_manager()
    if manager.ready or _needs_isolated_llm():
        try:
            if not manager.ready:
                manager.start(block=True)
            if manager.ready:
                task_list, metadata, raw = manager.extract(
                    compressed_text,
                    model=model,
                    out_name=out_name,
                    allow_fallback=allow_fallback,
                    compress_meta=compress_meta,
                    pipeline=pipeline,
                )
                if pipeline == "rules_llm":
                    metadata.setdefault("rules_sec", round(rules_sec, 4))
                return task_list, metadata, raw
        except Exception as exc:
            print(f"[LLM] 常驻 worker 失败，回退一次性进程: {exc}", flush=True)

    if pipeline == "rules_llm":
        return _run_agent_extract_subprocess(
            compressed_text,
            model=model,
            out_name=out_name,
            allow_fallback=allow_fallback,
            compress_meta=compress_meta,
            pipeline=pipeline,
            rules_sec=rules_sec,
        )

    if _needs_isolated_llm() or AGENT_WORKER.exists():
        return _run_agent_extract_subprocess(
            compressed_text,
            model=model,
            out_name=out_name,
            allow_fallback=allow_fallback,
            compress_meta=compress_meta,
            pipeline="llm",
        )
    return _run_agent_extract_direct(
        compressed_text,
        model=model,
        out_name=out_name,
        allow_fallback=allow_fallback,
        compress_meta=compress_meta,
    )


def format_summary_lines(task_list: dict, *, banner: str | None = None) -> list[str]:
    args = task_list["function_call"]["arguments"]
    title = args.get("meeting_title") or "未命名会议"
    summary = (args.get("summary") or "").strip()
    participants = args.get("participants") or []
    decisions = args.get("decisions") or []

    lines = [
        "********** 会议总结 **********",
        "",
    ]
    if banner:
        lines.extend([banner, ""])
    lines.append(f"会议主题：{title}")
    if participants:
        lines.append(f"参会人：{'、'.join(str(p) for p in participants)}")
    lines.append("")

    if summary:
        lines.append("【摘要】")
        for part in re.split(r"(?<=[。！？!?])\s*", summary):
            part = part.strip()
            if part:
                lines.append(f"  {part}")
        lines.append("")

    if decisions:
        lines.append("【会议决策】")
        for idx, item in enumerate(decisions, 1):
            text = item.get("decision") or ""
            owner = item.get("owner")
            due = item.get("due_date")
            suffix = []
            if owner:
                suffix.append(f"负责人:{owner}")
            if due:
                suffix.append(f"日期:{due}")
            extra = f" ({'，'.join(suffix)})" if suffix else ""
            lines.append(f"  {idx}. {text}{extra}")
        lines.append("")

    lines.append("********** 总结完毕 **********")
    return lines


def _display_width(text: str) -> int:
    width = 0
    for ch in text:
        width += 2 if ord(ch) > 127 else 1
    return width


def _truncate_display(text: str, max_width: int) -> str:
    text = (text or "").strip()
    if _display_width(text) <= max_width:
        return text
    out: list[str] = []
    used = 0
    for ch in text:
        step = 2 if ord(ch) > 127 else 1
        if used + step > max_width - 1:
            break
        out.append(ch)
        used += step
    return "".join(out) + "…"


def format_agenda_lines(task_list: dict, *, banner: str | None = None) -> list[str]:
    tasks = task_list["function_call"]["arguments"].get("tasks") or []
    lines: list[str] = [
        "********** 会议任务 **********",
        "",
    ]
    if banner:
        lines.extend([banner, ""])
    if not tasks:
        lines.append("暂无任务，请先录音或检查转写。")
    else:
        for idx, item in enumerate(tasks, 1):
            task = _truncate_display(item.get("task") or "未命名任务", 36)
            assignee = _truncate_display(item.get("assignee") or "待定", 10)
            due = item.get("due_date")
            due_text = str(due).strip() if due else "-"
            lines.append(f"{idx}. [{assignee}] {task}")
            if due_text and due_text != "-":
                lines.append(f"   截止: {due_text}")
    lines.extend(["", "********** 列表完毕 **********"])
    return lines
