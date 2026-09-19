#!/usr/bin/env python3
"""LLM Agent worker：单次模式或常驻 daemon（模型复用）。"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

UI_DIR = Path(__file__).resolve().parent
if str(UI_DIR) not in sys.path:
    sys.path.insert(0, str(UI_DIR))

PROJECT_ROOT = UI_DIR.parent
RUNTIME_DIR = PROJECT_ROOT / "ai_runtime"
if str(RUNTIME_DIR) not in sys.path:
    sys.path.insert(0, str(RUNTIME_DIR))

from fast_meeting_extract import extract_meeting_rules  # noqa: E402
from paths import apply_runtime_paths, load_config  # noqa: E402


def _redirect_stdout_to_stderr():
    """fiboaisdk 在 C 层写 fd=1，需 dup2 才能真正隔离 stdout。"""
    saved = os.dup(1)
    os.dup2(2, 1)
    return saved


def _restore_stdout(saved_fd: int) -> None:
    os.dup2(saved_fd, 1)
    os.close(saved_fd)


def _configure_agent_module():
    import agent_task_cli as agent  # noqa: WPS433

    runtime = apply_runtime_paths()
    agent.configure_agent_paths(
        base=PROJECT_ROOT,
        license_dir=runtime["license_dir"],
        model_root=PROJECT_ROOT / "ai_models" / "llm_models",
        out_dir=runtime["agent_out_dir"],
    )
    return agent


def _emit_response(payload: dict) -> None:
    sys.stdout.write(json.dumps(payload, ensure_ascii=False) + "\n")
    sys.stdout.flush()


def _resolve_pipeline(req: dict) -> str:
    pipeline = req.get("pipeline")
    if pipeline:
        return pipeline
    cfg = load_config().get("meeting_extract") or {}
    return cfg.get("mode", "rules_llm")


def _run_extract_pipeline(req: dict, agent, session=None) -> tuple[dict, dict, str]:
    meeting_text = req.get("meeting_text", "")
    model_name = req.get("model") or load_config().get("llm_model", "qwen")
    out_name = req.get("out_name")
    allow_fallback = req.get("allow_fallback", True)
    pipeline = _resolve_pipeline(req)

    if pipeline == "fast":
        t0 = time.perf_counter()
        task_list = extract_meeting_rules(meeting_text)
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
        agent.save_outputs(task_list, "", metadata, out_name)
        return task_list, metadata, ""

    if pipeline == "rules_llm":
        t0 = time.perf_counter()
        draft = extract_meeting_rules(meeting_text)
        rules_sec = time.perf_counter() - t0
        args = draft["function_call"]["arguments"]
        print(
            f"[规则] 草稿 {rules_sec:.3f}s tasks={len(args.get('tasks') or [])} "
            f"decisions={len(args.get('decisions') or [])}",
            flush=True,
        )
        metadata_extra = {
            "pipeline": "rules_llm",
            "rules_sec": round(rules_sec, 4),
            "rules_tasks": len(args.get("tasks") or []),
            "rules_decisions": len(args.get("decisions") or []),
        }
        return agent.refine_meeting_tasks(
            draft,
            meeting_text,
            model_name=model_name,
            out_name=out_name,
            allow_fallback=allow_fallback,
            session=session,
            metadata_extra=metadata_extra,
        )

    metadata_extra = {"mode": "ui_extract_llm", "pipeline": "llm"}
    return agent.extract_meeting_tasks(
        meeting_text,
        model_name=model_name,
        out_name=out_name,
        allow_fallback=allow_fallback,
        session=session,
        metadata_extra=metadata_extra,
    )


def _run_once(req: dict, agent) -> dict:
    saved_stdout = _redirect_stdout_to_stderr()
    try:
        if req.get("draft_task_list"):
            snippet_max = int(req.get("snippet_max_chars") or 400)
            task_list, metadata, raw_output = agent.refine_meeting_tasks(
                req["draft_task_list"],
                req.get("meeting_text", ""),
                model_name=req.get("model") or load_config().get("llm_model", "qwen"),
                out_name=req.get("out_name"),
                allow_fallback=req.get("allow_fallback", True),
                session=None,
                metadata_extra={"pipeline": "rules_llm"},
                snippet_max_chars=snippet_max,
            )
        else:
            task_list, metadata, raw_output = _run_extract_pipeline(req, agent, session=None)
    finally:
        _restore_stdout(saved_stdout)

    return {
        "ok": True,
        "task_list": task_list,
        "metadata": metadata,
        "raw_output": raw_output,
    }


def _handle_daemon_request(req: dict, agent, session) -> dict:
    req_id = req.get("id")
    cmd = req.get("cmd")
    model_name = req.get("model") or load_config().get("llm_model", "qwen")

    if cmd == "ping":
        return {
            "id": req_id,
            "ok": True,
            "ready": session.ready,
            "model": session.model_name,
        }

    if cmd == "warmup":
        saved_stdout = _redirect_stdout_to_stderr()
        try:
            meta = session.load_model(model_name)
        finally:
            _restore_stdout(saved_stdout)
        return {
            "id": req_id,
            "ok": True,
            "ready": session.ready,
            "metadata": meta,
        }

    if cmd == "refine":
        saved_stdout = _redirect_stdout_to_stderr()
        try:
            draft = req.get("draft_task_list")
            if not draft:
                raise ValueError("refine requires draft_task_list")
            meeting_text = req.get("meeting_text", "")
            allow_fallback = req.get("allow_fallback", True)
            metadata_extra = {
                "pipeline": "rules_llm",
                "rules_sec": req.get("rules_sec", 0.0),
            }
            snippet_max = int(req.get("snippet_max_chars") or 400)
            task_list, metadata, raw_output = agent.refine_meeting_tasks(
                draft,
                meeting_text,
                model_name=model_name,
                out_name=req.get("out_name"),
                allow_fallback=allow_fallback,
                session=session,
                metadata_extra=metadata_extra,
                snippet_max_chars=snippet_max,
            )
        finally:
            _restore_stdout(saved_stdout)
        return {
            "id": req_id,
            "ok": True,
            "task_list": task_list,
            "metadata": metadata,
            "raw_output": raw_output,
        }

    if cmd == "extract":
        saved_stdout = _redirect_stdout_to_stderr()
        try:
            task_list, metadata, raw_output = _run_extract_pipeline(req, agent, session=session)
        finally:
            _restore_stdout(saved_stdout)
        return {
            "id": req_id,
            "ok": True,
            "task_list": task_list,
            "metadata": metadata,
            "raw_output": raw_output,
        }

    if cmd == "shutdown":
        saved_stdout = _redirect_stdout_to_stderr()
        try:
            session.release()
        finally:
            _restore_stdout(saved_stdout)
        return {"id": req_id, "ok": True, "shutdown": True}

    return {"id": req_id, "ok": False, "error": f"unknown cmd: {cmd}"}


def daemon_main() -> int:
    agent = _configure_agent_module()
    session = agent.LlmSession()
    default_model = load_config().get("llm_model", "qwen")

    saved_stdout = _redirect_stdout_to_stderr()
    try:
        warmup_meta = session.load_model(default_model)
    except Exception as exc:
        _restore_stdout(saved_stdout)
        _emit_response({"event": "error", "error": str(exc)})
        return 1
    finally:
        _restore_stdout(saved_stdout)

    _emit_response(
        {
            "event": "ready",
            "ok": True,
            "model": default_model,
            "metadata": warmup_meta,
        }
    )

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except json.JSONDecodeError as exc:
            _emit_response({"ok": False, "error": f"invalid json: {exc}"})
            continue

        try:
            resp = _handle_daemon_request(req, agent, session)
        except Exception as exc:
            resp = {"id": req.get("id"), "ok": False, "error": str(exc)}

        _emit_response(resp)
        if resp.get("shutdown"):
            break

    saved_stdout = _redirect_stdout_to_stderr()
    try:
        session.release()
    finally:
        _restore_stdout(saved_stdout)
    return 0


def once_main() -> int:
    raw = sys.stdin.read()
    if not raw.strip():
        _emit_response({"ok": False, "error": "empty request"})
        return 1

    req = json.loads(raw)
    agent = _configure_agent_module()
    try:
        payload = _run_once(req, agent)
    except Exception as exc:
        _emit_response({"ok": False, "error": str(exc)})
        return 1

    _emit_response(payload)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Meeting Agent LLM worker")
    parser.add_argument("--daemon", action="store_true", help="常驻模式，复用已加载模型")
    args = parser.parse_args()
    if args.daemon:
        return daemon_main()
    return once_main()


if __name__ == "__main__":
    raise SystemExit(main())
