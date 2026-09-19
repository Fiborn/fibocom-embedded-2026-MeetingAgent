# -*- coding: utf-8 -*-
"""
Meeting Agent 动态 Web 前端（Flask + SSE）。

复用 desktop.controller.DesktopController，在浏览器中实时推送转写。
启动：
  cd /home/fibo/MeetingAgent/UI && bash web/run.sh
"""
from __future__ import annotations

import json
import queue
import sys
import threading
import time
from pathlib import Path

from flask import Flask, Response, jsonify, render_template, request, send_from_directory

UI_DIR = Path(__file__).resolve().parent.parent
WEB_DIR = Path(__file__).resolve().parent
if str(UI_DIR) not in sys.path:
    sys.path.insert(0, str(UI_DIR))

from desktop.controller import DesktopController  # noqa: E402

app = Flask(
    __name__,
    template_folder=str(WEB_DIR / "templates"),
    static_folder=str(WEB_DIR / "static"),
    static_url_path="/static",
)
app.config["TEMPLATES_AUTO_RELOAD"] = True
app.jinja_env.auto_reload = True

_clients: list[queue.Queue] = []
_clients_lock = threading.Lock()
_state = {
    "phase": "booting",
    "text": "正在初始化…",
    "recording": False,
    "paused": False,
    "stem": None,
    "lines": [],
    "summary": [],
    "agenda": [],
    "ready": False,
    "error": None,
}
_controller: DesktopController | None = None


def _broadcast(kind: str, payload):
    event = {"kind": kind, "payload": payload, "ts": time.time()}
    with _clients_lock:
        dead = []
        for q in _clients:
            try:
                q.put_nowait(event)
            except queue.Full:
                dead.append(q)
        for q in dead:
            try:
                _clients.remove(q)
            except ValueError:
                pass


def _on_controller_event(kind: str, payload):
    if kind == "status":
        _state["phase"] = payload.get("phase", "idle")
        _state["text"] = payload.get("text", "")
        if _controller is not None:
            _state["recording"] = bool(_controller.recording)
            _state["paused"] = bool(_controller.paused)
            _state["stem"] = _controller.stem
    elif kind == "ready":
        _state["ready"] = True
        _state["phase"] = "idle"
        _state["text"] = payload.get("text", "就绪")
        _state["error"] = None
    elif kind == "transcript":
        line = str(payload)
        _state["lines"].append(line)
        if len(_state["lines"]) > 2000:
            _state["lines"] = _state["lines"][-1500:]
    elif kind == "transcript_clear":
        _state["lines"] = []
    elif kind == "summary":
        _state["summary"] = payload if isinstance(payload, list) else [str(payload)]
    elif kind == "agenda":
        _state["agenda"] = payload if isinstance(payload, list) else [str(payload)]
    elif kind == "error":
        _state["error"] = str(payload)
        _state["text"] = str(payload)
    _broadcast(kind, payload)


def get_controller() -> DesktopController:
    global _controller
    if _controller is None:
        _controller = DesktopController(on_event=_on_controller_event)
        _controller.bootstrap()
    return _controller


@app.route("/")
def index():
    get_controller()
    return render_template("index.html")


@app.route("/api/state")
def api_state():
    c = get_controller()
    return jsonify(
        {
            "phase": _state["phase"],
            "text": _state["text"],
            "recording": bool(c.recording),
            "paused": bool(c.paused),
            "stem": c.stem,
            "ready": _state["ready"],
            "error": _state["error"],
            "lines": _state["lines"][-200:],
            "summary": _state["summary"],
            "agenda": _state["agenda"],
        }
    )


@app.route("/api/events")
def api_events():
    get_controller()
    q: queue.Queue = queue.Queue(maxsize=256)
    with _clients_lock:
        _clients.append(q)

    def gen():
        # 首包：完整快照，便于刷新后恢复
        snap = {
            "kind": "snapshot",
            "payload": {
                "phase": _state["phase"],
                "text": _state["text"],
                "recording": _state["recording"],
                "paused": _state["paused"],
                "stem": _state["stem"],
                "ready": _state["ready"],
                "error": _state["error"],
                "lines": _state["lines"][-200:],
                "summary": _state["summary"],
                "agenda": _state["agenda"],
            },
            "ts": time.time(),
        }
        yield "data: {}\n\n".format(json.dumps(snap, ensure_ascii=False))
        try:
            while True:
                try:
                    event = q.get(timeout=15)
                    yield "data: {}\n\n".format(json.dumps(event, ensure_ascii=False))
                except queue.Empty:
                    yield "data: {}\n\n".format(
                        json.dumps({"kind": "ping", "payload": None, "ts": time.time()})
                    )
        finally:
            with _clients_lock:
                try:
                    _clients.remove(q)
                except ValueError:
                    pass

    return Response(
        gen(),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


@app.route("/api/action/<name>", methods=["POST"])
def api_action(name: str):
    c = get_controller()
    data = request.get_json(silent=True) or {}
    try:
        if name == "start":
            ok = c.start()
            return jsonify({"ok": ok})
        if name == "pause":
            return jsonify({"ok": c.pause()})
        if name == "resume":
            return jsonify({"ok": c.resume()})
        if name == "end":
            return jsonify({"ok": c.end()})
        if name == "quit":
            return jsonify({"ok": c.quit_discard()})
        if name == "summary":
            c.request_summary()
            return jsonify({"ok": True})
        if name == "agenda":
            c.request_agenda()
            return jsonify({"ok": True})
        if name == "history":
            return jsonify({"ok": True, "items": c.list_history()})
        if name == "history_load":
            stem = str(data.get("stem") or "")
            return jsonify({"ok": True, "lines": c.load_history(stem)})
        if name == "enroll_list":
            return jsonify({"ok": True, "items": c.list_enrolled()})
        if name == "enroll_delete":
            nm = str(data.get("name") or "").strip()
            if not nm:
                return jsonify({"ok": False, "error": "缺少姓名"}), 400
            return jsonify({"ok": c.delete_enrolled(nm)})
        if name == "enroll":
            nm = str(data.get("name") or "").strip()
            sec = float(data.get("duration_sec") or 15)
            if not nm:
                return jsonify({"ok": False, "error": "请填写姓名"}), 400
            info = c.enroll_speaker(nm, duration_sec=sec)
            return jsonify({"ok": True, "profile": info})
        return jsonify({"ok": False, "error": "unknown action"}), 400
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc)}), 500


@app.route("/favicon.ico")
def favicon():
    return send_from_directory(app.static_folder, "favicon.svg", mimetype="image/svg+xml")


def main():
    host = "0.0.0.0"
    port = int((__import__("os").environ.get("MEETING_WEB_PORT") or "8787"))
    print("[web] Meeting Agent UI  http://127.0.0.1:{}".format(port))
    print("[web] 请用板端浏览器打开；勿在 Cursor 沙箱中抢麦")
    get_controller()
    # threaded=True：SSE 与 action 并行
    app.run(host=host, port=port, threaded=True, use_reloader=False)


if __name__ == "__main__":
    main()
