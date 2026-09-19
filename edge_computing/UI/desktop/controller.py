# -*- coding: utf-8 -*-
"""桌面 UI 业务控制器：复用 MeetingSession / agent / harness，不依赖串口屏。"""
from __future__ import annotations

import os
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional

from agent_bridge import (
    extract_rules_draft,
    format_agenda_lines,
    format_summary_lines,
    read_transcript_file,
    run_agent_llm_refine,
    should_run_llm_refine,
    shutdown_llm_worker,
    warmup_llm_worker,
)
from harness_bridge import augment_for_llm_refine, reset_writeback_schedule, schedule_memory_writeback
from meeting_ai import MeetingSession
from paths import apply_runtime_paths, setup_workdir
from scroll_text import extract_display_lines


class DesktopController:
    def __init__(self, on_event: Callable[[str, object], None]):
        """
        on_event(kind, payload) 在后台线程也可能被调用，UI 需自行 marshal 到主线程。
        kind: status | transcript | summary | agenda | history | error | ready
        """
        self._on_event = on_event
        self._runtime = apply_runtime_paths()
        setup_workdir()
        self.ui_dir = Path(self._runtime["ui_dir"])
        # 场次数据统一放在 <工程>/data/meetings/：{stem}.wav 与 {stem}.txt 同名配对
        self.meetings_dir = Path(self._runtime["meetings_dir"])
        self.record_dir = self.meetings_dir
        self.audio_dir = self.meetings_dir
        self.agent_out = Path(self._runtime["agent_out_dir"])
        self.meetings_dir.mkdir(parents=True, exist_ok=True)

        self.recording = False
        self.paused = False
        self.stem: Optional[str] = None
        self.txt_path: Optional[Path] = None
        self.wav_path: Optional[Path] = None
        self.lines: list[str] = []
        self._agent_busy = False
        self._agent_draft = None
        self._agent_final = None
        self._agent_stem = None
        self._lock = threading.Lock()

        self.session = MeetingSession(on_utterance=self._on_utterance)
        self._emit("status", {"phase": "booting", "text": "正在初始化…"})

    def _emit(self, kind: str, payload):
        try:
            self._on_event(kind, payload)
        except Exception as exc:
            print("[DesktopController] event error:", exc)

    def _probe_mic(self) -> str | None:
        """快速探测采集设备；失败返回错误文案。"""
        import subprocess

        try:
            out = subprocess.check_output(
                ["arecord", "-l"], text=True, stderr=subprocess.STDOUT, timeout=5
            )
        except Exception as exc:
            return "无法枚举声卡: {}".format(exc)
        if "card " not in out:
            return "未检测到声卡（/dev/snd 不可见？请在板端本机终端启动 UI，勿在沙箱中运行）"
        return None

    def bootstrap(self):
        """后台预热 ASR + 可选 LLM。"""

        def work():
            try:
                mic_err = self._probe_mic()
                if mic_err:
                    self._emit("error", mic_err)
                    # 仍继续预热模型，但提醒用户无麦则无 ASR
                    self._emit("status", {"phase": "booting", "text": "麦克风异常，仍尝试加载模型…"})
                cfg = __import__("paths", fromlist=["load_config"]).load_config()
                if (cfg.get("llm_worker") or {}).get("warmup_on_start", True):
                    warmup_llm_worker(background=True)
                self.session.warmup()
                # 等待就绪或失败
                deadline = time.time() + 180
                while time.time() < deadline:
                    if self.session._warmup_failed.is_set():
                        self._emit(
                            "error",
                            self.session._warmup_error or "预热失败",
                        )
                        return
                    if self.session._ready.is_set():
                        if mic_err:
                            self._emit("ready", {"text": "模型就绪，但麦克风不可用"})
                            self._emit("status", {"phase": "idle", "text": "无麦克风 · 无法实时 ASR"})
                        else:
                            self._emit("ready", {"text": "系统就绪，可开始会议"})
                            self._emit("status", {"phase": "idle", "text": "待命"})
                        return
                    time.sleep(0.3)
                self._emit("error", "预热超时，请检查麦克风与模型")
            except Exception as exc:
                self._emit("error", str(exc))

        threading.Thread(target=work, daemon=True, name="desktop-bootstrap").start()

    def _on_utterance(self, line: str):
        with self._lock:
            if not self.recording or self.paused:
                return
            self.lines.append(line)
            if self.txt_path:
                try:
                    with open(self.txt_path, "a", encoding="utf-8") as f:
                        f.write(line + "\n")
                except OSError as exc:
                    print("[record] write fail:", exc)
        self._emit("transcript", line)

    def start(self) -> bool:
        with self._lock:
            if self.recording:
                return False
            stem = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.stem = stem
            self.txt_path = self.record_dir / f"{stem}.txt"
            self.wav_path = self.audio_dir / f"{stem}.wav"
            self.lines = []
            self._agent_draft = None
            self._agent_final = None
            self._agent_stem = stem
            reset_writeback_schedule()
            header = (
                f"录音开始: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
                f"{'=' * 40}\n"
            )
            self.txt_path.write_text(header, encoding="utf-8")
        ok = self.session.start()
        if not ok:
            self._emit("error", "无法启动转写（预热未完成或麦克风异常）")
            with self._lock:
                self.recording = False
            return False
        with self._lock:
            self.recording = True
            self.paused = False
        self._emit("status", {"phase": "recording", "text": f"录音中 · {self.stem}"})
        return True

    def pause(self) -> bool:
        with self._lock:
            if not self.recording or self.paused:
                return False
            self.paused = True
        self.session.pause()
        self._emit("status", {"phase": "paused", "text": "已暂停"})
        return True

    def resume(self) -> bool:
        with self._lock:
            if not self.recording or not self.paused:
                return False
            self.paused = False
        self.session.resume()
        self._emit("status", {"phase": "recording", "text": f"录音中 · {self.stem}"})
        return True

    def end(self) -> bool:
        with self._lock:
            if not self.recording and self.stem is None:
                return False
            stem = self.stem
            wav = self.wav_path
            txt = self.txt_path
            self.recording = False
            self.paused = False
        try:
            self.session.stop(wav_path=str(wav) if wav else None)
        except Exception as exc:
            print("[AI] stop:", exc)
        if txt and txt.exists():
            with open(txt, "a", encoding="utf-8") as f:
                f.write(f"\n录音结束: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        self._schedule_memory(stem)
        self._emit("status", {"phase": "idle", "text": f"已结束 · {stem or ''}"})
        return True

    def quit_discard(self) -> bool:
        """放弃本场：停录并删除文件。"""
        with self._lock:
            wav = self.wav_path
            txt = self.txt_path
            self.recording = False
            self.paused = False
            self.stem = None
            self.lines = []
        try:
            self.session.stop(wav_path=None)
        except Exception as exc:
            print("[AI] quit stop:", exc)
        for p in (wav, txt):
            if p and Path(p).exists():
                try:
                    os.remove(p)
                except OSError:
                    pass
        self._emit("status", {"phase": "idle", "text": "已放弃本场"})
        self._emit("transcript_clear", None)
        return True

    def _meeting_text(self) -> str:
        if self.txt_path and self.txt_path.exists():
            text = read_transcript_file(self.txt_path)
            if text.strip():
                return text
        return "\n".join(self.lines)

    def _schedule_memory(self, stem: Optional[str]):
        if not stem:
            return

        def work():
            try:
                schedule_memory_writeback(self.txt_path, stem=stem or "")
            except Exception as exc:
                print("[harness]", exc)

        threading.Thread(target=work, daemon=True).start()

    def request_summary(self):
        self._run_agent("summary")

    def request_agenda(self):
        self._run_agent("agenda")

    def _run_agent(self, kind: str):
        if self._agent_busy:
            self._emit("error", "AI 正在处理，请稍候")
            return
        text = self._meeting_text().strip()
        if not text:
            self._emit(kind, ["暂无会议记录，请先开始录音"])
            return
        self._agent_busy = True
        self._emit("status", {"phase": "agent", "text": f"正在生成{'总结' if kind=='summary' else '日程'}…"})

        def work():
            try:
                draft, _meta = extract_rules_draft(text)
                self._agent_draft = draft
                lines = (
                    format_summary_lines(draft, banner="[会议总结·草稿]")
                    if kind == "summary"
                    else format_agenda_lines(draft, banner="[会议日程·草稿]")
                )
                self._emit(kind, lines)
                if should_run_llm_refine(text, draft):
                    llm_text, memories = augment_for_llm_refine(text)
                    if memories:
                        print(f"[harness] memories={len(memories)}")
                    task_list, _m, _raw = run_agent_llm_refine(
                        llm_text, draft, out_name=self.stem or "desktop"
                    )
                    self._agent_final = task_list
                    final_lines = (
                        format_summary_lines(task_list, banner="[会议总结]")
                        if kind == "summary"
                        else format_agenda_lines(task_list, banner="[会议日程]")
                    )
                    self._emit(kind, final_lines)
                    self._insert_into_record(final_lines, kind)
                else:
                    self._agent_final = draft
                    self._insert_into_record(lines, kind)
                self._emit("status", {"phase": "idle" if not self.recording else ("paused" if self.paused else "recording"),
                                      "text": "AI 完成"})
            except Exception as exc:
                self._emit("error", f"AI 失败: {exc}")
            finally:
                self._agent_busy = False

        threading.Thread(target=work, daemon=True, name=f"agent-{kind}").start()

    def _insert_into_record(self, lines: list, kind: str):
        if not self.txt_path or not self.txt_path.exists():
            return
        label = "[会议总结]" if kind == "summary" else "[会议日程]"
        try:
            content = self.txt_path.read_text(encoding="utf-8")
            block = label + "\n" + "\n".join(lines) + "\n\n"
            if label in content:
                # 简单替换旧块：插在文件头分隔线后
                pass
            # 追加到文件末尾前的标签区：这里直接 append
            with open(self.txt_path, "a", encoding="utf-8") as f:
                f.write("\n" + block)
        except OSError as exc:
            print("[record] insert fail:", exc)

    def list_history(self) -> list[dict]:
        items = []
        if not self.record_dir.is_dir():
            return items
        for path in sorted(self.record_dir.glob("*.txt"), key=lambda p: p.stat().st_mtime, reverse=True):
            try:
                text = path.read_text(encoding="utf-8")
            except OSError:
                continue
            preview = ""
            for line in extract_display_lines(text) or text.splitlines():
                line = line.strip()
                if line and not line.startswith("录音") and not line.startswith("="):
                    preview = line[:80]
                    break
            items.append({
                "stem": path.stem,
                "path": str(path),
                "mtime": datetime.fromtimestamp(path.stat().st_mtime).strftime("%Y-%m-%d %H:%M"),
                "preview": preview,
            })
        return items

    def load_history(self, stem: str) -> list[str]:
        path = self.record_dir / f"{stem}.txt"
        if not path.exists():
            return ["记录不存在"]
        try:
            content = path.read_text(encoding="utf-8")
        except OSError as exc:
            return [f"读取失败: {exc}"]
        lines = extract_display_lines(content)
        if not lines:
            body = read_transcript_file(path).strip()
            lines = body.splitlines() if body else ["（空记录）"]
        # 绑定当前会话上下文，便于随后生成总结/日程
        if not self.recording:
            self.stem = stem
            self.txt_path = path
            self.lines = list(lines)
        return lines

    def list_enrolled(self) -> list[dict]:
        return self.session.list_enrolled()

    def delete_enrolled(self, name: str) -> bool:
        if self.recording:
            raise RuntimeError("会议进行中，请先结束后再删除声纹")
        return self.session.delete_enrolled(name)

    def enroll_speaker(self, name: str, duration_sec: float = 15.0) -> dict:
        """前端注册：录音指定秒数并保存声纹。"""
        if self.recording:
            raise RuntimeError("会议进行中，请先结束会议再注册")
        self._emit(
            "status",
            {
                "phase": "agent",
                "text": "声纹注册中 · 请持续说话 {} 秒…".format(int(duration_sec)),
            },
        )
        try:
            info = self.session.enroll_speaker(name, duration_sec=duration_sec)
            self._emit(
                "status",
                {"phase": "idle", "text": "已注册 · {}".format(info.get("name"))},
            )
            self._emit("enroll", {"action": "saved", "profile": info})
            return info
        except Exception as exc:
            self._emit("status", {"phase": "idle", "text": "注册失败"})
            self._emit("error", "声纹注册失败: {}".format(exc))
            raise

    def shutdown(self):
        try:
            if self.recording:
                self.end()
        except Exception:
            pass
        try:
            self.session.shutdown()
        except Exception:
            pass
        try:
            shutdown_llm_worker()
        except Exception:
            pass
