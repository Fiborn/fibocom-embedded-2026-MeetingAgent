# -*- coding: utf-8 -*-
"""
Meeting Agent 桌面前端 —— 在板端 DSI/HDMI 屏上全屏运行。

集成：实时转写（说话人）/ 暂停恢复 / 会议总结 / 日程 / 历史回顾。
启动：
  cd /home/fibo/MeetingAgent/UI
  DISPLAY=:0 python3 -m desktop.app
或：
  bash desktop/run.sh
"""
from __future__ import annotations

import sys
import tkinter as tk
from pathlib import Path
from tkinter import font as tkfont
from tkinter import messagebox

# 保证以 UI/ 为工作根，便于复用 paths / meeting_ai
UI_DIR = Path(__file__).resolve().parent.parent
if str(UI_DIR) not in sys.path:
    sys.path.insert(0, str(UI_DIR))

from desktop import theme as T  # noqa: E402
from desktop.controller import DesktopController  # noqa: E402


class MeetingDesktopApp(tk.Tk):
    PAGES = (
        ("live", "实时会议"),
        ("summary", "会议总结"),
        ("agenda", "会议日程"),
        ("history", "历史记录"),
    )

    def __init__(self):
        super().__init__()
        self.title("Meeting Agent")
        self.configure(bg=T.BG)
        self.attributes("-fullscreen", True)
        self.bind("<Escape>", lambda _e: self._toggle_fullscreen())
        self.bind("<F11>", lambda _e: self._toggle_fullscreen())

        self._page = tk.StringVar(value="live")
        self._status_phase = "booting"
        self._history_items: list[dict] = []
        self._nav_btns: dict[str, tk.Button] = {}

        self._build_fonts()
        self.controller = DesktopController(on_event=self._on_controller_event)
        self._build_layout()
        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self.after(200, self.controller.bootstrap)
        self.after(80, self._pulse_rec_dot)

    def _build_fonts(self):
        # 若系统缺字，回退到默认
        for name in (T.FONT_FAMILY, T.FONT_MONO):
            try:
                tkfont.Font(family=name, size=12)
            except tk.TclError:
                pass

    def _toggle_fullscreen(self):
        self.attributes("-fullscreen", not self.attributes("-fullscreen"))

    def _build_layout(self):
        # 先固定顶栏 / 底栏，再填中间，避免按钮被挤出屏幕
        top = tk.Frame(self, bg=T.BG_RAIL, height=88)
        top.pack(fill="x", side="top")
        top.pack_propagate(False)

        brand = tk.Frame(top, bg=T.BG_RAIL)
        brand.pack(side="left", padx=28, pady=12)
        tk.Label(
            brand, text="Meeting Agent", font=T.FONT_BRAND,
            fg=T.FG, bg=T.BG_RAIL,
        ).pack(anchor="w")
        tk.Label(
            brand, text="SC171 实时会议助手", font=T.FONT_STATUS,
            fg=T.FG_MUTED, bg=T.BG_RAIL,
        ).pack(anchor="w")

        # 顶栏右侧：启动 / 关闭（始终可见）
        top_actions = tk.Frame(top, bg=T.BG_RAIL)
        top_actions.pack(side="right", padx=20, pady=14)
        self.btn_close = self._action_btn(
            top_actions, "关闭", T.DANGER, self._on_close, fg=T.FG, width=10,
        )
        self.btn_start_top = self._action_btn(
            top_actions, "启动", T.ACCENT, self._on_start, fg=T.BG, width=10,
        )
        # pack 顺序：先 close 再 start，因 side=right 会反向，启动在左、关闭在右
        self.btn_close.pack(side="right", padx=6)
        self.btn_start_top.pack(side="right", padx=6)

        status_box = tk.Frame(top, bg=T.BG_RAIL)
        status_box.pack(side="right", padx=16)
        self.rec_dot = tk.Canvas(
            status_box, width=18, height=18, bg=T.BG_RAIL,
            highlightthickness=0,
        )
        self.rec_dot.pack(side="left", padx=(0, 10))
        self._rec_dot_id = self.rec_dot.create_oval(3, 3, 15, 15, fill=T.FG_DIM, outline="")
        self.status_var = tk.StringVar(value="正在初始化…")
        tk.Label(
            status_box, textvariable=self.status_var, font=T.FONT_BODY,
            fg=T.FG, bg=T.BG_RAIL,
        ).pack(side="left")

        # 底栏控制（必须在 body 之前 pack）
        ctrl = tk.Frame(self, bg=T.BG_PANEL, height=112)
        ctrl.pack(fill="x", side="bottom")
        ctrl.pack_propagate(False)
        inner = tk.Frame(ctrl, bg=T.BG_PANEL)
        inner.pack(expand=True, fill="both", pady=18)

        self.btn_start = self._ctrl_btn(inner, "启动", T.ACCENT, self._on_start)
        self.btn_pause = self._ctrl_btn(inner, "暂停", T.WARN, self._on_pause)
        self.btn_resume = self._ctrl_btn(inner, "继续", T.OK, self._on_resume)
        self.btn_end = self._ctrl_btn(inner, "结束", "#3A4A52", self._on_end, fg=T.FG)
        self.btn_quit = self._ctrl_btn(inner, "放弃", T.DANGER, self._on_quit, fg=T.FG)
        self.btn_sum = self._ctrl_btn(inner, "生成总结", T.BG_CARD, self._on_summary, fg=T.FG)
        self.btn_agenda = self._ctrl_btn(inner, "生成日程", T.BG_CARD, self._on_agenda, fg=T.FG)
        self.btn_close_bottom = self._ctrl_btn(inner, "关闭", T.DANGER, self._on_close, fg=T.FG)

        # 主体：左导航 + 内容
        body = tk.Frame(self, bg=T.BG)
        body.pack(fill="both", expand=True)

        rail = tk.Frame(body, bg=T.BG_RAIL, width=T.NAV_W)
        rail.pack(side="left", fill="y")
        rail.pack_propagate(False)

        tk.Label(
            rail, text="功能", font=T.FONT_STATUS,
            fg=T.FG_DIM, bg=T.BG_RAIL, anchor="w",
        ).pack(fill="x", padx=24, pady=(28, 8))

        for key, label in self.PAGES:
            btn = tk.Button(
                rail, text=label, font=T.FONT_BTN,
                fg=T.FG, bg=T.BG_RAIL, activebackground=T.BG_CARD,
                activeforeground=T.ACCENT, bd=0, relief="flat",
                anchor="w", padx=24, pady=18,
                command=lambda k=key: self._show_page(k),
                cursor="hand2",
            )
            btn.pack(fill="x", padx=8, pady=4)
            self._nav_btns[key] = btn

        tk.Frame(rail, bg=T.BG_RAIL).pack(fill="both", expand=True)
        tk.Button(
            rail, text="退出全屏 Esc", font=T.FONT_STATUS,
            fg=T.FG_MUTED, bg=T.BG_CARD, bd=0, relief="flat",
            command=self._toggle_fullscreen, cursor="hand2", pady=12,
        ).pack(fill="x", padx=16, pady=8)

        content = tk.Frame(body, bg=T.BG)
        content.pack(side="left", fill="both", expand=True)

        self.pages: dict[str, tk.Frame] = {}
        self.pages["live"] = self._build_live_page(content)
        self.pages["summary"] = self._build_text_page(content, "会议总结", "summary")
        self.pages["agenda"] = self._build_text_page(content, "会议日程", "agenda")
        self.pages["history"] = self._build_history_page(content)

        self._show_page("live")
        self._refresh_ctrl_state()

    def _action_btn(self, parent, text, bg, cmd, fg=T.BG, width=8):
        return tk.Button(
            parent, text=text, font=T.FONT_BTN,
            fg=fg, bg=bg, activebackground=bg,
            activeforeground=fg, bd=0, relief="flat",
            padx=22, pady=12, width=width,
            cursor="hand2", command=cmd,
            highlightthickness=1, highlightbackground=T.BORDER,
        )

    def _ctrl_btn(self, parent, text, bg, cmd, fg=T.BG):
        btn = tk.Button(
            parent, text=text, font=T.FONT_BTN,
            fg=fg, bg=bg, activebackground=bg,
            activeforeground=fg, bd=0, relief="flat",
            padx=28, pady=16, cursor="hand2", command=cmd,
            highlightthickness=1, highlightbackground=T.BORDER,
        )
        btn.pack(side="left", padx=8)
        return btn

    def _build_live_page(self, parent) -> tk.Frame:
        page = tk.Frame(parent, bg=T.BG)
        head = tk.Frame(page, bg=T.BG)
        head.pack(fill="x", padx=32, pady=(28, 8))
        tk.Label(
            head, text="实时转写", font=T.FONT_H1,
            fg=T.FG, bg=T.BG,
        ).pack(side="left")
        tk.Label(
            head, text="说话人标签会随语音自动更新", font=T.FONT_STATUS,
            fg=T.FG_MUTED, bg=T.BG,
        ).pack(side="left", padx=16)

        card = tk.Frame(page, bg=T.BG_CARD, highlightbackground=T.BORDER, highlightthickness=1)
        card.pack(fill="both", expand=True, padx=32, pady=(0, 20))

        self.transcript = tk.Text(
            card, wrap="word", font=T.FONT_TRANSCRIPT,
            bg=T.BG_CARD, fg=T.FG, insertbackground=T.ACCENT,
            relief="flat", bd=0, padx=24, pady=20,
            state="disabled",
        )
        scroll = tk.Scrollbar(card, command=self.transcript.yview, bg=T.BG_PANEL)
        self.transcript.configure(yscrollcommand=scroll.set)
        scroll.pack(side="right", fill="y")
        self.transcript.pack(side="left", fill="both", expand=True)

        self.transcript.tag_configure("speaker", foreground=T.ACCENT)
        self.transcript.tag_configure("muted", foreground=T.FG_MUTED)
        return page

    def _build_text_page(self, parent, title: str, key: str) -> tk.Frame:
        page = tk.Frame(parent, bg=T.BG)
        head = tk.Frame(page, bg=T.BG)
        head.pack(fill="x", padx=32, pady=(28, 8))
        tk.Label(head, text=title, font=T.FONT_H1, fg=T.FG, bg=T.BG).pack(side="left")
        hint = "基于本场转写生成，可在录音中或结束后点击底栏按钮"
        tk.Label(head, text=hint, font=T.FONT_STATUS, fg=T.FG_MUTED, bg=T.BG).pack(side="left", padx=16)

        card = tk.Frame(page, bg=T.BG_CARD, highlightbackground=T.BORDER, highlightthickness=1)
        card.pack(fill="both", expand=True, padx=32, pady=(0, 20))
        text = tk.Text(
            card, wrap="word", font=T.FONT_BODY_LG,
            bg=T.BG_CARD, fg=T.FG, relief="flat", bd=0,
            padx=24, pady=20, state="disabled",
        )
        scroll = tk.Scrollbar(card, command=text.yview)
        text.configure(yscrollcommand=scroll.set)
        scroll.pack(side="right", fill="y")
        text.pack(side="left", fill="both", expand=True)
        setattr(self, f"{key}_text", text)
        return page

    def _build_history_page(self, parent) -> tk.Frame:
        page = tk.Frame(parent, bg=T.BG)
        head = tk.Frame(page, bg=T.BG)
        head.pack(fill="x", padx=32, pady=(28, 8))
        tk.Label(head, text="历史记录", font=T.FONT_H1, fg=T.FG, bg=T.BG).pack(side="left")
        tk.Button(
            head, text="刷新", font=T.FONT_STATUS,
            fg=T.ACCENT, bg=T.BG, bd=0, command=self._refresh_history, cursor="hand2",
        ).pack(side="left", padx=16)

        paned = tk.PanedWindow(page, orient="horizontal", bg=T.BG, sashwidth=6, bd=0)
        paned.pack(fill="both", expand=True, padx=32, pady=(0, 20))

        left = tk.Frame(paned, bg=T.BG_CARD, width=360)
        right = tk.Frame(paned, bg=T.BG_CARD)
        paned.add(left, minsize=280)
        paned.add(right, minsize=480)

        self.history_list = tk.Listbox(
            left, font=T.FONT_BODY, bg=T.BG_CARD, fg=T.FG,
            selectbackground=T.ACCENT_DIM, selectforeground=T.FG,
            activestyle="none", bd=0, highlightthickness=0,
        )
        self.history_list.pack(fill="both", expand=True, padx=8, pady=8)
        self.history_list.bind("<<ListboxSelect>>", self._on_history_select)

        self.history_view = tk.Text(
            right, wrap="word", font=T.FONT_BODY_LG,
            bg=T.BG_CARD, fg=T.FG, relief="flat", bd=0,
            padx=20, pady=16, state="disabled",
        )
        self.history_view.pack(fill="both", expand=True)
        return page

    def _show_page(self, key: str):
        self._page.set(key)
        for k, frame in self.pages.items():
            if k == key:
                frame.pack(fill="both", expand=True)
            else:
                frame.pack_forget()
        for k, btn in self._nav_btns.items():
            if k == key:
                btn.configure(bg=T.BG_CARD, fg=T.ACCENT)
            else:
                btn.configure(bg=T.BG_RAIL, fg=T.FG)
        if key == "history":
            self._refresh_history()

    def _set_text_widget(self, widget: tk.Text, lines):
        widget.configure(state="normal")
        widget.delete("1.0", "end")
        if isinstance(lines, str):
            widget.insert("end", lines)
        else:
            widget.insert("end", "\n".join(lines))
        widget.configure(state="disabled")
        widget.see("end")

    def _append_transcript(self, line: str):
        self.transcript.configure(state="normal")
        if ": " in line:
            spk, rest = line.split(": ", 1)
            self.transcript.insert("end", spk + ": ", ("speaker",))
            self.transcript.insert("end", rest + "\n")
        else:
            self.transcript.insert("end", line + "\n")
        self.transcript.configure(state="disabled")
        self.transcript.see("end")

    def _clear_transcript(self):
        self.transcript.configure(state="normal")
        self.transcript.delete("1.0", "end")
        self.transcript.configure(state="disabled")

    # ----- 控制 -----
    def _on_start(self):
        self._append_system("正在启动实时转写…（请对着 USB 麦克风说话）")
        ok = self.controller.start()
        if ok:
            self._append_system("已开录。若长时间无文字，请检查麦克风是否被占用或从板端终端重启 UI。")
        self._refresh_ctrl_state()

    def _on_pause(self):
        self.controller.pause()
        self._refresh_ctrl_state()

    def _on_resume(self):
        self.controller.resume()
        self._refresh_ctrl_state()

    def _on_end(self):
        self.controller.end()
        self._refresh_ctrl_state()

    def _on_quit(self):
        if messagebox.askyesno("放弃本场", "将删除本场录音与转写，确认？"):
            self.controller.quit_discard()
            self._clear_transcript()
            self._refresh_ctrl_state()

    def _on_summary(self):
        self._show_page("summary")
        self.controller.request_summary()

    def _on_agenda(self):
        self._show_page("agenda")
        self.controller.request_agenda()

    def _refresh_history(self):
        self._history_items = self.controller.list_history()
        self.history_list.delete(0, "end")
        for item in self._history_items:
            self.history_list.insert("end", f"{item['mtime']}  {item['stem']}")

    def _on_history_select(self, _evt=None):
        sel = self.history_list.curselection()
        if not sel:
            return
        item = self._history_items[sel[0]]
        lines = self.controller.load_history(item["stem"])
        self._set_text_widget(self.history_view, lines)

    def _refresh_ctrl_state(self):
        rec = self.controller.recording
        paused = self.controller.paused
        # 简单启用逻辑
        states = {
            self.btn_start: not rec,
            self.btn_start_top: not rec,
            self.btn_pause: rec and not paused,
            self.btn_resume: rec and paused,
            self.btn_end: rec,
            self.btn_quit: rec or bool(self.controller.stem),
            self.btn_sum: True,
            self.btn_agenda: True,
            self.btn_close: True,
            self.btn_close_bottom: True,
        }
        for btn, enabled in states.items():
            btn.configure(state=("normal" if enabled else "disabled"))

    def _pulse_rec_dot(self):
        phase = self._status_phase
        if phase == "recording":
            # 闪烁
            cur = self.rec_dot.itemcget(self._rec_dot_id, "fill")
            nxt = T.REC_GLOW if cur != T.REC_GLOW else T.BG_RAIL
            self.rec_dot.itemconfigure(self._rec_dot_id, fill=nxt)
        elif phase == "paused":
            self.rec_dot.itemconfigure(self._rec_dot_id, fill=T.WARN)
        elif phase == "agent":
            self.rec_dot.itemconfigure(self._rec_dot_id, fill=T.ACCENT)
        elif phase == "ready" or phase == "idle":
            self.rec_dot.itemconfigure(self._rec_dot_id, fill=T.OK)
        else:
            self.rec_dot.itemconfigure(self._rec_dot_id, fill=T.FG_DIM)
        self.after(500, self._pulse_rec_dot)

    # ----- 事件（可能来自后台线程） -----
    def _on_controller_event(self, kind: str, payload):
        self.after(0, lambda: self._handle_event(kind, payload))

    def _handle_event(self, kind: str, payload):
        if kind == "status":
            self._status_phase = payload.get("phase", "idle")
            self.status_var.set(payload.get("text", ""))
            self._refresh_ctrl_state()
        elif kind == "ready":
            self._status_phase = "idle"
            self.status_var.set(payload.get("text", "就绪"))
            self._append_system("系统就绪，点击「开始」进入会议")
        elif kind == "transcript":
            self._append_transcript(str(payload))
        elif kind == "transcript_clear":
            self._clear_transcript()
        elif kind == "summary":
            self._set_text_widget(self.summary_text, payload)
            self._show_page("summary")
        elif kind == "agenda":
            self._set_text_widget(self.agenda_text, payload)
            self._show_page("agenda")
        elif kind == "error":
            self.status_var.set(str(payload))
            self._append_system(f"错误：{payload}")
            messagebox.showerror("Meeting Agent", str(payload))
        self._refresh_ctrl_state()

    def _append_system(self, text: str):
        self.transcript.configure(state="normal")
        self.transcript.insert("end", text + "\n", ("muted",))
        self.transcript.configure(state="disabled")
        self.transcript.see("end")

    def _on_close(self):
        try:
            self.controller.shutdown()
        finally:
            self.destroy()


def main():
    app = MeetingDesktopApp()
    app.mainloop()


if __name__ == "__main__":
    main()
