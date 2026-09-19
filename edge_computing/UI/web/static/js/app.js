(() => {
  const $ = (id) => document.getElementById(id);

  const state = {
    phase: "booting",
    text: "正在初始化…",
    recording: false,
    paused: false,
    ready: false,
    lines: [],
  };

  const els = {
    statusPill: $("statusPill"),
    statusText: $("statusText"),
    transcript: $("transcript"),
    emptyHint: $("emptyHint"),
    summaryDoc: $("summaryDoc"),
    agendaDoc: $("agendaDoc"),
    historyList: $("historyList"),
    historyView: $("historyView"),
    liveMeter: $("liveMeter"),
    toast: $("toast"),
    teleClock: $("teleClock"),
    teleLink: $("teleLink"),
    btnStart: $("btnStart"),
    btnStartTop: $("btnStartTop"),
    btnPause: $("btnPause"),
    btnResume: $("btnResume"),
    btnEnd: $("btnEnd"),
    btnQuit: $("btnQuit"),
    btnSummary: $("btnSummary"),
    btnAgenda: $("btnAgenda"),
    btnCloseTop: $("btnCloseTop"),
    btnRefreshHistory: $("btnRefreshHistory"),
    enrollName: $("enrollName"),
    enrollSec: $("enrollSec"),
    btnEnrollStart: $("btnEnrollStart"),
    btnRefreshEnroll: $("btnRefreshEnroll"),
    btnEnrollKeyboard: $("btnEnrollKeyboard"),
    btnEnrollClearName: $("btnEnrollClearName"),
    enrollOsk: $("enrollOsk"),
    oskSurname: $("oskSurname"),
    oskGiven: $("oskGiven"),
    oskEn: $("oskEn"),
    oskNum: $("oskNum"),
    oskBackspace: $("oskBackspace"),
    oskHide: $("oskHide"),
    enrollProgress: $("enrollProgress"),
    enrollBar: $("enrollBar"),
    enrollProgressText: $("enrollProgressText"),
    enrollList: $("enrollList"),
  };

  const OSK_SURNAMES =
    "张李王刘陈杨黄赵吴周徐孙马朱胡郭何高林罗郑梁谢宋唐韩曹许邓萧冯曾程蔡彭潘袁于董余苏叶吕魏蒋田杜丁沈姜范江傅钟卢汪戴崔任陆廖姚方金邱夏谭韦贾邹石熊孟秦阎薛侯雷白龙段郝孔邵史毛常万顾赖武康贺严尹钱施牛洪龚".split(
      ""
    );
  const OSK_GIVEN =
    "伟芳娜敏静丽强磊军洋勇艳杰娟涛明超秀英华慧巧美娜静淑惠珠翠雅芝玉萍红娥玲芬芳燕彩春菊兰凤洁梅琳素云莲真环雪荣爱妹霞香月莺媛艳瑞凡佳嘉怡欣雨婷欣悦思涵诗琪婉清宁安然一诺子轩浩宇晨阳梓涵宇航俊杰文博志强建国国强建华文静晓明晓东志伟".split(
      ""
    );

  function tickClock() {
    if (!els.teleClock) return;
    const d = new Date();
    const p = (n) => String(n).padStart(2, "0");
    els.teleClock.textContent = `${p(d.getHours())}:${p(d.getMinutes())}:${p(d.getSeconds())}`;
  }
  tickClock();
  setInterval(tickClock, 1000);

  function toast(msg) {
    els.toast.hidden = false;
    els.toast.textContent = msg;
    clearTimeout(toast._t);
    toast._t = setTimeout(() => {
      els.toast.hidden = true;
    }, 3200);
  }

  function setStatus(phase, text) {
    if (phase) state.phase = phase;
    if (typeof text === "string") state.text = text;
    els.statusPill.dataset.phase = state.phase;
    els.statusText.textContent = state.text || "";
    const live = state.phase === "recording";
    els.liveMeter.classList.toggle("on", live);
  }

  function refreshButtons() {
    const rec = state.recording;
    const paused = state.paused;
    els.btnStart.disabled = rec;
    els.btnStartTop.disabled = rec;
    els.btnPause.disabled = !(rec && !paused);
    els.btnResume.disabled = !(rec && paused);
    els.btnEnd.disabled = !rec;
    els.btnQuit.disabled = !(rec || state.stem);
  }

  function parseLine(line) {
    const idx = line.indexOf(": ");
    if (idx > 0 && idx < 24) {
      return { spk: line.slice(0, idx), text: line.slice(idx + 2) };
    }
    return null;
  }

  function appendTranscript(line, { system = false } = {}) {
    if (els.emptyHint) els.emptyHint.remove();
    const row = document.createElement("div");
    if (system) {
      row.className = "line system";
      row.textContent = line;
    } else {
      const parsed = parseLine(line);
      row.className = "line";
      if (parsed) {
        const spk = document.createElement("div");
        spk.className = "spk";
        spk.dataset.spk = parsed.spk;
        spk.textContent = parsed.spk;
        const text = document.createElement("div");
        text.className = "text";
        text.textContent = parsed.text;
        row.append(spk, text);
      } else {
        row.className = "line system";
        row.textContent = line;
      }
    }
    els.transcript.appendChild(row);
    els.transcript.scrollTop = els.transcript.scrollHeight;
  }

  function clearTranscript() {
    els.transcript.innerHTML = "";
    const hint = document.createElement("div");
    hint.className = "empty-hint";
    hint.id = "emptyHint";
    hint.innerHTML =
      '<span class="mono">&gt; AWAITING COMMAND</span><br />点击「启动」开始会议';
    els.transcript.appendChild(hint);
    els.emptyHint = hint;
    state.lines = [];
  }

  function setDoc(el, lines) {
    if (Array.isArray(lines)) el.textContent = lines.join("\n");
    else el.textContent = String(lines || "");
  }

  function showPage(key) {
    document.querySelectorAll(".page").forEach((p) => {
      p.classList.toggle("active", p.dataset.page === key);
    });
    document.querySelectorAll(".nav-item").forEach((b) => {
      b.classList.toggle("active", b.dataset.page === key);
    });
    if (key === "history") loadHistory();
    if (key === "enroll") loadEnrollList();
  }

  function fmtPitch(v) {
    if (v == null || Number.isNaN(Number(v))) return "-";
    return `${Math.round(Number(v))}Hz`;
  }

  async function loadEnrollList() {
    if (!els.enrollList) return;
    const data = await action("enroll_list");
    const items = data.items || [];
    els.enrollList.innerHTML = "";
    if (!items.length) {
      const p = document.createElement("p");
      p.className = "enroll-empty";
      p.textContent = "暂无注册，请先录入";
      els.enrollList.appendChild(p);
      return;
    }
    items.forEach((item) => {
      const row = document.createElement("div");
      row.className = "enroll-item";
      const meta = document.createElement("div");
      meta.className = "enroll-item-meta";
      meta.innerHTML =
        `<div class="enroll-item-name">${item.name || ""}</div>` +
        `<div class="enroll-item-sub">pitch ${fmtPitch(item.pitch)} · F1 ${fmtPitch(item.f1)} · ` +
        `${item.segments || 1}段 · ${item.updated_at || item.created_at || ""}</div>`;
      const del = document.createElement("button");
      del.type = "button";
      del.className = "btn btn-danger";
      del.textContent = "删除";
      del.addEventListener("click", async () => {
        if (!confirm(`删除声纹「${item.name}」？`)) return;
        const res = await action("enroll_delete", { name: item.name });
        if (res.ok) {
          toast(`已删除 ${item.name}`);
          loadEnrollList();
        }
      });
      row.append(meta, del);
      els.enrollList.appendChild(row);
    });
  }

  function setEnrollName(value) {
    if (!els.enrollName) return;
    els.enrollName.value = String(value || "").slice(0, 20);
  }

  function appendEnrollName(ch) {
    if (!els.enrollName) return;
    const cur = els.enrollName.value || "";
    if (cur.length >= 20) {
      toast("姓名最多 20 字");
      return;
    }
    setEnrollName(cur + ch);
  }

  function fillOskKeys(container, chars, className) {
    if (!container) return;
    container.innerHTML = "";
    chars.forEach((ch) => {
      const b = document.createElement("button");
      b.type = "button";
      b.className = "osk-key" + (className ? ` ${className}` : "");
      b.textContent = ch;
      b.addEventListener("click", (ev) => {
        ev.preventDefault();
        appendEnrollName(ch);
      });
      container.appendChild(b);
    });
  }

  function initEnrollOsk() {
    if (!els.enrollOsk) return;
    fillOskKeys(els.oskSurname, OSK_SURNAMES);
    fillOskKeys(els.oskGiven, OSK_GIVEN);
    const rows = ["QWERTYUIOP", "ASDFGHJKL", "ZXCVBNM"];
    if (els.oskEn) {
      els.oskEn.innerHTML = "";
      rows.forEach((row) => {
        row.split("").forEach((ch) => {
          const b = document.createElement("button");
          b.type = "button";
          b.className = "osk-key";
          b.textContent = ch;
          b.addEventListener("click", (ev) => {
            ev.preventDefault();
            appendEnrollName(ch.toLowerCase());
          });
          els.oskEn.appendChild(b);
        });
      });
      ["_", "-"].forEach((ch) => {
        const b = document.createElement("button");
        b.type = "button";
        b.className = "osk-key";
        b.textContent = ch;
        b.addEventListener("click", (ev) => {
          ev.preventDefault();
          appendEnrollName(ch);
        });
        els.oskEn.appendChild(b);
      });
    }
    fillOskKeys(els.oskNum, "1234567890".split(""));

    els.enrollOsk.querySelectorAll(".osk-tab").forEach((tab) => {
      tab.addEventListener("click", () => {
        const key = tab.dataset.osk;
        els.enrollOsk.querySelectorAll(".osk-tab").forEach((t) => {
          t.classList.toggle("active", t === tab);
        });
        els.enrollOsk.querySelectorAll(".osk-panel").forEach((p) => {
          p.classList.toggle("active", p.dataset.panel === key);
        });
      });
    });

    if (els.oskBackspace) {
      els.oskBackspace.addEventListener("click", () => {
        const cur = els.enrollName.value || "";
        setEnrollName(cur.slice(0, -1));
      });
    }
    if (els.oskHide) {
      els.oskHide.addEventListener("click", () => {
        els.enrollOsk.hidden = true;
      });
    }
    if (els.btnEnrollKeyboard) {
      els.btnEnrollKeyboard.addEventListener("click", () => {
        els.enrollOsk.hidden = !els.enrollOsk.hidden;
      });
    }
    if (els.btnEnrollClearName) {
      els.btnEnrollClearName.addEventListener("click", () => setEnrollName(""));
    }
    if (els.enrollName) {
      // 触摸屏点输入框也弹出屏上键盘（避免依赖系统 IME）
      els.enrollName.addEventListener("focus", () => {
        els.enrollOsk.hidden = false;
      });
      els.enrollName.addEventListener("click", () => {
        els.enrollOsk.hidden = false;
      });
    }
  }

  async function startEnroll() {
    const name = (els.enrollName.value || "").trim();
    const sec = Number(els.enrollSec.value || 15);
    if (!name) {
      toast("请先用屏上键盘填写姓名");
      if (els.enrollOsk) els.enrollOsk.hidden = false;
      return;
    }
    if (state.recording) {
      toast("请先结束会议再注册声纹");
      return;
    }
    els.btnEnrollStart.disabled = true;
    els.enrollProgress.hidden = false;
    els.enrollBar.style.width = "0%";
    els.enrollProgressText.textContent = `请对着麦克风持续说话 ${sec} 秒…`;
    let elapsed = 0;
    const timer = setInterval(() => {
      elapsed += 0.2;
      const pct = Math.min(99, (elapsed / sec) * 100);
      els.enrollBar.style.width = `${pct}%`;
      const left = Math.max(0, Math.ceil(sec - elapsed));
      els.enrollProgressText.textContent =
        left > 0 ? `正在采集声纹… 还剩约 ${left} 秒` : "正在提取特征并保存…";
    }, 200);
    try {
      const data = await action("enroll", { name, duration_sec: sec });
      clearInterval(timer);
      if (data.ok) {
        els.enrollBar.style.width = "100%";
        const p = data.profile || {};
        els.enrollProgressText.textContent = `注册成功：${p.name || name}`;
        toast(`已注册 ${p.name || name}`);
        els.enrollName.value = "";
        loadEnrollList();
      } else {
        els.enrollProgressText.textContent = data.error || "注册失败";
      }
    } catch (err) {
      clearInterval(timer);
      els.enrollProgressText.textContent = "注册失败";
      toast(String(err));
    } finally {
      els.btnEnrollStart.disabled = false;
      setTimeout(() => {
        if (els.enrollProgress) els.enrollProgress.hidden = true;
      }, 2500);
    }
  }

  async function action(name, body) {
    const res = await fetch(`/api/action/${name}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body || {}),
    });
    const data = await res.json().catch(() => ({ ok: false }));
    if (!res.ok || data.ok === false) {
      toast(data.error || `操作失败: ${name}`);
    }
    return data;
  }

  async function loadHistory() {
    const data = await action("history");
    const items = data.items || [];
    // 保留 HUD 角标
    els.historyList.querySelectorAll(".hist-item").forEach((n) => n.remove());
    const corners = els.historyList.querySelectorAll(".corner");
    if (!corners.length) {
      ["tl", "tr", "bl", "br"].forEach((c) => {
        const s = document.createElement("span");
        s.className = `corner ${c}`;
        els.historyList.appendChild(s);
      });
    }
    items.forEach((item, i) => {
      const btn = document.createElement("button");
      btn.type = "button";
      btn.className = "hist-item";
      btn.innerHTML = `${item.mtime}<small>${item.preview || item.stem}</small>`;
      btn.addEventListener("click", async () => {
        els.historyList.querySelectorAll(".hist-item").forEach((n) => n.classList.remove("active"));
        btn.classList.add("active");
        const detail = await action("history_load", { stem: item.stem });
        setDoc(els.historyView, detail.lines || []);
      });
      els.historyList.appendChild(btn);
      if (i === 0) btn.click();
    });
    if (!items.length) {
      els.historyView.textContent = "暂无历史记录";
    }
  }

  function applySnapshot(payload) {
    if (!payload) return;
    state.recording = !!payload.recording;
    state.paused = !!payload.paused;
    state.stem = payload.stem;
    state.ready = !!payload.ready;
    setStatus(payload.phase, payload.text);
    if (Array.isArray(payload.lines)) {
      clearTranscript();
      if (els.emptyHint) els.emptyHint.remove();
      payload.lines.forEach((line) => appendTranscript(line));
      if (!payload.lines.length) clearTranscript();
    }
    if (payload.summary) setDoc(els.summaryDoc, payload.summary);
    if (payload.agenda) setDoc(els.agendaDoc, payload.agenda);
    refreshButtons();
  }

  function onEvent(kind, payload) {
    if (kind === "snapshot") {
      applySnapshot(payload);
      return;
    }
    if (kind === "ping") return;
    if (kind === "status") {
      state.recording = payload.phase === "recording" || (state.recording && payload.phase === "agent");
      if (payload.phase === "idle" && /已结束|已放弃|待命/.test(payload.text || "")) {
        state.recording = false;
        state.paused = false;
      }
      if (payload.phase === "paused") {
        state.recording = true;
        state.paused = true;
      }
      if (payload.phase === "recording") {
        state.recording = true;
        state.paused = false;
      }
      setStatus(payload.phase, payload.text);
      refreshButtons();
      return;
    }
    if (kind === "ready") {
      state.ready = true;
      setStatus("idle", payload.text || "就绪");
      appendTranscript(payload.text || "系统就绪", { system: true });
      refreshButtons();
      return;
    }
    if (kind === "transcript") {
      appendTranscript(String(payload));
      return;
    }
    if (kind === "transcript_clear") {
      clearTranscript();
      return;
    }
    if (kind === "summary") {
      setDoc(els.summaryDoc, payload);
      showPage("summary");
      return;
    }
    if (kind === "agenda") {
      setDoc(els.agendaDoc, payload);
      showPage("agenda");
      return;
    }
    if (kind === "enroll") {
      loadEnrollList();
      return;
    }
    if (kind === "error") {
      toast(String(payload));
      setStatus(state.phase, String(payload));
      appendTranscript(`错误：${payload}`, { system: true });
    }
  }

  function connectSSE() {
    if (els.teleLink) els.teleLink.textContent = "…";
    const es = new EventSource("/api/events");
    es.onopen = () => {
      if (els.teleLink) els.teleLink.textContent = "SSE";
    };
    es.onmessage = (ev) => {
      try {
        const msg = JSON.parse(ev.data);
        onEvent(msg.kind, msg.payload);
      } catch (_) {
        /* ignore */
      }
    };
    es.onerror = () => {
      if (els.teleLink) els.teleLink.textContent = "RETRY";
      es.close();
      setTimeout(connectSSE, 1500);
    };
  }

  document.querySelectorAll(".nav-item").forEach((btn) => {
    btn.addEventListener("click", () => showPage(btn.dataset.page));
  });

  const start = async () => {
    appendTranscript("正在启动实时转写…", { system: true });
    const data = await action("start");
    if (data.ok) {
      state.recording = true;
      state.paused = false;
      setStatus("recording", "录音中");
      appendTranscript("已开录，请对着麦克风说话", { system: true });
    }
    // 用服务端状态校正
    fetch("/api/state")
      .then((r) => r.json())
      .then(applySnapshot)
      .catch(() => {});
    refreshButtons();
  };

  els.btnStart.addEventListener("click", start);
  els.btnStartTop.addEventListener("click", start);
  els.btnPause.addEventListener("click", () => action("pause"));
  els.btnResume.addEventListener("click", () => action("resume"));
  els.btnEnd.addEventListener("click", () => action("end"));
  els.btnQuit.addEventListener("click", async () => {
    if (!confirm("将删除本场录音与转写，确认？")) return;
    await action("quit");
    clearTranscript();
  });
  els.btnSummary.addEventListener("click", () => {
    showPage("summary");
    action("summary");
  });
  els.btnAgenda.addEventListener("click", () => {
    showPage("agenda");
    action("agenda");
  });
  els.btnRefreshHistory.addEventListener("click", loadHistory);
  if (els.btnRefreshEnroll) els.btnRefreshEnroll.addEventListener("click", loadEnrollList);
  if (els.btnEnrollStart) els.btnEnrollStart.addEventListener("click", startEnroll);
  initEnrollOsk();
  els.btnCloseTop.addEventListener("click", () => {
    if (confirm("关闭页面？（后台服务仍可在终端停止）")) {
      window.close();
      toast("请在终端 Ctrl+C 停止服务");
    }
  });

  // 初始拉取 + SSE
  fetch("/api/state")
    .then((r) => r.json())
    .then(applySnapshot)
    .catch(() => setStatus("booting", "连接中…"))
    .finally(connectSSE);
})();
