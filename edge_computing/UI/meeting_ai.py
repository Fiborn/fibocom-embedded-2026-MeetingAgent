"""实时会议 AI 会话：麦克风 + 说话人 + ASR，供桌面 / Web UI 调度。"""
from __future__ import annotations

import sys
import threading
import time
from pathlib import Path
from typing import Callable, Optional

from paths import apply_runtime_paths


class MeetingSession:
    def __init__(self, on_utterance: Callable[[str], None]):
        self._on_utterance = on_utterance
        self._engine = None
        self._thread: Optional[threading.Thread] = None
        self._shutdown = threading.Event()
        self._recording = threading.Event()
        self._paused = threading.Event()
        self._ready = threading.Event()
        self._warmup_failed = threading.Event()
        self._warmup_error: str | None = None
        self._lock = threading.Lock()
        self._warm_started = False
        self._preroll_chunks = []  # list[(mono, multi|None)]
        self._preroll_samples = 0
        # 声纹注册：从实时麦流截取一段 mono
        self._enroll_active = threading.Event()
        self._enroll_done = threading.Event()
        self._enroll_chunks: list = []
        self._enroll_need = 0
        self._enroll_error: str | None = None
        self._setup_paths()

    @staticmethod
    def _setup_paths():
        import os

        os.environ.setdefault("PA_ALSA_PLUGHW", "1")

        runtime = apply_runtime_paths()
        speaker_root = runtime["speaker_id_dir"]
        speaker_src = speaker_root / "src"
        for path in (speaker_root, speaker_src):
            text = str(path)
            if text not in sys.path:
                sys.path.insert(0, text)

        import config as speaker_config  # noqa: WPS433

        mic_cfg = dict(runtime["mic"] or {})
        if mic_cfg.get("auto_detect_dov", True):
            detected = MeetingSession._detect_dov_device()
            if detected:
                device, channels = detected
                mic_cfg["alsa_device"] = device
                mic_cfg["channels"] = channels
                print(f"[AI] 自动检测麦克风 {device} channels={channels}")

        speaker_config.CAPTURE_BACKEND = mic_cfg.get("capture_backend", "alsa")
        speaker_config.ALSA_DEVICE = mic_cfg.get("alsa_device", "plughw:1,0")
        speaker_config.INPUT_DEVICE_INDEX = mic_cfg.get("device_index", 0)
        speaker_config.INPUT_CHANNELS = int(mic_cfg.get("channels", 2) or 2)
        speaker_config.PRIMARY_CHANNEL = mic_cfg.get("primary_channel", 0)
        speaker_config.SAMPLE_RATE = int(mic_cfg.get("sample_rate", 16000))
        speaker_config.CHUNK = int(mic_cfg.get("frames", 2048))
        # Timesintelli 常出现 ch1~3 音量为 0 → 阵列几乎静音、ASR 无字
        MeetingSession._ensure_usb_mic_gain(speaker_config.ALSA_DEVICE)

        speaker_cfg = runtime.get("speaker") or {}
        use_doa = bool(speaker_cfg.get("use_doa", True))
        # 环形阵列 DOA 需要真实 4 通道；2ch 假扩声会污染分人
        if speaker_config.INPUT_CHANNELS < 4:
            if use_doa:
                print(
                    f"[AI] 通道数={speaker_config.INPUT_CHANNELS}<4，已关闭 DOA，回退声纹+基频"
                )
            use_doa = False
        speaker_config.SPEAKER_USE_DOA = use_doa
        _float_map = {
            "doa_min_conf": "SPEAKER_DOA_MIN_CONF",
            "doa_diff_deg": "SPEAKER_DOA_DIFF_DEG",
            "doa_weight": "SPEAKER_DOA_WEIGHT",
            "same_threshold": "SPEAKER_SAME_THRESHOLD",
            "new_threshold": "SPEAKER_NEW_THRESHOLD",
            "sticky_min_sim": "SPEAKER_STICKY_MIN_SIM",
            "rematch_min": "SPEAKER_REMATCH_MIN",
            "freq_weight": "SPEAKER_FREQ_WEIGHT",
            "merge_threshold": "SPEAKER_MERGE_THRESHOLD",
            "enroll_match_threshold": "SPEAKER_ENROLL_MATCH_THRESHOLD",
            "enroll_match_floor": "SPEAKER_ENROLL_MATCH_FLOOR",
            "enroll_match_margin": "SPEAKER_ENROLL_MATCH_MARGIN",
        }
        for key, attr in _float_map.items():
            if key in speaker_cfg:
                setattr(speaker_config, attr, float(speaker_cfg[key]))
        # 自适应同人上限略高于 same_threshold，避免 bootstrap 后阈值掉回去
        if "same_threshold" in speaker_cfg:
            same = float(speaker_cfg["same_threshold"])
            cap = float(getattr(speaker_config, "SPEAKER_SAME_THRESHOLD_CAP", same))
            speaker_config.SPEAKER_SAME_THRESHOLD_CAP = max(cap, same + 0.06)

        asr_cfg = runtime.get("asr") or {}
        _asr_float = {
            "vad_threshold": "VAD_THRESHOLD",
            "min_silence_duration": "MIN_SILENCE_DURATION",
            "min_speech_duration": "MIN_SPEECH_DURATION",
            "target_rms": "ASR_TARGET_RMS",
            "max_gain": "ASR_MAX_GAIN",
            "short_merge_max_sec": "ASR_SHORT_MERGE_MAX_SEC",
        }
        for key, attr in _asr_float.items():
            if key in asr_cfg:
                setattr(speaker_config, attr, float(asr_cfg[key]))
        _asr_bool = {
            "adaptive_primary": "ASR_ADAPTIVE_PRIMARY",
            "use_best_channel": "ASR_USE_BEST_CHANNEL",
            "channel_mix": "ASR_CHANNEL_MIX",
            "post_correct": "ASR_POST_CORRECT",
        }
        for key, attr in _asr_bool.items():
            if key in asr_cfg:
                setattr(speaker_config, attr, bool(asr_cfg[key]))
        if "queue_max" in asr_cfg:
            speaker_config.ASR_QUEUE_MAX = int(asr_cfg["queue_max"])
        if asr_cfg:
            print(
                "[AI] ASR 调优: vad={:.2f} silence={:.2f}s best_ch={} mix={}".format(
                    float(getattr(speaker_config, "VAD_THRESHOLD", 0.22)),
                    float(getattr(speaker_config, "MIN_SILENCE_DURATION", 0.70)),
                    bool(getattr(speaker_config, "ASR_USE_BEST_CHANNEL", True)),
                    bool(getattr(speaker_config, "ASR_CHANNEL_MIX", True)),
                )
            )

    @staticmethod
    def _ensure_usb_mic_gain(alsa_device: str) -> None:
        """按声卡实际控件拉满 Capture，并打开 Keep Interface（兼容 DOV / Timesintelli）。"""
        import re
        import subprocess

        m = re.search(r"(?:plug)?hw:(\d+)", str(alsa_device or ""))
        if not m:
            return
        card = m.group(1)
        try:
            contents = subprocess.check_output(
                ["amixer", "-c", card, "contents"],
                text=True,
                stderr=subprocess.STDOUT,
                timeout=3,
            )
        except Exception:
            return
        if "Mic" not in contents and "Capture" not in contents and "Keep Interface" not in contents:
            return

        blocks = re.split(r"\n(?=numid=)", contents)
        applied = []
        for block in blocks:
            nm = re.search(r"numid=(\d+)", block)
            name_m = re.search(r"name='([^']+)'", block)
            if not nm or not name_m:
                continue
            numid = nm.group(1)
            name = name_m.group(1)
            type_m = re.search(r"type=([A-Z]+)", block)
            values_m = re.search(r"values=(\d+)", block)
            max_m = re.search(r"max=(\d+)", block)
            ctype = (type_m.group(1) if type_m else "").upper()
            nvals = int(values_m.group(1)) if values_m else 1
            vmax = int(max_m.group(1)) if max_m else None

            try:
                if name == "Keep Interface" and ctype == "BOOLEAN":
                    subprocess.run(
                        ["amixer", "-c", card, "-q", "cset", f"numid={numid}", "on"],
                        capture_output=True,
                        timeout=2,
                    )
                    applied.append(f"keep@{numid}=on")
                elif "Capture Switch" in name and ctype == "BOOLEAN":
                    payload = ",".join(["on"] * max(1, nvals))
                    subprocess.run(
                        ["amixer", "-c", card, "-q", "cset", f"numid={numid}", payload],
                        capture_output=True,
                        timeout=2,
                    )
                    applied.append(f"switch@{numid}={payload}")
                elif "Capture Volume" in name and ctype == "INTEGER" and vmax is not None:
                    payload = ",".join([str(vmax)] * max(1, nvals))
                    subprocess.run(
                        ["amixer", "-c", card, "-q", "cset", f"numid={numid}", payload],
                        capture_output=True,
                        timeout=2,
                    )
                    # 再写一次，避免部分 USB 麦只吃到首通道
                    subprocess.run(
                        ["amixer", "-c", card, "-q", "cset", f"numid={numid}", payload],
                        capture_output=True,
                        timeout=2,
                    )
                    applied.append(f"vol@{numid}={payload}")
            except Exception:
                continue

        if applied:
            print(f"[AI] USB 麦增益已配置 card={card} ({'; '.join(applied)})")
        else:
            print(f"[AI] USB 麦未找到可写 Capture 控件 card={card}")

    @staticmethod
    def _usb_mic_present_but_no_card() -> bool:
        """lsusb 能看到会议麦，但 ALSA 无对应 card → 设备挂死，需拔插。"""
        import re
        import subprocess

        try:
            usb = subprocess.check_output(["lsusb"], text=True, stderr=subprocess.STDOUT)
        except Exception:
            return False
        usb_l = usb.lower()
        # Timesintelli 或 CF-IC DOV
        if not any(x in usb_l for x in ("cafe:4110", "timesintelli", "ff00:0002", "cf-ic")):
            return False
        try:
            cards = Path("/proc/asound/cards").read_text(encoding="utf-8", errors="ignore")
        except OSError:
            return True
        return not bool(
            re.search(r"Timesintelli|USB-Audio|\[Device\s*\]|\bDOV\b", cards, re.I)
        )
    @staticmethod
    def _detect_dov_device():
        """返回 (alsa_device, channels) 或 None。优先 DOV 卡，探测 4/2 通道。"""
        import re
        import subprocess

        if MeetingSession._usb_mic_present_but_no_card():
            raise RuntimeError(
                "检测到 Timesintelli USB 麦，但系统未注册声卡（设备已挂死）。"
                "请拔掉 USB 麦等待 5 秒再插上，然后重启程序。"
            )

        try:
            out = subprocess.check_output(
                ["arecord", "-l"], text=True, stderr=subprocess.STDOUT
            )
        except Exception:
            return None
        card = None
        # 注意：arecord -l 里板载卡也有 "USB Audio Hostless" 行，不能泛匹配 "USB Audio"
        patterns = (
            r"^card (\d+):.*\bDOV\b",
            r"^card (\d+):.*USB Audio Device",
            r"^card (\d+):.*Timesintelli",
        )
        for pat in patterns:
            for line in out.splitlines():
                if "Hostless" in line:
                    continue
                m = re.search(pat, line, re.I)
                if m:
                    card = int(m.group(1))
                    break
            if card is not None:
                break
        # /proc/asound/cards 长名里常有 Timesintelli
        if card is None:
            try:
                cards_txt = Path("/proc/asound/cards").read_text(encoding="utf-8", errors="ignore")
            except OSError:
                cards_txt = ""
            m = re.search(
                r"^\s*(\d+)\s+\[.*\].*Timesintelli|^\s*(\d+)\s+\[Device\s*\].*USB-Audio",
                cards_txt,
                re.I | re.M,
            )
            if m:
                card = int(m.group(1) or m.group(2))
        if card is None:
            return None

        # Timesintelli / 部分 USB 阵列麦：硬件原生就是 4ch@16k（见 /proc/asound/cardX/stream0）
        # 强行 -c 2 会把设备打进 xrun/PREPARED 坏状态，表现为持续「麦克风读取失败」
        # 注意：CF-IC DOV 名含 DOV，但是 2ch@44.1/48k，不能按名字强制 4ch。
        native_4ch = False
        try:
            stream_txt = Path("/proc/asound/card{}/stream0".format(card)).read_text(
                encoding="utf-8", errors="ignore"
            )
            if re.search(r"Channels:\s*4", stream_txt) and re.search(
                r"Rates:\s*[^\n]*\b16000\b", stream_txt
            ):
                native_4ch = True
        except OSError:
            pass

        # 原生 4ch 优先 hw:（避免 plughw 改格式）；否则 plughw 兼容探测（DOV 需 48k→16k）
        device_candidates = (
            ["hw:{},0".format(card), "plughw:{},0".format(card)]
            if native_4ch
            else ["plughw:{},0".format(card), "hw:{},0".format(card)]
        )
        ch_order = (4, 2, 1) if native_4ch else (2, 1, 4)

        for device in device_candidates:
            for ch in ch_order:
                cmd = [
                    "arecord", "-D", device, "-f", "S16_LE", "-r", "16000",
                    "-c", str(ch), "-d", "1", "-t", "raw", "/tmp/_ui_mic_probe.raw",
                ]
                try:
                    proc = subprocess.run(cmd, capture_output=True, timeout=5)
                except Exception:
                    continue
                if proc.returncode != 0:
                    continue
                try:
                    import numpy as np

                    raw = np.fromfile("/tmp/_ui_mic_probe.raw", dtype=np.int16)
                    if raw.size < ch * 1600:  # 至少约 0.1s
                        continue
                    frames = raw.reshape(-1, ch).astype(np.float32)
                    live = sum(
                        1 for i in range(ch) if float(np.max(np.abs(frames[:, i]))) > 2
                    )
                    if ch == 4 and live < 3 and not native_4ch:
                        # 非原生 4ch 的假扩：后两路静音
                        continue
                    if ch == 4 and not native_4ch:
                        corr_hi = False
                        for a, b in ((0, 2), (1, 3)):
                            c1, c2 = frames[:, a], frames[:, b]
                            if np.std(c1) < 1e-3 or np.std(c2) < 1e-3:
                                continue
                            if abs(float(np.corrcoef(c1, c2)[0, 1])) >= 0.98:
                                corr_hi = True
                                break
                        if corr_hi:
                            print("[AI] 检测到假 4 通道，跳过")
                            continue
                    print(
                        "[AI] 麦克风探测成功 {} ch={} live={} native4={}".format(
                            device, ch, live, native_4ch
                        )
                    )
                    return device, ch
                except Exception:
                    return device, ch
        # 探测都失败时仍按硬件能力给默认值，避免回退成错误的 plughw/2ch
        if native_4ch:
            return "hw:{},0".format(card), 4
        return "plughw:{},0".format(card), 2

    def _handle_utterance(self, speaker: str, text: str):
        if self._paused.is_set() or not self._recording.is_set():
            return
        self._on_utterance(f"{speaker}: {text}")

    def warmup(self):
        """程序启动时预热：加载 ASR + 打开麦克风（仅首次较慢）。"""
        with self._lock:
            if self._warm_started:
                return
            self._warm_started = True
            self._shutdown.clear()
            self._thread = threading.Thread(target=self._run_warm_loop, daemon=True)
            self._thread.start()
        print("[AI] 正在预热模型与麦克风，请稍候…")

    def _fail_warmup(self, message: str) -> None:
        self._warmup_error = message
        self._warmup_failed.set()
        print(f"[AI] {message}")

    def start(self) -> bool:
        if not self._warm_started:
            self.warmup()
        if self._warmup_failed.is_set():
            print(f"[AI] 无法启动转写: {self._warmup_error or '预热失败'}")
            return False
        deadline = time.time() + 120
        while time.time() < deadline:
            if self._warmup_failed.is_set():
                print(f"[AI] 无法启动转写: {self._warmup_error or '预热失败'}")
                return False
            if self._ready.wait(timeout=0.5):
                break
        else:
            print("[AI] 预热超时，请检查麦克风与 ASR 模型")
            return False
        # USB 麦增益常被系统/重枚举打回 28,0,0,0，开录前强制拉满
        try:
            import config as speaker_config  # noqa: WPS433

            MeetingSession._ensure_usb_mic_gain(
                getattr(speaker_config, "ALSA_DEVICE", "hw:1,0")
            )
        except Exception as exc:
            print(f"[AI] 开录前增益校正失败: {exc}")
        with self._lock:
            if self._recording.is_set():
                print("[AI] 会话已在运行")
                return False
            if self._engine is not None:
                self._engine.begin_session()
                # 开录瞬间注入预滚，减轻首句从半截开始
                preroll = list(self._preroll_chunks)
                fed = 0
                for mono, multi in preroll:
                    if mono is None or len(mono) == 0:
                        continue
                    self._engine.feed_16k(mono, multi=multi)
                    fed += len(mono)
                if fed > 0:
                    print(f"[AI] 已注入预滚 {fed / 16000.0:.2f}s")
            self._paused.clear()
            self._recording.set()
        print("[AI] 实时转写已启动")
        return True

    def _push_preroll(self, mono, multi):
        """未录音时维护最近 ASR_PREROLL_SEC 音频。"""
        if mono is None or len(mono) == 0:
            return
        try:
            import config as speaker_config  # noqa: WPS433

            max_sec = float(getattr(speaker_config, "ASR_PREROLL_SEC", 0.80))
        except Exception:
            max_sec = 0.80
        max_samples = int(max_sec * 16000)
        if max_samples <= 0:
            return
        self._preroll_chunks.append((mono.copy(), None if multi is None else multi.copy()))
        self._preroll_samples += len(mono)
        while self._preroll_samples > max_samples and self._preroll_chunks:
            old_m, _old_mu = self._preroll_chunks.pop(0)
            self._preroll_samples -= len(old_m)
        if self._preroll_samples < 0:
            self._preroll_samples = 0

    def _run_warm_loop(self):
        from meeting_realtime import RealtimeMeetingEngine  # noqa: WPS433

        mic = None
        # 先加载引擎/ASR，再开麦克风。否则 arecord 管道在等待 ASR 时被写满后退出，
        # 表现为持续「麦克风读取失败」。
        try:
            self._engine = RealtimeMeetingEngine(on_utterance=self._handle_utterance)
        except Exception as exc:
            self._fail_warmup(f"引擎加载失败: {exc}")
            return

        if not self._engine._asr_ready.wait(timeout=120):
            self._fail_warmup("ASR 加载超时（>120s），请检查模型文件")
            return
        if self._engine._asr_error is not None:
            self._fail_warmup(f"ASR 加载失败: {self._engine._asr_error}")
            return

        try:
            from audio_capture import create_microphone_reader  # noqa: WPS433

            mic = create_microphone_reader()
        except Exception as exc:
            self._fail_warmup(f"麦克风打开失败: {exc}")
            return

        self._ready.set()
        print("[AI] 预热完成（含 ASR），可按 start 开始转写")

        fail_streak = 0
        silent_streak = 0
        gain_watch = 0
        try:
            while not self._shutdown.is_set():
                try:
                    from audio_utils import unpack_read_16k  # noqa: WPS433

                    # 必须用原生 CHUNK（默认 2048）整块读；硬写 1024 会和泵块不对齐
                    chunk, level, multi = unpack_read_16k(mic.read_16k())
                    fail_streak = 0
                    # 周期性校正 USB 麦增益（DOV / Timesintelli 控件布局不同，走统一配置函数）
                    gain_watch += 1
                    if self._recording.is_set() and gain_watch % 40 == 0:
                        try:
                            import config as _sc  # noqa: WPS433

                            MeetingSession._ensure_usb_mic_gain(
                                getattr(_sc, "ALSA_DEVICE", "plughw:1,0")
                            )
                        except Exception:
                            pass
                    # 录音中长时间近似静音 → 多半是麦挂死/增益回落，不是 ASR 坏了
                    if self._recording.is_set() and not self._paused.is_set():
                        if float(level or 0.0) < 0.0015:
                            silent_streak += 1
                            if silent_streak in (40, 120):
                                print(
                                    "[AI] 麦克风近似静音，重新拉满 USB 增益…"
                                )
                                try:
                                    import config as _sc  # noqa: WPS433

                                    MeetingSession._ensure_usb_mic_gain(
                                        getattr(_sc, "ALSA_DEVICE", "hw:1,0")
                                    )
                                except Exception:
                                    pass
                            if silent_streak == 80:
                                print(
                                    "[AI] 警告: 麦克风输入接近静音。"
                                    "请大声说话，或拔插 USB 麦后重启 UI。"
                                )
                        else:
                            silent_streak = 0
                    else:
                        silent_streak = 0
                except OSError as exc:
                    fail_streak += 1
                    print(f"[AI] 麦克风读取失败: {exc}")
                    if fail_streak >= 6:
                        msg = (
                            "麦克风连续失败。若日志出现 endpoint not enabled / error -71，"
                            "请拔掉再插上 USB 麦后重启 UI。"
                        )
                        print(f"[AI] {msg} ({exc})")
                        self._warmup_error = msg
                        self._ready.clear()
                        break
                    # 退避重开，避免把 USB 打挂；每 3 次重建 reader 并重新探测设备
                    backoff = min(2.0, 0.4 * fail_streak)
                    if self._shutdown.wait(backoff):
                        break
                    try:
                        if fail_streak % 3 == 0:
                            try:
                                mic.close()
                            except Exception:
                                pass
                            # 重新跑一次设备探测，刷新卡号/通道
                            try:
                                self._setup_paths()
                            except Exception as setup_exc:
                                print(f"[AI] 重新探测麦克风失败: {setup_exc}")
                            from audio_capture import create_microphone_reader  # noqa: WPS433

                            mic = create_microphone_reader()
                        else:
                            mic._open_stream()
                    except Exception as reopen_exc:
                        print(f"[AI] 无法恢复麦克风: {reopen_exc}")
                        continue
                    continue

                # 声纹注册采集：占用麦流，不进会议 ASR；优先多通道能量混合
                if self._enroll_active.is_set() and chunk is not None and len(chunk):
                    try:
                        import numpy as np  # noqa: WPS433
                        from audio_utils import select_asr_mono  # noqa: WPS433

                        enroll_chunk = select_asr_mono(chunk, multi)
                        enroll_chunk = np.asarray(enroll_chunk, dtype=np.float32).reshape(-1)
                    except Exception:
                        enroll_chunk = chunk
                    self._enroll_chunks.append(enroll_chunk.copy())
                    got = sum(len(c) for c in self._enroll_chunks)
                    if got >= self._enroll_need:
                        self._enroll_active.clear()
                        self._enroll_done.set()
                    continue

                if not self._recording.is_set() or self._paused.is_set():
                    # 未开录也持续缓存预滚，供 start 时注入
                    self._push_preroll(chunk, multi)
                    continue
                if self._engine is not None:
                    self._engine.feed_16k(chunk, multi=multi)
                    # 录音中也滑动更新预滚，便于 pause/resume 后仍有上下文
                    self._push_preroll(chunk, multi)
        finally:
            if mic is not None:
                try:
                    mic.close()
                except Exception:
                    pass
            self._ready.clear()
            self._recording.clear()

    def capture_enroll_audio(self, duration_sec: float = 15.0, timeout: float | None = None):
        """从当前 ALSA 麦流截取 duration_sec 秒 mono float32，供声纹注册。"""
        import numpy as np

        if not self._ready.is_set():
            raise RuntimeError(self._warmup_error or "系统尚未就绪，请稍候")
        if self._recording.is_set():
            raise RuntimeError("会议进行中，请先结束会议再注册声纹")
        if self._enroll_active.is_set():
            raise RuntimeError("已有注册录音在进行")

        duration_sec = float(duration_sec)
        if duration_sec < 8.0:
            duration_sec = 8.0
        if duration_sec > 30.0:
            duration_sec = 30.0
        if timeout is None:
            timeout = duration_sec + 12.0

        self._enroll_chunks = []
        self._enroll_need = int(duration_sec * 16000)
        self._enroll_error = None
        self._enroll_done.clear()
        self._enroll_active.set()
        ok = self._enroll_done.wait(timeout=timeout)
        self._enroll_active.clear()
        if not ok:
            raise TimeoutError("注册录音超时，请检查麦克风是否正常")
        if not self._enroll_chunks:
            raise RuntimeError("未采集到音频")
        audio = np.concatenate(self._enroll_chunks).astype(np.float32, copy=False)
        self._enroll_chunks = []
        return audio

    def enroll_speaker(self, name: str, duration_sec: float = 15.0):
        """录音 → 提取声纹/频率 → 落盘 → 刷新 Tracker。"""
        from speaker_enroll import (  # noqa: WPS433
            EnrollStore,
            build_profile_from_audio,
            sanitize_name,
        )

        name = sanitize_name(name)
        samples = self.capture_enroll_audio(duration_sec=duration_sec)
        if self._engine is None or getattr(self._engine, "tracker", None) is None:
            raise RuntimeError("说话人引擎未就绪")
        tracker = self._engine.tracker
        emb, feats = build_profile_from_audio(tracker, samples, 16000)
        info = EnrollStore().save_profile(
            name,
            emb,
            feats,
            duration_sec=len(samples) / 16000.0,
            segments=1,
        )
        tracker.reload_enrolled()
        print(
            "[注册] 完成 {} pitch={} f1={} f2={}".format(
                info["name"],
                info.get("pitch"),
                info.get("f1"),
                info.get("f2"),
            )
        )
        return info

    def list_enrolled(self):
        from speaker_enroll import EnrollStore  # noqa: WPS433

        return EnrollStore().list_profiles()

    def delete_enrolled(self, name: str) -> bool:
        from speaker_enroll import EnrollStore  # noqa: WPS433

        ok = EnrollStore().delete(name)
        if ok and self._engine is not None and getattr(self._engine, "tracker", None):
            # 清匿名后重载注册表（会议中勿随意删；UI 会拦截录音中删除）
            if not self._recording.is_set():
                self._engine.tracker.reset()
            else:
                self._engine.tracker.reload_enrolled()
        return ok

    def pause(self):
        self._paused.set()
        if self._engine is not None:
            if hasattr(self._engine, "on_pause"):
                self._engine.on_pause()
            else:
                self._engine.paused = True
        print("[AI] 转写已暂停")

    def resume(self):
        self._paused.clear()
        if self._engine is not None:
            if hasattr(self._engine, "on_resume"):
                self._engine.on_resume()
            else:
                self._engine.paused = False
        print("[AI] 转写已恢复")

    def stop(self, wav_path: str | Path | None = None) -> bool:
        if not self._recording.is_set():
            return False
        self._recording.clear()
        self._paused.clear()

        saved = False
        if self._engine is not None:
            try:
                saved = self._engine.end_session(wav_path)
                if wav_path:
                    if saved:
                        print(f"[AI] 音频已保存: {wav_path}")
                    else:
                        print(f"[AI] 无有效音频数据: {wav_path}")
            except Exception as exc:
                print(f"[AI] 结束会话失败: {exc}")

        print("[AI] 实时转写已停止")
        return saved

    def shutdown(self):
        self._recording.clear()
        self._shutdown.set()
        if self._thread is not None:
            self._thread.join(timeout=8)
            self._thread = None
        self._engine = None
        self._warm_started = False
        self._ready.clear()
        self._warmup_failed.clear()
        self._warmup_error = None
        print("[AI] 会话已关闭")

    @property
    def running(self) -> bool:
        return self._recording.is_set()

    @property
    def ready(self) -> bool:
        return self._ready.is_set()
