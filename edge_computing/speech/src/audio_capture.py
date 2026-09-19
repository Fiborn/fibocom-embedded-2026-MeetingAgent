# -*- coding: utf-8 -*-
"""麦克风采集：优先 ALSA 直连（板端 USB 阵列麦），PyAudio 作为备选。"""
import os
import re
import subprocess
import sys
import threading
from pathlib import Path

import numpy as np
import pyaudio

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config
from audio_utils import (
    TARGET_SR,
    align_multi_to_mono,
    int16_bytes_to_multichannel_float32,
    resample,
    resample_multichannel,
    rms,
)

os.environ.setdefault("PA_ALSA_PLUGHW", "1")
_PA_LOCK = threading.Lock()


def list_input_devices(pa):
    devices = []
    try:
        default_idx = pa.get_default_input_device_info()["index"]
    except Exception:
        default_idx = -1
    for i in range(pa.get_device_count()):
        info = pa.get_device_info_by_index(i)
        if info["maxInputChannels"] > 0:
            devices.append({
                "index": i,
                "name": info["name"],
                "channels": int(info["maxInputChannels"]),
                "rate": int(info["defaultSampleRate"]),
                "default": i == default_idx,
            })
    return devices


def _open_input_stream(pa, idx, ch, rate, frames):
    info = pa.get_device_info_by_index(idx)
    if not pa.is_format_supported(
        rate,
        input_device=idx,
        input_channels=ch,
        input_format=pyaudio.paInt16,
    ):
        return None
    stream = pa.open(
        format=pyaudio.paInt16,
        channels=ch,
        rate=rate,
        input=True,
        input_device_index=idx,
        frames_per_buffer=frames,
    )
    print("[麦克风] 设备 [{}] {}".format(idx, info["name"]))
    print("[麦克风] 采样率 {} Hz, {} 声道, buffer={}".format(rate, ch, frames))
    return stream, rate, ch, idx, frames


def open_best_input_stream(pa, device_index=None):
    if device_index is None:
        device_index = config.INPUT_DEVICE_INDEX

    if device_index is not None:
        candidates = [device_index]
    else:
        devs = list_input_devices(pa)
        default = [d["index"] for d in devs if d["default"]]
        others = [d["index"] for d in devs if not d["default"]]
        candidates = default + others

    preferred_rate = int(getattr(config, "SAMPLE_RATE", 16000))
    preferred_frames = int(getattr(config, "CHUNK", 2048))
    fallback_rates = [preferred_rate, 16000, 48000, 44100, 32000, 8000]
    fallback_frames = [preferred_frames, 2048, 1024, 512]

    last_err = None
    for idx in candidates:
        try:
            info = pa.get_device_info_by_index(idx)
        except Exception as exc:
            last_err = exc
            continue
        max_ch = max(1, int(info["maxInputChannels"]))
        requested_ch = max(1, int(getattr(config, "INPUT_CHANNELS", 1) or 1))
        dev_ch: list[int] = []
        for ch in (min(requested_ch, max_ch), max_ch, 2, 1):
            if ch >= 1 and ch <= max_ch and ch not in dev_ch:
                dev_ch.append(ch)
        dev_rates = list(dict.fromkeys([preferred_rate, int(info["defaultSampleRate"])] + fallback_rates))
        for ch in dev_ch:
            for rate in dev_rates:
                for frames in fallback_frames:
                    try:
                        if not pa.is_format_supported(
                            rate,
                            input_device=idx,
                            input_channels=ch,
                            input_format=pyaudio.paInt16,
                        ):
                            continue
                        opened = _open_input_stream(pa, idx, ch, rate, frames)
                        if opened is not None:
                            if ch != requested_ch:
                                print(
                                    "[麦克风] 提示: 配置 {} 声道不可用，已改用 {} 声道".format(
                                        requested_ch, ch
                                    )
                                )
                            return opened
                    except Exception as exc:
                        last_err = exc

    raise RuntimeError(
        "无法打开麦克风。请运行 python3 src/list_devices.py 并设置 INPUT_DEVICE_INDEX\n"
        "最后错误: {}".format(last_err)
    )


class MicrophoneReader:
    """从麦克风读取，统一输出 16kHz float32"""

    def __init__(self):
        with _PA_LOCK:
            self.pa = pyaudio.PyAudio()
        self.stream = None
        self.capture_rate = TARGET_SR
        self.channels = 1
        self.device_index = 0
        self._block_native = 1024
        self._native_buf = np.array([], dtype=np.float32)
        self._native_multi_buf = np.zeros((1, 0), dtype=np.float32)
        self.max_rms = 0.0
        self.total_16k_samples = 0
        self._open_stream()

    def _open_stream(self):
        if self.stream is not None:
            self._close_stream_only()
        (
            self.stream,
            self.capture_rate,
            self.channels,
            self.device_index,
            self._block_native,
        ) = open_best_input_stream(self.pa, config.INPUT_DEVICE_INDEX)
        if (
            getattr(self, "_native_multi_buf", None) is None
            or self._native_multi_buf.size == 0
            or self._native_multi_buf.shape[0] != self.channels
        ):
            self._native_multi_buf = np.zeros((self.channels, 0), dtype=np.float32)

    def _close_stream_only(self):
        if self.stream is None:
            return
        try:
            if self.stream.is_active():
                self.stream.stop_stream()
        except Exception:
            pass
        try:
            self.stream.close()
        except Exception:
            pass
        self.stream = None

    def read_16k(self, block_native=None):
        block_native = block_native or self._block_native
        raw = None
        for attempt in range(3):
            try:
                raw = self.stream.read(block_native, exception_on_overflow=False)
                break
            except OSError as exc:
                if attempt < 2:
                    sys.stdout.write(
                        "\n[麦克风] 音频中断，正在恢复 ({})…\n".format(exc)
                    )
                    sys.stdout.flush()
                    try:
                        self._open_stream()
                    except Exception:
                        pass
                    continue
                raise

        multi_native = int16_bytes_to_multichannel_float32(raw, self.channels)
        # 与 ALSA 路径一致：优先能量最强通道
        if not hasattr(self, "_adaptive_primary"):
            self._adaptive_primary = int(getattr(config, "PRIMARY_CHANNEL", 0) or 0)
        primary = int(self._adaptive_primary)
        primary = max(0, min(primary, multi_native.shape[0] - 1))
        if multi_native.shape[0] > 1 and bool(
            getattr(config, "ASR_ADAPTIVE_PRIMARY", True)
        ):
            rms_ch = np.sqrt(
                np.mean(np.square(multi_native, dtype=np.float64), axis=1)
            )
            best = int(np.argmax(rms_ch))
            if float(rms_ch[best]) > float(rms_ch[primary]) * 1.35:
                primary = best
                self._adaptive_primary = best
        chunk = multi_native[primary]
        level = rms(chunk)
        self.max_rms = max(self.max_rms, level)

        self._native_buf = np.concatenate([self._native_buf, chunk])
        self._native_multi_buf = np.concatenate(
            [self._native_multi_buf, multi_native], axis=1
        )
        min_native = max(block_native, int(self.capture_rate * 0.05))
        out_16k = np.array([], dtype=np.float32)
        out_multi = np.zeros((self.channels, 0), dtype=np.float32)

        while len(self._native_buf) >= min_native:
            block = self._native_buf[:min_native]
            multi_block = self._native_multi_buf[:, :min_native]
            self._native_buf = self._native_buf[min_native:]
            self._native_multi_buf = self._native_multi_buf[:, min_native:]
            resampled = resample(block, self.capture_rate, TARGET_SR)
            multi_16k = resample_multichannel(
                multi_block, self.capture_rate, TARGET_SR
            )
            resampled, multi_16k = align_multi_to_mono(resampled, multi_16k)
            out_16k = np.concatenate([out_16k, resampled])
            if multi_16k is not None and multi_16k.size > 0:
                if out_multi.shape[0] != multi_16k.shape[0]:
                    out_multi = np.zeros((multi_16k.shape[0], 0), dtype=np.float32)
                out_multi = np.concatenate([out_multi, multi_16k], axis=1)

        if len(out_16k) > 0:
            self.total_16k_samples += len(out_16k)
        if out_multi.shape[1] == 0:
            out_multi = np.zeros((self.channels, 0), dtype=np.float32)
        out_16k, out_multi = align_multi_to_mono(out_16k, out_multi)
        return out_16k, level, out_multi

    def flush_16k(self):
        tail = np.array([], dtype=np.float32)
        tail_multi = np.zeros((self.channels, 0), dtype=np.float32)
        if len(self._native_buf) > 0:
            tail = resample(self._native_buf, self.capture_rate, TARGET_SR)
            tail_multi = resample_multichannel(
                self._native_multi_buf, self.capture_rate, TARGET_SR
            )
            self._native_buf = np.array([], dtype=np.float32)
            self._native_multi_buf = np.zeros((self.channels, 0), dtype=np.float32)
            self.total_16k_samples += len(tail)
        tail, tail_multi = align_multi_to_mono(tail, tail_multi)
        if tail_multi is None:
            tail_multi = np.zeros((self.channels, 0), dtype=np.float32)
        return tail, tail_multi

    def close(self):
        self._close_stream_only()
        if self.pa is not None:
            with _PA_LOCK:
                try:
                    self.pa.terminate()
                except Exception:
                    pass
            self.pa = None

    @property
    def duration_sec(self):
        return self.total_16k_samples / float(TARGET_SR)


def resolve_capture_card(prefer_names=("Timesintelli", "USB Audio Device", "DOV")):
    """从 /proc/asound/cards 解析当前 USB/阵列麦卡号，卡号漂移时也能找回。"""
    try:
        text = Path("/proc/asound/cards").read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return None
    for name in prefer_names:
        m = re.search(r"^\s*(\d+)\s+\[[^\]]*\]:.*{}".format(re.escape(name)), text, re.I | re.M)
        if m:
            return int(m.group(1))
        m = re.search(r"^\s*(\d+)\s+\[[^\]]*\].*\n\s+.*{}".format(re.escape(name)), text, re.I | re.M)
        if m:
            return int(m.group(1))
    return None


class AlsaMicrophoneReader:
    """通过 arecord 采集 USB/ALSA 设备，规避 PyAudio 在 QCS6490 上的兼容问题。

    使用后台线程持续排空 stdout，避免消费者稍慢时管道塞满导致 arecord 退出。
    Timesintelli 等设备原生多为 4ch@16k，勿强行 -c 2。
    """

    def __init__(self):
        self.device = getattr(config, "ALSA_DEVICE", "plughw:0,0")
        self.capture_rate = int(getattr(config, "SAMPLE_RATE", TARGET_SR))
        self.channels = max(1, int(getattr(config, "INPUT_CHANNELS", 1) or 1))
        self._block_native = int(getattr(config, "CHUNK", 2048))
        self._native_bytes = self._block_native * self.channels * 2
        self._native_buf = np.array([], dtype=np.float32)
        self._native_multi_buf = np.zeros((self.channels, 0), dtype=np.float32)
        # 泵线程按整块入队；消费者若按更小块取，必须回灌剩余字节，否则会丢一半音频
        self._byte_leftover = bytearray()
        self.max_rms = 0.0
        self.total_16k_samples = 0
        self._proc = None
        self._raw_q = __import__("queue").Queue(maxsize=64)
        self._reader_stop = threading.Event()
        self._reader_thread = None
        self._reader_error = None
        self._last_open_ts = 0.0
        self._adaptive_primary = int(getattr(config, "PRIMARY_CHANNEL", 0) or 0)
        self._open_stream()

    def _refresh_device_path(self):
        """卡号漂移时按设备名刷新 hw:/plughw: 路径。"""
        card = resolve_capture_card()
        if card is None:
            return
        # 保留当前前缀（hw / plughw）
        prefix = "hw"
        if isinstance(self.device, str) and self.device.startswith("plughw:"):
            prefix = "plughw"
        self.device = "{}:{},0".format(prefix, card)

    @staticmethod
    def _device_node_path(device: str):
        """hw:N,M / plughw:N,M → /dev/snd/pcmCN D M c；解析失败返回 None。"""
        m = re.search(r"(?:plug)?hw:(\d+),(\d+)", str(device or ""))
        if not m:
            return None
        return "/dev/snd/pcmC{}D{}c".format(m.group(1), m.group(2))

    def _assert_device_node_visible(self):
        """设备节点可见性检查。

        - USB 重枚举瞬间：/proc 有卡但 pcm 节点晚几百毫秒出现 → 短暂重试
        - Cursor 沙箱：长时间只有 /proc、没有 /dev/snd/pcmC* → 明确报错
        """
        import time as _time

        node = self._device_node_path(self.device)
        if node is None:
            return

        deadline = _time.time() + 3.0
        while _time.time() < deadline:
            self._refresh_device_path()
            node = self._device_node_path(self.device) or node
            if os.path.exists(node):
                return
            # 卡号可能漂移，按设备名再解析一次
            card = resolve_capture_card()
            if card is not None:
                alt = "/dev/snd/pcmC{}D0c".format(card)
                if os.path.exists(alt):
                    prefix = "plughw" if str(self.device).startswith("plughw:") else "hw"
                    self.device = "{}:{},0".format(prefix, card)
                    return
            _time.sleep(0.25)

        card_m = re.search(r"(?:plug)?hw:(\d+),", str(self.device or ""))
        card_listed = False
        if card_m:
            try:
                cards = open("/proc/asound/cards", "r", encoding="utf-8", errors="ignore").read()
                card_listed = re.search(
                    r"^\s*{}\s+\[".format(card_m.group(1)), cards, re.M
                ) is not None
            except OSError:
                pass
        usb_present = False
        try:
            import subprocess

            usb = subprocess.check_output(["lsusb"], text=True, stderr=subprocess.STDOUT)
            usb_l = usb.lower()
            usb_present = ("cafe:4110" in usb_l) or ("timesintelli" in usb_l)
        except Exception:
            pass

        if card_listed and not os.path.exists(node):
            # 有卡无节点：区分沙箱隔离 vs USB 半挂死
            snd_ok = os.path.isdir("/dev/snd") and bool(
                [p for p in os.listdir("/dev/snd") if p.startswith("pcmC0")]
            )
            if not snd_ok:
                raise OSError(
                    "打开麦克风失败 {}:{} — 声卡在 /proc 可见，但 /dev/snd 不可用（常见于 Cursor 沙箱）。"
                    "请在板端本机终端运行: cd /home/fibo/MeetingAgent/UI && bash web/run.sh".format(
                        self.device, self.channels
                    )
                )
            raise OSError(
                "打开麦克风失败 {}:{} — 设备节点 {} 暂不可用（USB 麦可能半挂死）。"
                "请拔掉 Timesintelli USB 麦等待 5 秒再插上，然后重启 UI。".format(
                    self.device, self.channels, node
                )
            )
        if usb_present:
            raise OSError(
                "打开麦克风失败 {}:{} — lsusb 能看到 USB 麦，但 ALSA 未注册 pcm 节点。"
                "请拔插 USB 麦后重试。".format(self.device, self.channels)
            )
        raise OSError(
            "打开麦克风失败 {}:{} — 设备节点 {} 不存在，请确认 USB 麦已连接且 arecord -l 有对应卡".format(
                self.device, self.channels, node
            )
        )

    def _format_open_error(self, msg: str) -> str:
        text = (msg or "").strip()
        low = text.lower()
        hung = any(
            k in low
            for k in (
                "xrun",
                "prepared",
                "endpoint not enabled",
                "error -71",
                "error -2",
                "no data",
                "无数据",
            )
        )
        # arecord 常把启动横幅打到 stderr，本身不是根因
        if "recording raw data" in low and not hung:
            hung = True
        if hung:
            return (
                "打开麦克风失败 {}:{} — USB 麦已挂死（endpoint not enabled / 无音频数据）。"
                "请拔掉 Timesintelli USB 麦，等待 5 秒再插上，然后重启 Web UI。"
                "详情: {}".format(self.device, self.channels, text[:160] or "n/a")
            )
        if "Invalid value for card" in text or "No such file or directory" in text:
            node = self._device_node_path(self.device) or "/dev/snd/pcmC?D?c"
            return (
                "打开麦克风失败 {}:{} — {}。"
                "若 {}: 不存在，说明当前环境隔离了 /dev/snd（Cursor 沙箱常见），"
                "请到板端本机终端运行 bash web/run.sh".format(
                    self.device, self.channels, text, node
                )
            )
        return "打开麦克风失败 {}:{} — {}".format(self.device, self.channels, text)

    def _open_stream(self):
        import time as _time

        # 避免狂重开把 USB 打进 endpoint not enabled
        gap = _time.time() - float(self._last_open_ts or 0.0)
        if gap < 0.8:
            _time.sleep(0.8 - gap)

        self._close_stream_only()
        self._refresh_device_path()
        self._assert_device_node_visible()
        self._reader_error = None
        self._reader_stop.clear()
        self._byte_leftover = bytearray()
        self._native_buf = np.array([], dtype=np.float32)
        self._native_multi_buf = np.zeros((self.channels, 0), dtype=np.float32)
        q = self._raw_q
        while True:
            try:
                q.get_nowait()
            except Exception:
                break

        # Timesintelli full-speed USB：加大缓冲，降低 ASR 忙时 overrun 丢样
        cmd = [
            "arecord",
            "-D",
            self.device,
            "-f",
            "S16_LE",
            "-r",
            str(self.capture_rate),
            "-c",
            str(self.channels),
            "-t",
            "raw",
            "--buffer-time=500000",
            "--period-time=32000",
        ]
        pipe_buf = max(self._native_bytes * 32, 1 << 20)
        self._proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=pipe_buf,
        )
        self._last_open_ts = _time.time()
        # 快速失败：进程立刻退出则抛错，便于上层换参数
        _time.sleep(0.05)
        if self._proc.poll() is not None:
            err = b""
            try:
                err = self._proc.stderr.read() if self._proc.stderr else b""
            except Exception:
                pass
            msg = err.decode("utf-8", errors="ignore").strip() or "arecord 立即退出"
            self._proc = None
            raise OSError(self._format_open_error(msg))

        self._stderr_thread = threading.Thread(
            target=self._drain_stderr, daemon=True
        )
        self._stderr_thread.start()
        self._reader_thread = threading.Thread(
            target=self._pump_stdout, daemon=True, name="alsa-arecord-pump"
        )
        self._reader_thread.start()

        # Timesintelli 挂死时常见：arecord 能启动，但 URB 提交失败 → 读不到任何数据
        # （dmesg: endpoint not enabled / error -71）。这里做首包探测，避免假开麦。
        try:
            probe = self._raw_q.get(timeout=1.2)
            if probe:
                self._byte_leftover.extend(probe)
        except Exception:
            err_hint = ""
            try:
                if self._reader_error is not None:
                    err_hint = str(self._reader_error)
            except Exception:
                pass
            self._close_stream_only()
            raise OSError(
                self._format_open_error(
                    err_hint
                    or "arecord 无数据（USB 麦 endpoint 可能未启用）"
                )
            )

        print("[麦克风] ALSA 设备 {}".format(self.device))
        print(
            "[麦克风] 采样率 {} Hz, {} 声道, buffer={}".format(
                self.capture_rate, self.channels, self._block_native
            )
        )

    def _pump_stdout(self):
        """持续读取 arecord stdout，防止管道背压把录音进程噎死。"""
        proc = self._proc
        if proc is None or proc.stdout is None:
            return
        chunk_bytes = max(self._native_bytes, 4096)
        try:
            while not self._reader_stop.is_set():
                if proc.poll() is not None:
                    self._reader_error = OSError(
                        "arecord 意外退出 (code={})".format(proc.returncode)
                    )
                    break
                data = proc.stdout.read(chunk_bytes)
                if not data:
                    if proc.poll() is not None:
                        self._reader_error = OSError(
                            "arecord 意外退出 (code={})".format(proc.returncode)
                        )
                    else:
                        self._reader_error = OSError("arecord 无数据输出")
                    break
                try:
                    self._raw_q.put(data, timeout=1.0)
                except Exception:
                    # 队列满：丢最旧块，保住实时性
                    try:
                        self._raw_q.get_nowait()
                    except Exception:
                        pass
                    try:
                        self._raw_q.put_nowait(data)
                    except Exception:
                        pass
        except Exception as exc:
            self._reader_error = OSError("arecord 泵线程异常: {}".format(exc))

    def _drain_stderr(self):
        proc = self._proc
        if proc is None or proc.stderr is None:
            return
        try:
            for line in iter(proc.stderr.readline, b""):
                msg = line.decode("utf-8", errors="ignore").strip()
                if msg:
                    sys.stderr.write("[arecord] {}\n".format(msg))
                    sys.stderr.flush()
        except Exception:
            pass

    def _close_stream_only(self):
        self._reader_stop.set()
        proc = self._proc
        self._proc = None
        if proc is not None:
            try:
                proc.terminate()
                proc.wait(timeout=2)
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass
        thr = self._reader_thread
        self._reader_thread = None
        if thr is not None and thr.is_alive():
            thr.join(timeout=1.0)

    def _pick_primary(self, multi_native):
        """自适应选择能量最强通道，供 VAD 更稳地听到语音。"""
        nch = int(multi_native.shape[0])
        primary = int(getattr(self, "_adaptive_primary", 0) or 0)
        primary = max(0, min(primary, nch - 1))
        if nch < 2 or not bool(getattr(config, "ASR_ADAPTIVE_PRIMARY", True)):
            return primary
        rms_ch = np.sqrt(np.mean(np.square(multi_native, dtype=np.float64), axis=1))
        best = int(np.argmax(rms_ch))
        best_r = float(rms_ch[best])
        cur_r = float(rms_ch[primary])
        # 有明显语音且显著强于当前通道才切换，避免来回抖
        if best_r > 1e-4 and best_r > cur_r * 1.35:
            if best != primary and config.DEBUG:
                print("[麦克风] 自适应主通道 {} -> {}".format(primary, best))
            self._adaptive_primary = best
            return best
        return primary

    def read_16k(self, block_native=None):
        block_native = block_native or self._block_native
        need_bytes = block_native * self.channels * 2
        raw = self._read_exact(need_bytes)
        multi_native = int16_bytes_to_multichannel_float32(raw, self.channels)
        primary = self._pick_primary(multi_native)
        chunk = multi_native[primary]
        level = rms(chunk)
        self.max_rms = max(self.max_rms, level)

        self._native_buf = np.concatenate([self._native_buf, chunk])
        self._native_multi_buf = np.concatenate(
            [self._native_multi_buf, multi_native], axis=1
        )
        min_native = max(block_native, int(self.capture_rate * 0.05))
        out_16k = np.array([], dtype=np.float32)
        out_multi = np.zeros((self.channels, 0), dtype=np.float32)

        while len(self._native_buf) >= min_native:
            block = self._native_buf[:min_native]
            multi_block = self._native_multi_buf[:, :min_native]
            self._native_buf = self._native_buf[min_native:]
            self._native_multi_buf = self._native_multi_buf[:, min_native:]
            resampled = resample(block, self.capture_rate, TARGET_SR)
            multi_16k = resample_multichannel(
                multi_block, self.capture_rate, TARGET_SR
            )
            resampled, multi_16k = align_multi_to_mono(resampled, multi_16k)
            out_16k = np.concatenate([out_16k, resampled])
            if multi_16k is not None and multi_16k.size > 0:
                if out_multi.shape[0] != multi_16k.shape[0]:
                    out_multi = np.zeros((multi_16k.shape[0], 0), dtype=np.float32)
                out_multi = np.concatenate([out_multi, multi_16k], axis=1)

        if len(out_16k) > 0:
            self.total_16k_samples += len(out_16k)
        if out_multi.shape[1] == 0:
            out_multi = np.zeros((self.channels, 0), dtype=np.float32)
        out_16k, out_multi = align_multi_to_mono(out_16k, out_multi)
        return out_16k, level, out_multi

    def _read_exact(self, nbytes: int) -> bytes:
        """精确取 nbytes；多读部分回灌，禁止丢弃（否则实时 ASR 会周期性丢半段）。"""
        if nbytes <= 0:
            return b""
        buf = self._byte_leftover
        self._byte_leftover = bytearray()
        while len(buf) < nbytes:
            if self._reader_error is not None:
                raise self._reader_error
            if self._proc is None:
                raise OSError("ALSA 录音进程未启动")
            try:
                chunk = self._raw_q.get(timeout=2.0)
            except Exception:
                if self._reader_error is not None:
                    raise self._reader_error
                if self._proc is not None and self._proc.poll() is not None:
                    raise OSError(
                        "arecord 意外退出 (code={})".format(self._proc.returncode)
                    )
                raise OSError("arecord 读取超时（无音频数据）")
            if chunk:
                buf.extend(chunk)
        out = bytes(buf[:nbytes])
        if len(buf) > nbytes:
            self._byte_leftover = bytearray(buf[nbytes:])
        return out

    def flush_16k(self):
        tail = np.array([], dtype=np.float32)
        tail_multi = np.zeros((self.channels, 0), dtype=np.float32)
        if len(self._native_buf) > 0:
            tail = resample(self._native_buf, self.capture_rate, TARGET_SR)
            tail_multi = resample_multichannel(
                self._native_multi_buf, self.capture_rate, TARGET_SR
            )
            self._native_buf = np.array([], dtype=np.float32)
            self._native_multi_buf = np.zeros((self.channels, 0), dtype=np.float32)
            self.total_16k_samples += len(tail)
        tail, tail_multi = align_multi_to_mono(tail, tail_multi)
        if tail_multi is None:
            tail_multi = np.zeros((self.channels, 0), dtype=np.float32)
        return tail, tail_multi

    def close(self):
        self._close_stream_only()


def parse_arecord_devices():
    try:
        out = subprocess.check_output(["arecord", "-l"], text=True, stderr=subprocess.STDOUT)
    except Exception:
        return []
    devices = []
    card = None
    for line in out.splitlines():
        card_match = re.search(r"^card (\d+): (.+?) \[(.+?)\]", line)
        if card_match:
            card = int(card_match.group(1))
            continue
        dev_match = re.search(r"device (\d+):", line)
        if card is not None and dev_match is not None:
            dev = int(dev_match.group(1))
            devices.append(
                {
                    "alsa": "plughw:{},{}".format(card, dev),
                    "name": line.strip(),
                }
            )
    return devices


def test_alsa_device(device, seconds=2, rate=16000, channels=1):
    cmd = [
        "arecord",
        "-D",
        device,
        "-f",
        "S16_LE",
        "-r",
        str(rate),
        "-c",
        str(channels),
        "-d",
        str(seconds),
        "-t",
        "raw",
    ]
    proc = subprocess.run(cmd, capture_output=True)
    if proc.returncode != 0:
        return 0.0
    samples = np.frombuffer(proc.stdout, dtype=np.int16).astype(np.float32) / 32768.0
    if channels > 1:
        samples = samples.reshape(-1, channels)[:, 0]
    return float(np.max(np.abs(samples))) if len(samples) else 0.0


def find_best_alsa_device(seconds=2):
    devices = parse_arecord_devices()
    if not devices:
        devices = [{"alsa": "plughw:0,0", "name": "fallback"}]
    best = (0.0, devices[0]["alsa"], 16000, 1)
    for item in devices:
        for ch in (4, 2, 1):
            peak = test_alsa_device(item["alsa"], seconds=seconds, rate=16000, channels=ch)
            if peak > best[0]:
                best = (peak, item["alsa"], 16000, ch)
    return best[1], best[2], best[3], best[0]


def create_microphone_reader():
    backend = str(getattr(config, "CAPTURE_BACKEND", "alsa")).lower()
    if backend == "alsa":
        return AlsaMicrophoneReader()
    return MicrophoneReader()


def print_level_meter(level, threshold=0.008):
    bar_len = 30
    filled = min(bar_len, int(level * 300))
    bar = "#" * filled + "-" * (bar_len - filled)
    status = "有声音" if level > threshold else "静音"
    sys.stdout.write("\r音量 [{bar}] {lvl:.4f}  {st}  ".format(
        bar=bar, lvl=level, st=status
    ))
    sys.stdout.flush()
