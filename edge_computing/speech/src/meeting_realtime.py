# -*- coding: utf-8 -*-
"""
实时会议引擎：麦克风 → 说完一句 → 立即打印「说话人N: 内容」
支持 4 通道 DOA 融合与合并后历史标签回写。
"""
import queue as queue_mod
import sys
import threading
import time
from pathlib import Path

import numpy as np
import sherpa_onnx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config
import re

from asr_engine import AsrEngine

try:
    from asr_engine import is_valid_transcript
except ImportError:
    def is_valid_transcript(text):
        text = (text or "").strip()
        if not text or text.startswith("{"):
            return False
        return bool(re.search(r"[\u4e00-\u9fffA-Za-z]", text))


def _is_noise_transcript(text):
    """重复词/语气词等低质量 ASR，不参与声纹更新"""
    t = text.strip().lower()
    words = t.split()
    if len(words) >= 3 and len(set(words)) <= 2:
        return True
    if re.fullmatch(r"(嗯|啊|哦|v|the)+", t.replace(" ", "")):
        return True
    if len(t) <= 6 and re.fullmatch(r"[\u4e00-\u9fff]{1,3}", t):
        return True
    return False


from audio_capture import MicrophoneReader, print_level_meter
from audio_utils import (
    TARGET_SR,
    align_multi_to_mono,
    normalize,
    normalize_for_asr,
    rms,
    unpack_flush_16k,
    unpack_read_16k,
)
from doa_bridge import DoaBridge
from speaker_tracker import SpeakerTracker, TRACKER_VERSION
from vad_webrtc import webrtc_available, webrtc_segments


class RealtimeMeetingEngine:
    def __init__(self, on_utterance=None):
        if not config.VAD_MODEL.exists():
            raise FileNotFoundError(
                "VAD 模型缺失，请运行: python3 scripts/download_model.py"
            )

        t0 = time.time()
        print("正在加载说话人 + VAD…", flush=True)
        self.tracker = SpeakerTracker()
        self.doa = DoaBridge()
        self.on_utterance = on_utterance or self._default_output
        self.min_speech_samples = int(config.MIN_SPEECH_DURATION * TARGET_SR)

        vad_cfg = sherpa_onnx.VadModelConfig()
        vad_cfg.silero_vad.model = str(config.VAD_MODEL)
        vad_cfg.silero_vad.min_silence_duration = config.MIN_SILENCE_DURATION
        vad_cfg.silero_vad.min_speech_duration = float(
            getattr(config, "VAD_MIN_SPEECH_DURATION", 0.10)
        )
        vad_cfg.silero_vad.threshold = float(getattr(config, "VAD_THRESHOLD", 0.12))
        vad_cfg.sample_rate = TARGET_SR
        self.window_size = vad_cfg.silero_vad.window_size
        self.vad = sherpa_onnx.VoiceActivityDetector(vad_cfg, buffer_size_in_seconds=120)

        self.asr = None
        self._asr_error = None
        self._asr_ready = threading.Event()
        self._asr_wait_printed = False
        print("轻量模块就绪 ({:.1f}s)".format(time.time() - t0), flush=True)
        threading.Thread(target=self._load_asr_background, daemon=True).start()

        self._audio_buffer = np.array([], dtype=np.float32)
        self._multi_ring = None  # (ch, samples) 与 mono 同步的环形缓冲
        self._all_16k_chunks = []
        self._log_lines = []
        self._seg_count = 0
        self._seen_transcripts = set()
        self._dropped_segments = 0
        self.paused = False
        self._session_t0 = None
        self._coldstart_active = False
        self._pending_short = None  # (asr_samples, multi, seg_rms, duration, t)

        self._asr_queue = queue_mod.Queue(
            maxsize=max(3, int(getattr(config, "ASR_QUEUE_MAX", 8)))
        )
        self._output_lock = threading.Lock()
        self._asr_worker_thread = threading.Thread(
            target=self._asr_worker, daemon=True
        )
        self._asr_worker_thread.start()

    def _load_asr_background(self):
        t0 = time.time()
        try:
            print("后台加载 ASR（2023 模型约 230MB，板端约 15~40s）…", flush=True)
            self.asr = AsrEngine()
            print("ASR 就绪 ({:.1f}s)".format(time.time() - t0), flush=True)
        except Exception as exc:
            self._asr_error = exc
            print("ASR 加载失败: {}".format(exc), flush=True)
        finally:
            self._asr_ready.set()

    def _get_asr(self):
        if self.asr is not None:
            return self.asr
        if self._asr_error is not None:
            raise self._asr_error
        if not self._asr_ready.wait(timeout=180):
            raise RuntimeError("ASR 加载超时（>180s）")
        if self._asr_error is not None:
            raise self._asr_error
        return self.asr

    @staticmethod
    def _default_output(speaker, text):
        print("\n{}: {}".format(speaker, text), flush=True)

    def _append_multi_ring(self, multi):
        if multi is None:
            return
        multi = np.asarray(multi, dtype=np.float32)
        if multi.ndim != 2 or multi.shape[1] == 0:
            return
        if self._multi_ring is None or self._multi_ring.size == 0:
            self._multi_ring = multi.copy()
        elif self._multi_ring.shape[0] != multi.shape[0]:
            self._multi_ring = multi.copy()
        else:
            self._multi_ring = np.concatenate([self._multi_ring, multi], axis=1)
        max_samples = int(
            getattr(config, "SPEAKER_MULTI_RING_SEC", 30.0) * TARGET_SR
        )
        if self._multi_ring.shape[1] > max_samples:
            self._multi_ring = self._multi_ring[:, -max_samples:]

    def _slice_multi_for_segment(self, seg_len):
        """从 multi ring 取与 VAD 段大致对齐的窗口。

        Silero 出段时 ring 尾部通常还叠了约 min_silence 的静音；若直接取
        ring[:, -seg_len:]，会丢掉句首、多吃句尾静音，ASR 表现为漏字。
        """
        if self._multi_ring is None or self._multi_ring.size == 0:
            return None
        if seg_len <= 0:
            return None
        n = self._multi_ring.shape[1]
        if n < max(8, seg_len // 4):
            return None
        # 优先扣除尾部静音垫；冷启动期间用当前 VAD 静音阈值
        try:
            pad = int(
                float(self.vad.config.silero_vad.min_silence_duration) * TARGET_SR
            )
        except Exception:
            pad = int(
                float(getattr(config, "MIN_SILENCE_DURATION", 0.55)) * TARGET_SR
            )
        pad = max(0, min(pad, n // 3))
        end = n - pad if (n - pad) >= max(8, seg_len // 4) else n
        start = max(0, end - seg_len)
        return self._multi_ring[:, start:end].copy()

    def _apply_label_map(self, merges):
        if not merges:
            return
        mapping = {}
        for old, new in merges:
            mapping[old] = mapping.get(new, new)
        # 传递闭包：A→B, B→C ⇒ A→C
        changed = True
        while changed:
            changed = False
            for k, v in list(mapping.items()):
                if v in mapping and mapping[v] != v:
                    mapping[k] = mapping[v]
                    changed = True
        with self._output_lock:
            new_lines = []
            for line in self._log_lines:
                if ": " not in line:
                    new_lines.append(line)
                    continue
                speaker, rest = line.split(": ", 1)
                speaker = mapping.get(speaker, speaker)
                new_lines.append("{}: {}".format(speaker, rest))
            self._log_lines = new_lines

    def _process_vad_queue(self):
        while not self.vad.empty():
            seg = np.array(self.vad.front.samples, dtype=np.float32)
            self.vad.pop()
            multi = self._slice_multi_for_segment(len(seg))
            self._handle_segment(seg, multi=multi)

    def _handle_segment(self, samples, multi=None):
        self._seg_count += 1
        samples = normalize(samples)
        duration = len(samples) / TARGET_SR

        max_seg_sec = getattr(config, "ASR_MAX_SEGMENT_SEC", 8.0)
        max_seg_samples = int(max_seg_sec * TARGET_SR)
        if len(samples) > max_seg_samples:
            for start in range(0, len(samples), max_seg_samples):
                end = start + max_seg_samples
                sub = samples[start:end]
                sub_multi = None
                if multi is not None and multi.shape[1] >= end:
                    sub_multi = multi[:, start:end]
                elif multi is not None and multi.shape[1] > 0:
                    # 长度不一致时取尾部对齐
                    take = min(len(sub), multi.shape[1])
                    sub_multi = multi[:, -take:]
                self._handle_single_segment(sub, multi=sub_multi)
            return
        self._handle_single_segment(samples, multi=multi)

    def _enqueue_asr(self, asr_samples, multi, seg_rms, duration):
        if not self._asr_ready.is_set() and not self._asr_wait_printed:
            print("\n[等待 ASR 后台加载完成…]", flush=True)
            self._asr_wait_printed = True

        item = (asr_samples, multi, seg_rms, duration)
        try:
            self._asr_queue.put_nowait(item)
        except queue_mod.Full:
            try:
                self._asr_queue.get_nowait()
                self._dropped_segments += 1
                print(
                    "[警告] ASR 队列满，丢弃旧段 (累计 {})".format(
                        self._dropped_segments
                    ),
                    flush=True,
                )
                self._asr_queue.put_nowait(item)
            except Exception:
                pass

    def _concat_multi(self, a, b):
        if a is None:
            return b
        if b is None:
            return a
        a = np.asarray(a, dtype=np.float32)
        b = np.asarray(b, dtype=np.float32)
        if a.ndim != 2 or b.ndim != 2 or a.shape[0] != b.shape[0]:
            return b if b is not None else a
        return np.concatenate([a, b], axis=1)

    def _flush_pending_short(self, force=False):
        """超时或会话结束时送出暂存短句。"""
        if self._pending_short is None:
            return
        asr_samples, multi, seg_rms, duration, t0 = self._pending_short
        gap = float(getattr(config, "ASR_SHORT_MERGE_GAP_SEC", 1.2))
        if not force and (time.time() - t0) < gap:
            return
        self._pending_short = None
        self._enqueue_asr(asr_samples, multi, seg_rms, duration)

    def _handle_single_segment(self, samples, multi=None):
        from audio_utils import enhance_for_asr, select_asr_mono  # noqa: WPS433

        # ASR 必须以 VAD 切段时间为准。multi 仅在与 mono 足够对齐时才换通道，
        # 否则会吃进“句首丢失 + 句尾静音”的错窗，表现为大量漏字。
        vad_mono = np.asarray(samples, dtype=np.float32).reshape(-1)
        if multi is not None:
            multi_arr = np.asarray(multi, dtype=np.float32)
            if (
                multi_arr.ndim == 2
                and multi_arr.shape[1] >= max(16, int(len(vad_mono) * 0.85))
            ):
                n = min(len(vad_mono), multi_arr.shape[1])
                # 用最佳通道与 VAD mono 的相关/能量比判断是否同窗
                cand = select_asr_mono(vad_mono[:n], multi_arr[:, :n])
                a = vad_mono[:n].astype(np.float64)
                b = np.asarray(cand, dtype=np.float64).reshape(-1)[:n]
                if len(a) >= 16 and len(b) >= 16:
                    a = a - a.mean()
                    b = b - b.mean()
                    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-12
                    corr = float(np.dot(a, b) / denom)
                    if corr >= 0.55:
                        samples = cand.astype(np.float32)
                    else:
                        samples = vad_mono
                        if config.DEBUG:
                            print(
                                "[DEBUG] multi 与 VAD 不对齐 corr={:.2f}，ASR 用 VAD mono".format(
                                    corr
                                ),
                                flush=True,
                            )
                else:
                    samples = vad_mono
            else:
                samples = vad_mono
        else:
            samples = vad_mono

        raw_peak = float(np.max(np.abs(samples))) if len(samples) else 0.0
        raw_rms = rms(samples)
        duration = len(samples) / TARGET_SR

        if len(samples) < self.min_speech_samples:
            print("[跳过] 片段过短 {:.2f}s".format(duration), flush=True)
            return

        # USB 麦挂死时常见：peak≈0，却被旧 normalize 抬成假语音 → ASR 空串
        if raw_peak < 0.002 or raw_rms < 5e-4:
            print(
                "[跳过] 麦克风近似静音 peak={:.5f} rms={:.5f}（请拔插 USB 麦或检查增益）".format(
                    raw_peak, raw_rms
                ),
                flush=True,
            )
            return

        samples = enhance_for_asr(samples, TARGET_SR)
        # Timesintelli 阵列：弱段限制增益，避免把底噪抬进 Paraformer 产生乱字
        max_gain = float(getattr(config, "ASR_MAX_GAIN", 12.0))
        if raw_rms < 0.008:
            max_gain = min(max_gain, 8.0)
        if raw_rms < 0.004:
            max_gain = min(max_gain, 4.0)
        asr_samples = normalize_for_asr(
            samples,
            target_rms=getattr(config, "ASR_TARGET_RMS", 0.10),
            max_gain=max_gain,
        )
        seg_rms = rms(asr_samples)
        if config.DEBUG:
            print("[DEBUG] 片段 {:.2f}s RMS {:.4f}".format(duration, seg_rms))

        # 用增益前能量做门控：增益后几乎总能过 ASR_MIN_RMS
        min_raw = float(getattr(config, "ASR_MIN_RAW_RMS", 0.002))
        if raw_rms < min_raw:
            print(
                "[跳过] 原始音量过低 raw_rms={:.5f} < {:.5f}".format(raw_rms, min_raw),
                flush=True,
            )
            return
        min_rms = getattr(config, "ASR_MIN_RMS", 0.008)
        if seg_rms < min_rms:
            print("[跳过] 音量过低 RMS {:.4f} < {:.4f}".format(seg_rms, min_rms), flush=True)
            return

        merge_max = float(getattr(config, "ASR_SHORT_MERGE_MAX_SEC", 0.85))
        gap = float(getattr(config, "ASR_SHORT_MERGE_GAP_SEC", 1.2))
        now = time.time()

        # 若有暂存短句：间隔内则拼接，否则先送出暂存
        if self._pending_short is not None:
            p_samples, p_multi, p_rms, p_dur, p_t0 = self._pending_short
            if (now - p_t0) <= gap:
                asr_samples = np.concatenate([p_samples, asr_samples])
                multi = self._concat_multi(p_multi, multi)
                duration = p_dur + duration
                seg_rms = max(float(p_rms), float(seg_rms))
                self._pending_short = None
                print(
                    "[短句合并] -> {:.2f}s 再送 ASR".format(duration),
                    flush=True,
                )
            else:
                self._flush_pending_short(force=True)

        if duration < merge_max:
            self._pending_short = (asr_samples, multi, seg_rms, duration, now)
            if config.DEBUG:
                print("[DEBUG] 短句暂存 {:.2f}s".format(duration), flush=True)
            return

        self._enqueue_asr(asr_samples, multi, seg_rms, duration)

    def _asr_worker(self):
        while True:
            item = self._asr_queue.get()
            if item is None:
                break
            try:
                if isinstance(item, tuple):
                    asr_samples, multi, seg_rms, duration = item
                    self._process_asr(asr_samples, multi=multi, seg_rms=seg_rms, duration=duration)
                else:
                    self._process_asr(item)
            except Exception as exc:
                print("\n[ASR worker 错误] {}".format(exc), flush=True)

    def _resolve_doa(self, multi, duration):
        doa = None
        if multi is not None:
            doa = self.doa.analyze_segment(multi)
        if doa is None and self.doa.last_azimuth is not None:
            az, conf = self.doa.last_azimuth
            if conf >= float(getattr(config, "SPEAKER_DOA_MIN_CONF", 0.6)):
                doa = (az, conf)
        if config.DEBUG and doa is not None:
            print(
                "[DEBUG] DOA az={:.1f} conf={:.2f} dur={:.2f}s".format(
                    doa[0], doa[1], duration or 0.0
                ),
                flush=True,
            )
        return doa

    def _process_asr(self, asr_samples, multi=None, seg_rms=None, duration=None):
        try:
            text = self._get_asr().transcribe(
                asr_samples, TARGET_SR, already_normalized=True
            )
        except TypeError:
            text = self._get_asr().transcribe(asr_samples, TARGET_SR)
        except Exception as exc:
            print("\n[ASR 错误] {}".format(exc), flush=True)
            return
        if not is_valid_transcript(text):
            print(
                "[跳过] ASR 无有效文字: \"{}\" (dur={:.2f}s rms={:.4f} n={})".format(
                    text,
                    float(duration or (len(asr_samples) / float(TARGET_SR))),
                    float(seg_rms if seg_rms is not None else rms(asr_samples)),
                    len(asr_samples),
                ),
                flush=True,
            )
            return

        doa = self._resolve_doa(multi, duration)
        # 质量门控：短段 / 低 RMS / 噪声转写 → 只 match，不更新质心
        min_embed = float(getattr(config, "SPEAKER_EMBED_MIN_SEC", 0.80))
        min_rms = float(getattr(config, "ASR_MIN_RMS", 0.003))
        dur = duration if duration is not None else (len(asr_samples) / float(TARGET_SR))
        level = seg_rms if seg_rms is not None else rms(asr_samples)
        noise = _is_noise_transcript(text)
        update_profile = not noise and dur >= min_embed and level >= min_rms * 1.5

        try:
            if update_profile:
                speaker = self.tracker.assign(
                    asr_samples, TARGET_SR, update_profile=True, doa=doa
                )
            else:
                speaker = (
                    self.tracker.match(asr_samples, TARGET_SR, doa=doa)
                    or self.tracker.last_label
                    or config.SPEAKER_LABEL_PREFIX + "1"
                )
            merges = self.tracker.pop_merges()
            if merges:
                mapping = {}
                for old, new in merges:
                    mapping[old] = mapping.get(new, new)
                speaker = mapping.get(speaker, speaker)
                self._apply_label_map(merges)
        except Exception as exc:
            if config.DEBUG:
                print("[DEBUG] 说话人识别失败: {}".format(exc))
            speaker = config.SPEAKER_LABEL_PREFIX + "1"
        line = "{}: {}".format(speaker, text)
        with self._output_lock:
            self._log_lines.append(line)
        self.on_utterance(speaker, text)

    def get_transcript_text(self):
        return "\n".join(self._log_lines)

    def get_transcript_lines(self):
        return list(self._log_lines)

    def save_wav(self, path):
        if not self._all_16k_chunks:
            return False
        try:
            import soundfile as sf

            all_data = np.concatenate(self._all_16k_chunks)
            sf.write(str(path), all_data, TARGET_SR, subtype="PCM_16")
            return True
        except Exception as exc:
            print("[wav] 保存失败: {}".format(exc), flush=True)
            return False

    def on_pause(self):
        """暂停采集：丢弃未完成的音频窗，避免恢复后拼接异常。"""
        self.paused = True
        self._audio_buffer = np.array([], dtype=np.float32)

    def on_resume(self):
        """恢复采集：清空半成品缓冲，丢弃暂停前未识别的语音段。"""
        self.paused = False
        self._audio_buffer = np.array([], dtype=np.float32)
        try:
            while not self.vad.empty():
                self.vad.pop()
        except Exception:
            pass

    def _set_vad_min_silence(self, sec):
        try:
            self.vad.config.silero_vad.min_silence_duration = float(sec)
        except Exception:
            pass

    def _update_coldstart_vad(self):
        """开场一段时间内用更短静音切段，随后恢复默认。"""
        if self._session_t0 is None:
            return
        cold_sec = float(getattr(config, "VAD_COLDSTART_SEC", 15.0))
        cold_sil = float(getattr(config, "VAD_COLDSTART_MIN_SILENCE", 0.45))
        normal_sil = float(getattr(config, "MIN_SILENCE_DURATION", 1.0))
        elapsed = time.time() - self._session_t0
        if elapsed < cold_sec:
            if not self._coldstart_active:
                self._set_vad_min_silence(cold_sil)
                self._coldstart_active = True
                print(
                    "[VAD] 开场冷启动 min_silence={:.2f}s ({}s内)".format(
                        cold_sil, cold_sec
                    ),
                    flush=True,
                )
        elif self._coldstart_active:
            self._set_vad_min_silence(normal_sil)
            self._coldstart_active = False
            print(
                "[VAD] 恢复默认 min_silence={:.2f}s".format(normal_sil),
                flush=True,
            )

    def feed_16k(self, chunk, multi=None):
        if chunk is None or len(chunk) == 0 or self.paused:
            return
        chunk = np.asarray(chunk, dtype=np.float32)
        chunk, multi = align_multi_to_mono(chunk, multi)
        self._all_16k_chunks.append(chunk)
        if multi is not None and multi.ndim == 2 and multi.shape[1] > 0:
            self._append_multi_ring(multi)
            self.doa.feed_block(multi)
        self._update_coldstart_vad()
        self._flush_pending_short(force=False)
        self._audio_buffer = np.concatenate([self._audio_buffer, chunk])
        while len(self._audio_buffer) >= self.window_size:
            self.vad.accept_waveform(self._audio_buffer[: self.window_size])
            self._audio_buffer = self._audio_buffer[self.window_size :]
            self._process_vad_queue()

    def flush(self):
        if hasattr(self.vad, "flush"):
            self.vad.flush()
        self._process_vad_queue()
        self._flush_pending_short(force=True)

    def begin_session(self):
        """新一轮录音：清空缓冲，保留已加载模型。"""
        self.paused = False
        self._audio_buffer = np.array([], dtype=np.float32)
        self._multi_ring = None
        self._all_16k_chunks = []
        self._log_lines = []
        self._seg_count = 0
        self._seen_transcripts = set()
        self._dropped_segments = 0
        self._asr_wait_printed = False
        self._pending_short = None
        self._session_t0 = time.time()
        self._coldstart_active = False
        cold_sil = float(getattr(config, "VAD_COLDSTART_MIN_SILENCE", 0.45))
        self._set_vad_min_silence(cold_sil)
        self._coldstart_active = True
        if hasattr(self, "doa") and self.doa is not None:
            self.doa.reset()
        try:
            while not self._asr_queue.empty():
                self._asr_queue.get_nowait()
        except Exception:
            pass
        try:
            while not self.vad.empty():
                self.vad.pop()
            if hasattr(self.vad, "flush"):
                self.vad.flush()
        except Exception:
            pass
        if hasattr(self.tracker, "reset"):
            self.tracker.reset()
        print(
            "[会话] 开始：预滚由 UI 注入；开场 min_silence={:.2f}s".format(cold_sil),
            flush=True,
        )

    def end_session(self, wav_path=None):
        """结束本轮：flush、轻量二次合并回标，可选保存 WAV，然后重置。"""
        self._flush_pending_short(force=True)
        tail = np.array([], dtype=np.float32)
        if len(self._audio_buffer) > 0:
            tail = normalize(self._audio_buffer)
        if len(tail) >= int(0.5 * TARGET_SR):
            multi = self._slice_multi_for_segment(len(tail))
            self._handle_segment(tail, multi=multi)
        self.flush()
        for _ in range(30):
            if self._asr_queue.empty():
                break
            time.sleep(0.1)
        try:
            merges = self.tracker.finalize_merges()
            self._apply_label_map(merges)
        except Exception as exc:
            if config.DEBUG:
                print("[DEBUG] finalize_merges 失败: {}".format(exc), flush=True)
        saved = self.save_wav(wav_path) if wav_path else False
        self.begin_session()
        return saved

    def fallback_webrtc(self, samples):
        """Silero 无片段时，用 WebRTC VAD 切分整段录音"""
        if not webrtc_available():
            if config.DEBUG:
                print("[DEBUG] webrtcvad 未安装，跳过备用切分")
                print("        安装: python3 -m pip install --user webrtcvad")
            return
        if config.DEBUG:
            print("[DEBUG] 启用 WebRTC VAD 备用切分")
        for s, e in webrtc_segments(samples, TARGET_SR):
            self._handle_segment(samples[s:e])

    def save_log(self, path=None):
        content = "\n".join(self._log_lines)
        if self._log_lines:
            content += "\n"
        candidates = [
            Path(path) if path else config.OUTPUT_LOG,
            Path("/tmp/meeting_transcript.txt"),
            Path.home() / "meeting_transcript.txt",
        ]
        for target in candidates:
            try:
                target.parent.mkdir(parents=True, exist_ok=True)
                with open(target, "w", encoding="utf-8") as f:
                    f.write(content)
                return target
            except (PermissionError, OSError):
                continue
        print("[警告] 无法写入日志，请检查 output 目录权限")
        return None

    def run_from_microphone(self):
        from audio_capture import create_microphone_reader

        mic = create_microphone_reader()

        print("=" * 50)
        print("实时会议识别已启动  [{}]".format(TRACKER_VERSION))
        print("说话时音量条应有 ####，静音时为 ----")
        if not self._asr_ready.is_set():
            print("（ASR 仍在后台加载，加载完即可转文字）")
        print("若无反应，请运行: python3 src/mic_test.py")
        print("按 Ctrl+C 结束")
        print("=" * 50)

        try:
            while True:
                try:
                    chunk, level, multi = unpack_read_16k(mic.read_16k())
                except OSError as exc:
                    print("\n[警告] 麦克风读取失败: {}".format(exc))
                    print("尝试重新打开麦克风…")
                    try:
                        mic._open_stream()
                    except Exception as reopen_exc:
                        print("[错误] 无法恢复: {}".format(reopen_exc))
                        break
                    continue
                print_level_meter(level)
                self.feed_16k(chunk, multi=multi)
        except KeyboardInterrupt:
            print("\n\n结束录音…")
            tail, tail_multi = unpack_flush_16k(mic.flush_16k())
            if len(tail) >= int(0.5 * TARGET_SR):
                self._handle_segment(normalize(tail), multi=tail_multi)

        finally:
            try:
                mic.close()
            except Exception:
                pass
            print("\n最大音量 RMS: {:.4f}  录音时长: {:.1f}s".format(
                mic.max_rms, mic.duration_sec
            ))

            if self._log_lines:
                p = self.save_log()
                if p:
                    print("记录已保存: {}".format(p))
                summary = self.tracker.summary()
                if summary:
                    print("说话人统计: {}".format(
                        ", ".join(
                            "{}({}句)".format(s["label"], s["count"])
                            for s in summary
                        )
                    ))
            elif mic.max_rms < 0.005:
                print("\n[错误] 麦克风几乎无信号（静音）")
                print("  1. 运行: python3 src/list_devices.py")
                print("  2. 修改 config.py → INPUT_DEVICE_INDEX = 正确编号")
                print("  3. 若双声道麦，试 INPUT_CHANNELS = 2")
                print("  4. 运行: python3 src/mic_test.py 验证")
            elif self._seg_count == 0:
                print("\n有音量但无识别结果，请运行: python3 src/test_asr.py")
            else:
                print("\n检测到语音段但 ASR 无文字，请运行: python3 src/test_asr.py")


def main():
    RealtimeMeetingEngine().run_from_microphone()


if __name__ == "__main__":
    main()
