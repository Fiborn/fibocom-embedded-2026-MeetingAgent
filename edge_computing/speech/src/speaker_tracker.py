# -*- coding: utf-8 -*-
"""
在线说话人分离 — competition-v2 + DOA + 频率增强

声纹质心为主；基频分布 / 共振峰 / 频谱质心 / 阵列方位为辅。
目标：同人稳定，尽量不串人。
"""
import math

import numpy as np
import sherpa_onnx

import config

TRACKER_VERSION = "competition-v2-doa-freq"


def _empty_feats():
    return {
        "pitch": None,
        "pitch_p25": None,
        "pitch_p75": None,
        "pitch_iqr": None,
        "f1": None,
        "f2": None,
        "centroid": None,
    }


def _norm_feats(feats):
    if not feats:
        return _empty_feats()
    out = _empty_feats()
    out.update(feats)
    return out


class _SpeakerProfile:
    __slots__ = (
        "label",
        "centroid",
        "count",
        "pitch",
        "pitch_iqr",
        "f1",
        "f2",
        "centroid_hz",
        "azimuth",
        "azimuth_conf",
        "enrolled",
    )

    def __init__(
        self,
        label,
        centroid,
        feats=None,
        azimuth=None,
        azimuth_conf=0.0,
        enrolled=False,
    ):
        feats = _norm_feats(feats)
        self.label = label
        self.centroid = centroid.astype(np.float32, copy=False)
        self.count = 1
        self.pitch = feats.get("pitch")
        self.pitch_iqr = feats.get("pitch_iqr")
        self.f1 = feats.get("f1")
        self.f2 = feats.get("f2")
        self.centroid_hz = feats.get("centroid")
        self.azimuth = azimuth
        self.azimuth_conf = float(azimuth_conf or 0.0)
        self.enrolled = bool(enrolled)


class SpeakerTracker:
    def __init__(self):
        if not config.EMBEDDING_MODEL.exists():
            raise FileNotFoundError(
                "嵌入模型缺失，请运行: python3 scripts/download_model.py"
            )
        self.extractor = sherpa_onnx.SpeakerEmbeddingExtractor(
            config=sherpa_onnx.SpeakerEmbeddingExtractorConfig(
                model=str(config.EMBEDDING_MODEL),
                num_threads=getattr(config, "SPEAKER_NUM_THREADS", 2),
                debug=False,
                provider="cpu",
            )
        )
        self.speakers = []
        self._last_label = None
        self._self_sims = []
        self._pending_streak = 0
        self._pending_merges = []
        max_sp = getattr(config, "SPEAKER_MAX", 0)
        self._unlimited = not max_sp or max_sp <= 0
        self._max_speakers = int(max_sp) if not self._unlimited else 0
        self._use_doa = bool(getattr(config, "SPEAKER_USE_DOA", True))
        self._anon_seq = 0

        limit_text = "不限" if self._unlimited else str(self._max_speakers)
        print("[说话人模块] {}  人数上限: {}  DOA: {}  频率增强: 开".format(
            TRACKER_VERSION, limit_text, "开" if self._use_doa else "关"
        ))
        self.reload_enrolled()

    @property
    def last_label(self):
        return self._last_label

    def pop_merges(self):
        merges = list(self._pending_merges)
        self._pending_merges = []
        return merges

    @staticmethod
    def _l2_norm(vec):
        vec = np.asarray(vec, dtype=np.float32)
        n = float(np.linalg.norm(vec))
        if n < 1e-8:
            return vec
        return vec / n

    @staticmethod
    def _cosine(a, b):
        a = SpeakerTracker._l2_norm(a)
        b = SpeakerTracker._l2_norm(b)
        return float(np.dot(a, b))

    @staticmethod
    def _circular_diff(a, b):
        """最短角差，范围 [0, 180]。"""
        if a is None or b is None:
            return None
        d = abs(float(a) - float(b)) % 360.0
        return d if d <= 180.0 else 360.0 - d

    @staticmethod
    def _circular_ema(old, new, alpha):
        if old is None:
            return float(new)
        old_r = math.radians(float(old))
        new_r = math.radians(float(new))
        x = (1.0 - alpha) * math.cos(old_r) + alpha * math.cos(new_r)
        y = (1.0 - alpha) * math.sin(old_r) + alpha * math.sin(new_r)
        return math.degrees(math.atan2(y, x)) % 360.0

    @staticmethod
    def _ema(old, new, alpha):
        if new is None:
            return old
        if old is None:
            return float(new)
        return (1.0 - alpha) * float(old) + alpha * float(new)

    def _extract(self, samples, sample_rate=16000):
        from audio_utils import prepare_for_embedding

        samples = prepare_for_embedding(samples)
        if len(samples) < int(
            getattr(config, "SPEAKER_EMBED_MIN_SEC", 0.65) * sample_rate
        ):
            return None

        samples = np.ascontiguousarray(samples, dtype=np.float32)
        stream = self.extractor.create_stream()
        stream.accept_waveform(sample_rate, samples)
        if hasattr(stream, "input_finished"):
            stream.input_finished()
        if not self.extractor.is_ready(stream):
            return None
        return self._l2_norm(np.array(self.extractor.compute(stream)))

    def _estimate_voice_feats(self, samples, sample_rate=16000):
        from audio_utils import extract_voice_freq_features

        return _norm_feats(extract_voice_freq_features(samples, sample_rate))

    def _parse_doa(self, doa):
        if not self._use_doa or doa is None:
            return None, 0.0
        if isinstance(doa, (tuple, list)) and len(doa) >= 2:
            return float(doa[0]), float(doa[1])
        return None, 0.0

    def _doa_usable(self, az, conf):
        if az is None:
            return False
        min_conf = float(getattr(config, "SPEAKER_DOA_MIN_CONF", 0.6))
        return conf >= min_conf

    def _doa_ok(self, az, conf, profile):
        """高置信且方位差过大 → 否决同人。"""
        if not self._doa_usable(az, conf):
            return True
        if profile.azimuth is None or profile.azimuth_conf < getattr(
            config, "SPEAKER_DOA_MIN_CONF", 0.6
        ):
            return True
        diff = self._circular_diff(az, profile.azimuth)
        if diff is None:
            return True
        limit = float(getattr(config, "SPEAKER_DOA_DIFF_DEG", 50.0))
        return diff < limit

    def _doa_bonus(self, az, conf, profile):
        """方位兼容加分/减分，叠加到 embedding 余弦分。"""
        if not self._doa_usable(az, conf) or profile.azimuth is None:
            return 0.0
        if profile.azimuth_conf < getattr(config, "SPEAKER_DOA_MIN_CONF", 0.6):
            return 0.0
        diff = self._circular_diff(az, profile.azimuth)
        if diff is None:
            return 0.0
        weight = float(getattr(config, "SPEAKER_DOA_WEIGHT", 0.25))
        sim = 1.0 - (diff / 180.0)
        return weight * (2.0 * sim - 1.0)

    def _pitch_limit(self, profile):
        base = float(getattr(config, "SPEAKER_PITCH_DIFF_HZ", 55))
        iqr = float(profile.pitch_iqr or 0.0)
        # 允许随说话人自身基频波动放宽
        return max(base, 0.6 * iqr + 25.0)

    def _pitch_diff_ok(self, feats, profile):
        feats = _norm_feats(feats)
        pitch = feats.get("pitch")
        if pitch is None or profile.pitch is None:
            return True
        return abs(pitch - profile.pitch) < self._pitch_limit(profile)

    def _formant_diff(self, feats, profile):
        feats = _norm_feats(feats)
        if feats.get("f1") is None or profile.f1 is None:
            return None
        d1 = abs(float(feats["f1"]) - float(profile.f1))
        if feats.get("f2") is not None and profile.f2 is not None:
            d2 = abs(float(feats["f2"]) - float(profile.f2))
            return 0.55 * d1 + 0.45 * d2
        return d1

    def _freq_conflict(self, feats, profile, emb_only, same_th):
        """声纹不够像时，频率明显冲突 → 辅助拆人。"""
        # 仅在声纹「非常像」时忽略频率冲突，避免异人被高粘贴吞掉
        if emb_only >= same_th + 0.04:
            return False
        feats = _norm_feats(feats)
        pitch = feats.get("pitch")
        pitch_hit = (
            pitch is not None
            and profile.pitch is not None
            and abs(pitch - profile.pitch) >= self._pitch_limit(profile)
        )
        fdiff = self._formant_diff(feats, profile)
        formant_lim = float(getattr(config, "SPEAKER_FORMANT_DIFF_HZ", 350.0))
        formant_hit = fdiff is not None and fdiff >= formant_lim
        # 基频+共振峰同时冲突
        if pitch_hit and formant_hit:
            return True
        # 基频明显不同且声纹未达同人阈值
        if pitch_hit and emb_only < same_th:
            return True
        # 共振峰差较大
        if formant_hit and fdiff >= formant_lim * 1.2 and emb_only < same_th:
            return True
        return False

    def _freq_bonus(self, feats, profile):
        """频率兼容加分/减分（权重宜小，避免压过声纹）。"""
        feats = _norm_feats(feats)
        weight = float(getattr(config, "SPEAKER_FREQ_WEIGHT", 0.10))
        if weight <= 0:
            return 0.0
        parts = []

        pitch = feats.get("pitch")
        if pitch is not None and profile.pitch is not None:
            lim = self._pitch_limit(profile)
            rel = min(1.0, abs(pitch - profile.pitch) / max(lim, 1.0))
            parts.append(1.0 - 2.0 * rel)  # 近 → +1，远 → -1

        fdiff = self._formant_diff(feats, profile)
        if fdiff is not None:
            lim = float(getattr(config, "SPEAKER_FORMANT_DIFF_HZ", 350.0))
            rel = min(1.0, fdiff / max(lim, 1.0))
            parts.append(1.0 - 2.0 * rel)

        cen = feats.get("centroid")
        if cen is not None and profile.centroid_hz is not None:
            lim = float(getattr(config, "SPEAKER_CENTROID_DIFF_HZ", 800.0))
            rel = min(1.0, abs(cen - profile.centroid_hz) / max(lim, 1.0))
            parts.append(1.0 - 2.0 * rel)

        if not parts:
            return 0.0
        return weight * float(np.mean(parts))

    def _score(self, emb, profile, az=None, conf=0.0, feats=None):
        base = self._cosine(emb, profile.centroid)
        return base + self._doa_bonus(az, conf, profile) + self._freq_bonus(feats, profile)

    def _same_threshold(self):
        base = getattr(config, "SPEAKER_SAME_THRESHOLD", 0.56)
        need = getattr(config, "SPEAKER_BOOTSTRAP_SEGMENTS", 4)
        if len(self._self_sims) < need:
            return base
        arr = np.array(self._self_sims, dtype=np.float32)
        adaptive = float(np.percentile(arr, 20))
        cap = getattr(config, "SPEAKER_SAME_THRESHOLD_CAP", 0.48)
        return max(base, min(adaptive, cap))

    def _new_threshold(self):
        return getattr(config, "SPEAKER_NEW_THRESHOLD", 0.24)

    def _record_self_sim(self, sim):
        self._self_sims.append(float(sim))
        keep = getattr(config, "SPEAKER_SELF_SIM_HISTORY", 30)
        if len(self._self_sims) > keep:
            self._self_sims = self._self_sims[-keep:]

    def _update_freq_profile(self, sp, feats, strong=True):
        feats = _norm_feats(feats)
        alpha = 0.20 if strong else 0.10
        sp.pitch = self._ema(sp.pitch, feats.get("pitch"), alpha)
        sp.pitch_iqr = self._ema(sp.pitch_iqr, feats.get("pitch_iqr"), alpha)
        sp.f1 = self._ema(sp.f1, feats.get("f1"), alpha)
        sp.f2 = self._ema(sp.f2, feats.get("f2"), alpha)
        sp.centroid_hz = self._ema(sp.centroid_hz, feats.get("centroid"), alpha)

    def reload_enrolled(self):
        """从磁盘加载预注册声纹，写入 speakers（会清除未注册匿名档案）。"""
        keep_anon = []
        # 仅在“追加刷新”场景保留匿名；默认由 reset/ begin_session 先清空
        enrolled_labels = set()
        try:
            from speaker_enroll import EnrollStore

            profiles = EnrollStore().load_all()
        except Exception as exc:
            print("[注册] 加载失败: {}".format(exc))
            profiles = []

        # 去掉旧 enrolled，保留匿名
        for sp in self.speakers:
            if getattr(sp, "enrolled", False):
                continue
            keep_anon.append(sp)
        self.speakers = []
        for name, emb, feats in profiles:
            self.speakers.append(
                _SpeakerProfile(
                    name,
                    emb.astype(np.float32, copy=False),
                    feats=feats,
                    enrolled=True,
                )
            )
            # 注册档案给较高初始 count，降低被合并风险
            self.speakers[-1].count = max(8, int(getattr(config, "SPEAKER_ENROLL_COUNT", 12)))
            enrolled_labels.add(name)
        # 匿名标签避开已占用姓名
        for sp in keep_anon:
            if sp.label in enrolled_labels:
                continue
            self.speakers.append(sp)
        if profiles:
            print(
                "[注册] 已加载 {} 人: {}".format(
                    len(profiles),
                    ", ".join(n for n, _, _ in profiles),
                )
            )
        return len(profiles)

    def _enroll_match_threshold(self):
        return float(getattr(config, "SPEAKER_ENROLL_MATCH_THRESHOLD", 0.40))

    def _match_enrolled(self, emb, feats, az=None, conf=0.0):
        """优先匹配预注册档案；命中则返回 (index, score)。"""
        enrolled = [
            (i, sp)
            for i, sp in enumerate(self.speakers)
            if getattr(sp, "enrolled", False)
        ]
        if not enrolled:
            return None
        th = self._enroll_match_threshold()
        floor = float(getattr(config, "SPEAKER_ENROLL_MATCH_FLOOR", max(0.30, th - 0.08)))
        margin_need = float(getattr(config, "SPEAKER_ENROLL_MATCH_MARGIN", 0.08))
        ranked = []
        for i, sp in enrolled:
            emb_only = self._cosine(emb, sp.centroid)
            score = self._score(emb, sp, az=az, conf=conf, feats=feats)
            ranked.append((i, sp, emb_only, score))
        ranked.sort(key=lambda t: t[2], reverse=True)
        best_i, sp, best_emb, best_s = ranked[0]
        second_emb = ranked[1][2] if len(ranked) > 1 else -1.0
        gap = best_emb - second_emb if second_emb >= 0 else 1.0
        # 绝对阈值，或「够底线且明显优于第二名」
        accept = best_emb >= th or (best_emb >= floor and gap >= margin_need)
        if not accept:
            if getattr(config, "DEBUG", False) or best_emb >= floor:
                print(
                    "[注册匹配] 未命中 best={}={:.3f} second={:.3f} th={:.2f}".format(
                        sp.label, best_emb, second_emb, th
                    ),
                    flush=True,
                )
            return None
        # 注册匹配：声纹为主；频率严重冲突则放弃（避免认错）
        if self._freq_conflict(feats, sp, best_emb, th) and best_emb < th + 0.06:
            print(
                "[注册匹配] 频率冲突放弃 {} emb={:.3f}".format(sp.label, best_emb),
                flush=True,
            )
            return None
        return best_i, best_s

    def _update_centroid(self, idx, emb, feats=None, strong=True, az=None, conf=0.0):
        sp = self.speakers[idx]
        alpha = getattr(config, "SPEAKER_EMA_ALPHA", 0.15)
        if getattr(sp, "enrolled", False):
            # 注册档案只做极弱适应，避免被会场噪声带偏
            alpha = min(alpha, 0.04) if strong else 0.0
        if not strong:
            alpha *= 0.4
        sim = self._cosine(emb, sp.centroid)
        if sim < 0.45:
            alpha *= 0.3
        if alpha > 0:
            merged = (1.0 - alpha) * sp.centroid + alpha * emb
            sp.centroid = self._l2_norm(merged)
        sp.count += 1
        if not getattr(sp, "enrolled", False) or strong:
            self._update_freq_profile(sp, feats, strong=strong and not sp.enrolled)
        if self._doa_usable(az, conf):
            az_alpha = float(getattr(config, "SPEAKER_DOA_PROFILE_ALPHA", 0.2))
            if not strong:
                az_alpha *= 0.5
            sp.azimuth = self._circular_ema(sp.azimuth, az, az_alpha)
            sp.azimuth_conf = max(sp.azimuth_conf, float(conf))
        self._last_label = sp.label
        self._pending_streak = 0
        return sp.label

    def _next_anon_label(self):
        taken = {sp.label for sp in self.speakers}
        prefix = config.SPEAKER_LABEL_PREFIX
        while True:
            self._anon_seq += 1
            label = "{}{}".format(prefix, self._anon_seq)
            if label not in taken:
                return label

    def _create_profile(self, emb, feats=None, az=None, conf=0.0):
        feats = _norm_feats(feats)
        anon_n = sum(1 for sp in self.speakers if not getattr(sp, "enrolled", False))
        if (
            not self._unlimited
            and self._max_speakers
            and anon_n >= self._max_speakers
        ):
            # 只在匿名档案里找最像的更新，避免冲掉注册姓名
            ranked = [
                (i, self._score(emb, sp, az=az, conf=conf, feats=feats))
                for i, sp in enumerate(self.speakers)
                if not getattr(sp, "enrolled", False)
            ]
            if ranked:
                ranked.sort(key=lambda x: x[1], reverse=True)
                return self._update_centroid(
                    ranked[0][0], emb, feats=feats, strong=False, az=az, conf=conf
                )

        label = self._next_anon_label()
        use_az = az if self._doa_usable(az, conf) else None
        self.speakers.append(
            _SpeakerProfile(
                label,
                emb.copy(),
                feats=feats,
                azimuth=use_az,
                azimuth_conf=float(conf) if use_az is not None else 0.0,
                enrolled=False,
            )
        )
        self._last_label = label
        self._pending_streak = 0
        if config.DEBUG:
            print(
                "[DEBUG] 新建 {} pitch={} f1={} f2={} az={}".format(
                    label,
                    None if feats.get("pitch") is None else round(feats["pitch"], 1),
                    None if feats.get("f1") is None else round(feats["f1"], 1),
                    None if feats.get("f2") is None else round(feats["f2"], 1),
                    None if use_az is None else round(use_az, 1),
                )
            )
        return label

    def _rank(self, emb, az=None, conf=0.0, feats=None):
        scored = [
            (i, self._score(emb, sp, az=az, conf=conf, feats=feats))
            for i, sp in enumerate(self.speakers)
        ]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored

    def _find_label_index(self, label):
        if not label:
            return None
        for i, sp in enumerate(self.speakers):
            if sp.label == label:
                return i
        return None

    def _emb_floor(self):
        return float(getattr(config, "SPEAKER_CONFIDENT_EMB_MIN", 0.50))

    def _sticky_min_sim(self):
        return float(getattr(config, "SPEAKER_STICKY_MIN_SIM", 0.40))

    def _rematch_min(self):
        return float(getattr(config, "SPEAKER_REMATCH_MIN", 0.42))

    def _should_stick_to_last(self, emb, feats, az, conf):
        """模糊区间是否允许沿用 last_label。"""
        idx = self._find_label_index(self._last_label)
        if idx is None:
            return False
        sp = self.speakers[idx]
        sim = self._cosine(emb, sp.centroid)
        if sim < self._sticky_min_sim():
            return False
        same_th = self._same_threshold()
        if not self._doa_ok(az, conf, sp) and sim < same_th:
            return False
        if self._freq_conflict(feats, sp, sim, same_th):
            return False
        if not self._pitch_diff_ok(feats, sp) and sim < same_th:
            return False
        return True

    def _soft_rematch(self, best_i, emb, feats, az, conf, emb_only, allow_conflict=False):
        """与已有最佳档案足够像 → 弱更新，不新建。"""
        if emb_only < self._rematch_min():
            return None
        sp = self.speakers[best_i]
        same_th = self._same_threshold()
        if not allow_conflict:
            if not self._doa_ok(az, conf, sp) and emb_only < same_th:
                return None
            if self._freq_conflict(feats, sp, emb_only, same_th):
                return None
        return self._update_centroid(
            best_i, emb, feats=feats, strong=False, az=az, conf=conf
        )

    def _assign_when_one(self, emb, feats, sim, same_th, new_th, need_pending, az, conf):
        high = max(same_th, getattr(config, "SPEAKER_FIRST_HIGH", 0.65))
        sp0 = self.speakers[0]
        doa_ok = self._doa_ok(az, conf, sp0)
        min_conf = float(getattr(config, "SPEAKER_DOA_MIN_CONF", 0.6))
        doa_conflict = (
            self._doa_usable(az, conf)
            and sp0.azimuth is not None
            and sp0.azimuth_conf >= min_conf
            and not doa_ok
            and sim < same_th
        )
        freq_conflict = self._freq_conflict(feats, sp0, sim, same_th)

        if sim >= high and sim >= self._emb_floor() and self._pitch_diff_ok(feats, sp0):
            if doa_ok or sim >= high + 0.05:
                label = self._update_centroid(
                    0, emb, feats=feats, strong=True, az=az, conf=conf
                )
                self._record_self_sim(sim)
                return label

        # 方位/频率冲突或声纹偏低 → 优先拆出第二人（防异人同标）
        if (
            sim < new_th
            or doa_conflict
            or (freq_conflict and sim < same_th)
        ):
            return self._create_profile(emb, feats=feats, az=az, conf=conf)

        # 无冲突时才允许软更新/粘贴
        if sim >= self._sticky_min_sim():
            rematch = self._soft_rematch(0, emb, feats, az, conf, sim)
            if rematch is not None:
                return rematch
            self._last_label = sp0.label
            return sp0.label

        self._pending_streak += 1
        if self._pending_streak >= need_pending:
            return self._create_profile(emb, feats=feats, az=az, conf=conf)
        return self._last_label or sp0.label

    def _try_merge(self):
        th = getattr(config, "SPEAKER_MERGE_THRESHOLD", 0.66)
        max_count = getattr(config, "SPEAKER_MERGE_MAX_COUNT", 8)
        doa_diff = float(getattr(config, "SPEAKER_DOA_DIFF_DEG", 50.0))
        min_conf = float(getattr(config, "SPEAKER_DOA_MIN_CONF", 0.6))
        formant_lim = float(getattr(config, "SPEAKER_FORMANT_DIFF_HZ", 350.0))
        merged = True
        while merged:
            merged = False
            for i in range(len(self.speakers)):
                if merged:
                    break
                for j in range(i + 1, len(self.speakers)):
                    si, sj = self.speakers[i], self.speakers[j]
                    # 预注册档案禁止被合并/吞掉
                    if getattr(si, "enrolled", False) or getattr(sj, "enrolled", False):
                        continue
                    if min(si.count, sj.count) > max_count:
                        continue
                    if self._cosine(si.centroid, sj.centroid) < th:
                        continue
                    if (
                        si.pitch is not None
                        and sj.pitch is not None
                        and abs(si.pitch - sj.pitch)
                        >= max(self._pitch_limit(si), self._pitch_limit(sj))
                    ):
                        continue
                    if (
                        si.f1 is not None
                        and sj.f1 is not None
                        and abs(si.f1 - sj.f1) >= formant_lim * 1.2
                    ):
                        continue
                    if (
                        si.azimuth is not None
                        and sj.azimuth is not None
                        and si.azimuth_conf >= min_conf
                        and sj.azimuth_conf >= min_conf
                    ):
                        adiff = self._circular_diff(si.azimuth, sj.azimuth)
                        if adiff is not None and adiff >= doa_diff:
                            continue
                    wi = float(si.count)
                    wj = float(sj.count)
                    old_label = sj.label
                    new_label = si.label
                    si.centroid = self._l2_norm(
                        (si.centroid * wi + sj.centroid * wj) / (wi + wj)
                    )
                    si.count += sj.count
                    for attr in ("pitch", "pitch_iqr", "f1", "f2", "centroid_hz"):
                        va, vb = getattr(si, attr), getattr(sj, attr)
                        if va is None:
                            setattr(si, attr, vb)
                        elif vb is not None:
                            setattr(si, attr, (va * wi + vb * wj) / (wi + wj))
                    if si.azimuth is None:
                        si.azimuth = sj.azimuth
                        si.azimuth_conf = sj.azimuth_conf
                    elif sj.azimuth is not None:
                        si.azimuth = self._circular_ema(
                            si.azimuth, sj.azimuth, wj / (wi + wj)
                        )
                        si.azimuth_conf = max(si.azimuth_conf, sj.azimuth_conf)
                    if self._last_label == old_label:
                        self._last_label = new_label
                    self._pending_merges.append((old_label, new_label))
                    del self.speakers[j]
                    if config.DEBUG:
                        print("[DEBUG] 合并 {} -> {}".format(old_label, new_label))
                    merged = True
                    break

    def finalize_merges(self):
        """会话结束前再跑一轮合并，返回本次产生的 (old, new) 列表。"""
        self._try_merge()
        return self.pop_merges()

    def _resolve_merged_label(self, label):
        """若 label 刚被合并，返回存活标签。"""
        if not label or not self._pending_merges:
            return label
        mapping = {old: new for old, new in self._pending_merges}
        for _ in range(8):
            nxt = mapping.get(label)
            if not nxt or nxt == label:
                break
            label = nxt
        return label

    def assign(self, samples, sample_rate=16000, update_profile=True, doa=None):
        az, conf = self._parse_doa(doa)
        emb = self._extract(samples, sample_rate)
        feats = self._estimate_voice_feats(samples, sample_rate)
        if emb is None:
            return self._last_label or (config.SPEAKER_LABEL_PREFIX + "1")

        if not update_profile:
            return self.match(samples, sample_rate, doa=doa) or self._last_label

        # 预注册优先：命中则直接用真名
        hit = self._match_enrolled(emb, feats, az=az, conf=conf)
        if hit is not None:
            idx, _ = hit
            label = self._update_centroid(
                idx, emb, feats=feats, strong=True, az=az, conf=conf
            )
            self._record_self_sim(self._cosine(emb, self.speakers[idx].centroid))
            return label

        if not self.speakers:
            return self._create_profile(emb, feats=feats, az=az, conf=conf)

        ranked = self._rank(emb, az, conf, feats)
        best_i, best_s = ranked[0]
        best_sp = self.speakers[best_i]
        emb_only = self._cosine(emb, best_sp.centroid)
        second_s = ranked[1][1] if len(ranked) > 1 else -1.0
        margin = best_s - second_s if second_s >= 0 else 1.0
        same_th = self._same_threshold()
        new_th = self._new_threshold()
        need_pending = getattr(config, "SPEAKER_PENDING_COUNT", 4)
        need_margin = getattr(config, "SPEAKER_MARGIN", 0.06)
        pitch_ok = self._pitch_diff_ok(feats, best_sp)
        doa_ok = self._doa_ok(az, conf, best_sp)

        if config.DEBUG:
            tops = [(self.speakers[i].label, round(s, 3)) for i, s in ranked[:3]]
            print(
                "[DEBUG] top={} pitch={} f1={} f2={} az={} same={:.2f} margin={:.3f}".format(
                    tops,
                    None if feats.get("pitch") is None else round(feats["pitch"], 1),
                    None if feats.get("f1") is None else round(feats["f1"], 1),
                    None if feats.get("f2") is None else round(feats["f2"], 1),
                    None if az is None else "{:.0f}@{:.2f}".format(az, conf),
                    same_th,
                    margin,
                )
            )

        if len(self.speakers) == 1:
            label = self._assign_when_one(
                emb, feats, emb_only, same_th, new_th, need_pending, az, conf
            )
            self._try_merge()
            return self._resolve_merged_label(label)

        emb_floor = self._emb_floor()
        confident = (
            best_s >= same_th
            and emb_only >= emb_floor
            and margin >= need_margin
            and pitch_ok
        )
        if confident and (doa_ok or emb_only >= same_th + 0.05):
            label = self._update_centroid(
                best_i, emb, feats=feats, strong=True, az=az, conf=conf
            )
            self._record_self_sim(emb_only)
            self._try_merge()
            return self._resolve_merged_label(label)

        min_conf = float(getattr(config, "SPEAKER_DOA_MIN_CONF", 0.6))
        doa_conflict = (
            self._doa_usable(az, conf)
            and best_sp.azimuth is not None
            and best_sp.azimuth_conf >= min_conf
            and not doa_ok
            and emb_only < same_th
        )
        freq_conflict = self._freq_conflict(feats, best_sp, emb_only, same_th)
        # 方位冲突或频率冲突时，只要声纹未达同人阈值就拆人
        force_new = best_s < new_th or (
            (doa_conflict or freq_conflict) and emb_only < same_th
        )
        if force_new:
            label = self._create_profile(emb, feats=feats, az=az, conf=conf)
            self._try_merge()
            return self._resolve_merged_label(label)

        rematch = self._soft_rematch(best_i, emb, feats, az, conf, emb_only)
        if rematch is not None:
            self._try_merge()
            return self._resolve_merged_label(rematch)

        self._pending_streak += 1
        if self._pending_streak >= need_pending and emb_only < self._rematch_min():
            label = self._create_profile(emb, feats=feats, az=az, conf=conf)
            self._try_merge()
            return self._resolve_merged_label(label)

        if self._should_stick_to_last(emb, feats, az, conf):
            return self._last_label
        # 与最佳档案也不够像时，新建而不是硬贴 best
        if emb_only < self._sticky_min_sim():
            label = self._create_profile(emb, feats=feats, az=az, conf=conf)
            self._try_merge()
            return self._resolve_merged_label(label)
        return best_sp.label

    def match(self, samples, sample_rate=16000, doa=None):
        if not self.speakers:
            return None
        az, conf = self._parse_doa(doa)
        emb = self._extract(samples, sample_rate)
        feats = self._estimate_voice_feats(samples, sample_rate)
        if emb is None:
            return self._last_label
        hit = self._match_enrolled(emb, feats, az=az, conf=conf)
        if hit is not None:
            return self.speakers[hit[0]].label
        ranked = self._rank(emb, az, conf, feats)
        best_i, best_s = ranked[0]
        best_sp = self.speakers[best_i]
        emb_only = self._cosine(emb, best_sp.centroid)
        second_s = ranked[1][1] if len(ranked) > 1 else -1.0
        margin = best_s - second_s if second_s >= 0 else 1.0
        same_th = self._same_threshold()
        need_margin = getattr(config, "SPEAKER_MARGIN", 0.06)
        if (
            best_s >= same_th
            and emb_only >= self._emb_floor()
            and margin >= need_margin
            and self._pitch_diff_ok(feats, best_sp)
        ):
            return best_sp.label
        if emb_only >= self._rematch_min():
            return best_sp.label
        if self._should_stick_to_last(emb, feats, az, conf):
            return self._last_label
        return best_sp.label if emb_only >= self._sticky_min_sim() else self._last_label

    def reset(self):
        self.speakers = []
        self._last_label = None
        self._self_sims = []
        self._pending_streak = 0
        self._pending_merges = []
        self._anon_seq = 0
        self.reload_enrolled()

    def summary(self):
        rows = []
        for sp in self.speakers:
            rows.append({
                "label": sp.label,
                "count": sp.count,
                "pitch": None if sp.pitch is None else round(sp.pitch, 1),
                "f1": None if sp.f1 is None else round(sp.f1, 1),
                "f2": None if sp.f2 is None else round(sp.f2, 1),
                "azimuth": None if sp.azimuth is None else round(sp.azimuth, 1),
            })
        return rows
