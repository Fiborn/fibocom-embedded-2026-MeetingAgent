# -*- coding: utf-8 -*-
"""说话人声纹注册：会前建档（embedding + 频率特征），开会时优先匹配。"""
from __future__ import annotations

import json
import re
import time
from pathlib import Path

import numpy as np

import config

_NAME_RE = re.compile(r"^[\u4e00-\u9fffA-Za-z0-9_\-]{1,20}$")


def enroll_dir() -> Path:
    root = Path(getattr(config, "ENROLL_DIR", config.ROOT / "data" / "enrolled"))
    root.mkdir(parents=True, exist_ok=True)
    return root


def sanitize_name(name: str) -> str:
    name = (name or "").strip().replace(" ", "")
    if not name:
        raise ValueError("姓名不能为空")
    if not _NAME_RE.match(name):
        raise ValueError("姓名仅支持中文/英文/数字/下划线/短横线，最长 20 字")
    return name


def _safe_stem(name: str) -> str:
    # 文件名安全：中文保留，其余非法字符替换
    stem = re.sub(r"[^\u4e00-\u9fffA-Za-z0-9_\-]", "_", name)
    return stem[:40] or "speaker"


class EnrollStore:
    def __init__(self, directory=None):
        self.dir = Path(directory) if directory else enroll_dir()
        self.dir.mkdir(parents=True, exist_ok=True)

    def list_profiles(self):
        items = []
        for p in sorted(self.dir.glob("*.json")):
            try:
                data = json.loads(p.read_text(encoding="utf-8"))
            except Exception:
                continue
            items.append(
                {
                    "name": data.get("name") or p.stem,
                    "created_at": data.get("created_at"),
                    "updated_at": data.get("updated_at"),
                    "duration_sec": data.get("duration_sec"),
                    "pitch": data.get("feats", {}).get("pitch"),
                    "f1": data.get("feats", {}).get("f1"),
                    "f2": data.get("feats", {}).get("f2"),
                    "segments": data.get("segments", 1),
                }
            )
        return items

    def path_for(self, name: str) -> Path:
        return self.dir / "{}.json".format(_safe_stem(sanitize_name(name)))

    def delete(self, name: str) -> bool:
        path = self.path_for(name)
        if path.exists():
            path.unlink()
            return True
        # 兼容旧文件名
        for p in self.dir.glob("*.json"):
            try:
                data = json.loads(p.read_text(encoding="utf-8"))
            except Exception:
                continue
            if data.get("name") == name:
                p.unlink()
                return True
        return False

    def load_all(self):
        """返回 [(name, emb np.float32, feats dict), ...]"""
        out = []
        for p in sorted(self.dir.glob("*.json")):
            try:
                data = json.loads(p.read_text(encoding="utf-8"))
                emb = np.asarray(data["embedding"], dtype=np.float32)
                n = float(np.linalg.norm(emb))
                if n > 1e-8:
                    emb = emb / n
                feats = data.get("feats") or {}
                name = data.get("name") or p.stem
                out.append((name, emb, feats))
            except Exception as exc:
                print("[注册] 跳过损坏档案 {}: {}".format(p.name, exc))
        return out

    def save_profile(self, name, embedding, feats, duration_sec=0.0, segments=1):
        name = sanitize_name(name)
        emb = np.asarray(embedding, dtype=np.float32).reshape(-1)
        n = float(np.linalg.norm(emb))
        if n < 1e-8:
            raise ValueError("声纹向量无效（近似静音）")
        emb = emb / n
        now = time.strftime("%Y-%m-%d %H:%M:%S")
        path = self.path_for(name)
        prev = {}
        if path.exists():
            try:
                prev = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                prev = {}
        # 若已有档案：质心融合，便于多次补录
        if prev.get("embedding"):
            old = np.asarray(prev["embedding"], dtype=np.float32)
            on = float(np.linalg.norm(old))
            if on > 1e-8:
                old = old / on
                emb = emb * 0.55 + old * 0.45
                emb = emb / (float(np.linalg.norm(emb)) + 1e-8)
            segments = int(prev.get("segments", 1)) + int(segments)
            created = prev.get("created_at", now)
        else:
            created = now

        clean_feats = {}
        for k in ("pitch", "pitch_iqr", "f1", "f2", "centroid"):
            v = (feats or {}).get(k)
            if v is None:
                clean_feats[k] = None
            else:
                clean_feats[k] = float(v)

        payload = {
            "name": name,
            "embedding": emb.astype(float).tolist(),
            "feats": clean_feats,
            "duration_sec": float(duration_sec),
            "segments": segments,
            "created_at": created,
            "updated_at": now,
            "version": 1,
        }
        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return {
            "name": name,
            "path": str(path),
            "pitch": clean_feats.get("pitch"),
            "f1": clean_feats.get("f1"),
            "f2": clean_feats.get("f2"),
            "segments": segments,
            "duration_sec": float(duration_sec),
        }


# librosa.yin(fmax=350) 失败时常见贴边值 ≈ 16000/45
_PITCH_SENTINEL = 16000.0 / 45.0
_PITCH_MIN_HZ = 85.0
_PITCH_MAX_HZ = 300.0


def _speech_activity_ratio(samples: np.ndarray, sample_rate: int = 16000) -> float:
    """有语音帧占比（按短时能量）。"""
    if samples is None or len(samples) == 0:
        return 0.0
    frame = max(1, int(0.03 * sample_rate))
    hop = max(1, frame // 2)
    if len(samples) < frame:
        return 0.0
    thr = 0.008
    voiced = 0
    total = 0
    for i in range(0, len(samples) - frame + 1, hop):
        seg = samples[i : i + frame]
        total += 1
        if float(np.sqrt(np.mean(np.square(seg)))) >= thr:
            voiced += 1
    return float(voiced) / float(total) if total else 0.0


def _best_energy_window(samples: np.ndarray, sample_rate: int, win_sec: float = 4.0):
    """取能量最高的连续窗口，改善基频估计。"""
    samples = np.asarray(samples, dtype=np.float32).reshape(-1)
    win = int(win_sec * sample_rate)
    if len(samples) <= win:
        return samples
    step = max(1, win // 2)
    best_i, best_e = 0, -1.0
    for i in range(0, len(samples) - win + 1, step):
        seg = samples[i : i + win]
        e = float(np.mean(np.square(seg)))
        if e > best_e:
            best_e, best_i = e, i
    return samples[best_i : best_i + win]


def _pitch_ok(pitch) -> bool:
    if pitch is None:
        return False
    pitch = float(pitch)
    if abs(pitch - _PITCH_SENTINEL) < 1.0:
        return False
    return _PITCH_MIN_HZ <= pitch <= _PITCH_MAX_HZ


def _sanitize_enroll_feats(feats: dict) -> dict:
    """清理异常频率特征；基频失败时置空，仍允许靠 embedding 注册。"""
    out = dict(feats or {})
    pitch = out.get("pitch")
    if not _pitch_ok(pitch):
        out["pitch"] = None
        out["pitch_iqr"] = None
        # 贴边基频时常伴随不可靠共振峰
        if pitch is not None and abs(float(pitch) - _PITCH_SENTINEL) < 1.0:
            out["f1"] = None
            out["f2"] = None
    return out


def build_profile_from_audio(tracker, samples, sample_rate=16000):
    """用现有 SpeakerTracker 提取 embedding + 频率特征。"""
    samples = np.asarray(samples, dtype=np.float32).reshape(-1)
    if len(samples) < int(sample_rate * 3.0):
        raise ValueError("录音过短，请至少连续说话 8~15 秒")
    # 能量门：近静音拒绝
    peak = float(np.max(np.abs(samples))) if len(samples) else 0.0
    rms = float(np.sqrt(np.mean(np.square(samples)))) if len(samples) else 0.0
    if peak < 0.015 or rms < 0.003:
        raise ValueError("声音太小，请靠近麦克风大声清晰朗读约 15 秒")
    activity = _speech_activity_ratio(samples, sample_rate)
    if activity < 0.25:
        raise ValueError(
            "有效说话比例过低({:.0%})，请连续说话、少停顿后重录".format(activity)
        )

    # 峰值归一化，减轻阵列麦输出偏小导致 yin 估成贴边值
    if 0.015 <= peak < 0.35:
        samples = samples * (0.8 / peak)

    emb = tracker._extract(samples, sample_rate)
    if emb is None:
        win = _best_energy_window(samples, sample_rate, 4.0)
        emb = tracker._extract(win, sample_rate)
    if emb is None:
        raise ValueError("未能提取声纹，请重新录制并保证持续说话")

    feats = tracker._estimate_voice_feats(samples, sample_rate)
    if not _pitch_ok((feats or {}).get("pitch")):
        # 再对高能量窗估一次基频
        win = _best_energy_window(samples, sample_rate, 3.0)
        feats2 = tracker._estimate_voice_feats(win, sample_rate)
        if _pitch_ok((feats2 or {}).get("pitch")):
            feats = feats2
        else:
            feats = _sanitize_enroll_feats(feats)
            print(
                "[注册] 警告: 基频不可靠，已降级为仅声纹注册 pitch={}".format(
                    (feats2 or {}).get("pitch")
                ),
                flush=True,
            )
    return emb, feats
