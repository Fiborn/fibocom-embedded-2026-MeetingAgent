"""UI 工作目录与项目路径配置。"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

UI_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = UI_DIR.parent

CONFIG_PATH = UI_DIR / "ui_config.json"
DEFAULT_CONFIG = {
    "llm_model": "qwen",
    "license_dir": "../qcom_6490_license",
    "speaker_id_dir": "speech",
    "mic": {
        "capture_backend": "alsa",
        "alsa_device": "plughw:1,0",
        "device_index": 0,
        "channels": 2,
        "primary_channel": 0,
        "sample_rate": 16000,
        "frames": 2048,
        "auto_detect_dov": True,
    },
    "speaker": {
        "use_doa": True,
        "doa_min_conf": 0.70,
        "doa_diff_deg": 70,
        "doa_weight": 0.15,
    },
    "asr": {
        "vad_threshold": 0.18,
        "min_silence_duration": 0.70,
        "min_speech_duration": 0.35,
        "target_rms": 0.11,
        "max_gain": 28.0,
        "adaptive_primary": True,
        "use_best_channel": True,
        "channel_mix": False,
        "post_correct": True,
        "queue_max": 8,
        "short_merge_max_sec": 1.10,
    },
    "llm_worker": {
        "warmup_on_start": True,
        "transcript_max_lines": 40,
        "transcript_max_chars": 3500,
        "merge_same_speaker": True,
        "request_timeout_sec": 900,
        "warmup_timeout_sec": 180,
        "refine_timeout_sec": 300,
        "progress_heartbeat_sec": 30,
    },
    "meeting_extract": {
        "mode": "rules_llm",
        "llm_refine": "auto",
        "preview_rules_first": True,
        "refine_snippet_max_chars": 400,
    },
    "harness": {
        "enabled": True,
        "enhance_extract": True,
        "write_memory_on_end": True,
        "backend": "auto",
        "top_k": 5,
        "query_chars": 800,
        "min_chars": 20,
        "memory_tags": ["meeting", "ui"],
        "search_timeout_sec": 30,
        "write_timeout_sec": 120,
    },
}


def load_config() -> dict:
    cfg = dict(DEFAULT_CONFIG)
    if CONFIG_PATH.exists():
        with CONFIG_PATH.open("r", encoding="utf-8") as f:
            loaded = json.load(f)
        for key, value in loaded.items():
            if isinstance(value, dict) and isinstance(cfg.get(key), dict):
                cfg[key].update(value)
            else:
                cfg[key] = value
    return cfg


def resolve_path(value: str | Path, base: Path = PROJECT_ROOT) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = (base / path).resolve()
    return path


def setup_workdir() -> Path:
    os.chdir(UI_DIR)
    return UI_DIR


def apply_timezone(cfg: dict) -> str:
    """按配置设置进程时区（板端默认 UTC，需切到 Asia/Shanghai 等）。

    影响此后所有 datetime.now() 的取值；各入口调用
    apply_runtime_paths() 时统一生效。
    """
    tz = str(cfg.get("timezone") or "Asia/Shanghai").strip()
    if tz and sys.platform != "win32":
        os.environ["TZ"] = tz
        try:
            time.tzset()
        except AttributeError:
            pass
    return tz


def apply_runtime_paths(cfg: dict | None = None) -> dict:
    cfg = cfg or load_config()
    apply_timezone(cfg)
    paths = {
        "ui_dir": UI_DIR,
        "project_root": PROJECT_ROOT,
        "data_dir": PROJECT_ROOT / "data",
        "meetings_dir": PROJECT_ROOT / "data" / "meetings",
        "license_dir": resolve_path(cfg.get("license_dir", "../qcom_6490_license")),
        "speaker_id_dir": resolve_path(cfg.get("speaker_id_dir", "speech")),
        "agent_out_dir": PROJECT_ROOT / "data" / "agent_tasks",
        "llm_model": cfg.get("llm_model", "qwen"),
        "mic": cfg.get("mic", DEFAULT_CONFIG["mic"]),
        "speaker": cfg.get("speaker", DEFAULT_CONFIG.get("speaker", {})),
        "asr": cfg.get("asr", DEFAULT_CONFIG.get("asr", {})),

    }
    paths["agent_out_dir"].mkdir(parents=True, exist_ok=True)
    paths["meetings_dir"].mkdir(parents=True, exist_ok=True)
    return paths
