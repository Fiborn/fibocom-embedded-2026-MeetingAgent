# -*- coding: utf-8 -*-
"""
声源定位 (DOA) 独立配置 —— 与 src/ 麦克风/ASR 模块完全隔离。
修改本文件不会影响 meeting_realtime.py 等原有程序。
"""
from pathlib import Path

DOA_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = DOA_ROOT.parent

# ===== 麦克风阵列几何（PDF：4 麦环形阵列，半径 3 cm）=====
NUM_MICS = 4
ARRAY_GEOMETRY = "ring"  # ring | linear
ARRAY_RADIUS_M = 0.03
LINEAR_MIC_SPACING_M = 0.018
# 麦克风在 XY 平面上的起始角度（度），顺时针递增
MIC_START_ANGLE_DEG = 0.0
# USB 阵列麦通道顺序若与物理 Mic0..3 不一致，在此调整
MIC_CHANNEL_ORDER = [0, 1, 2, 3]
SPEED_OF_SOUND = 340.0

# 平面阵列无法可靠估计俯仰，固定为 0°（只输出水平方位）
FIX_ELEVATION_DEG = 0.0
AZIMUTH_SEARCH_STEP_DEG = 2.0
# 相邻通道 TDOA 过小说明各通道几乎相同，角度不可信
MIN_TDOA_ABS_SEC = 8e-6
# 方位角时间平滑（帧数，1=不平滑）
AZIMUTH_SMOOTH_FRAMES = 5

# ===== 采集（独立设备参数，可与 config.py 不同）=====
INPUT_DEVICE_INDEX = 37
INPUT_CHANNELS = 4
SAMPLE_RATE = 16000
CHUNK = 1024
# 每次 DOA 分析使用的音频窗口（秒）
ANALYSIS_WINDOW_SEC = 0.25

# ===== 输出频率（PDF：100 ms 周期 ≈ 10 Hz）=====
BROADCAST_INTERVAL_MS = 100

# ===== VAD / 置信度 =====
VAD_RMS_THRESHOLD = 0.008
CONFIDENCE_MIN_FOR_ACTION = 0.6
# 8 方向量化（PDF 建议：0°, 45°, 90° …）
AZIMUTH_QUANTIZE_DEG = 45

# ===== 输出路径 =====
OUTPUT_DIR = PROJECT_ROOT / "output" / "doa"
JSON_LOG = OUTPUT_DIR / "doa_stream.jsonl"

# ===== 模拟模式（无硬件时测试 UI/接口）=====
SIMULATION_MODE = False
SIM_AZIMUTH_DEG = 125.5

DEBUG = False
