"""
全局配置 —— 实时会议：说话人 + 说话内容（无时间戳）
竞赛版参数经多轮实测调优，一般无需再改。
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# ===== 模型路径（python3 scripts/download_model.py）=====
ASR_DIR = ROOT / "models" / "asr"
ASR_MODEL = ASR_DIR / "model.onnx"
ASR_TOKENS = ASR_DIR / "tokens.txt"
EMBEDDING_MODEL = ROOT / "models" / "embedding.onnx"
VAD_MODEL = ROOT / "models" / "silero_vad.onnx"

# ===== 麦克风（USB DOV；卡号可能漂移，UI 会 auto_detect）=====
CAPTURE_BACKEND = "alsa"
ALSA_DEVICE = "hw:1,0"
INPUT_DEVICE_INDEX = 0
INPUT_CHANNELS = 4
PRIMARY_CHANNEL = 0
SAMPLE_RATE = 16000
CHUNK = 2048

# ===== 说话人分离（防异人同标；同人粘贴略收紧）=====
SPEAKER_MAX = 8
SPEAKER_LABEL_PREFIX = "说话人"
SPEAKER_NUM_THREADS = 2
SPEAKER_SAME_THRESHOLD = 0.54
SPEAKER_SAME_THRESHOLD_CAP = 0.62
SPEAKER_NEW_THRESHOLD = 0.38
SPEAKER_MARGIN = 0.04
SPEAKER_PENDING_COUNT = 3
SPEAKER_BOOTSTRAP_SEGMENTS = 4
SPEAKER_SELF_SIM_HISTORY = 30
SPEAKER_EMA_ALPHA = 0.10
SPEAKER_FIRST_HIGH = 0.58
SPEAKER_PITCH_DIFF_HZ = 45
SPEAKER_MERGE_THRESHOLD = 0.64
SPEAKER_MERGE_MAX_COUNT = 10
SPEAKER_EMBED_MIN_SEC = 0.80
# 判定「确认为同一人」时的声纹下限
SPEAKER_CONFIDENT_EMB_MIN = 0.48
# 模糊区间粘贴上一标签的最低相似度（过低易异人同标）
SPEAKER_STICKY_MIN_SIM = 0.46
# 模糊时弱更新已有档案的最低声纹分（过低会把第二人粘到说话人1）
SPEAKER_REMATCH_MIN = 0.50
# 频率辅助：基频分布 + 共振峰 + 频谱质心
SPEAKER_FREQ_WEIGHT = 0.14
SPEAKER_FORMANT_DIFF_HZ = 280.0
SPEAKER_CENTROID_DIFF_HZ = 700.0

# ===== DOA 融合（仅真 4 麦阵列有价值；假 4ch 时 UI 会关闭）=====
SPEAKER_USE_DOA = True
SPEAKER_DOA_WEIGHT = 0.22
SPEAKER_DOA_DIFF_DEG = 45.0
SPEAKER_DOA_MIN_CONF = 0.55
SPEAKER_DOA_PROFILE_ALPHA = 0.15
SPEAKER_DOA_MIN_SEC = 0.35
SPEAKER_MULTI_RING_SEC = 30.0
# ===== 声纹预注册 =====
ENROLL_DIR = ROOT / "data" / "enrolled"
# 实时短句嵌入与 15s 注册档案差异大，0.52 过高会导致永远落成「说话人N」
SPEAKER_ENROLL_MATCH_THRESHOLD = 0.40
SPEAKER_ENROLL_MATCH_MARGIN = 0.08
SPEAKER_ENROLL_MATCH_FLOOR = 0.34
SPEAKER_ENROLL_COUNT = 12
SPEAKER_ENROLL_DEFAULT_SEC = 15.0

# ===== VAD（略灵敏：减少句首/轻声漏切；静音稍短避免段被拖断）=====
MIN_SPEECH_DURATION = 0.30
MIN_SILENCE_DURATION = 0.55
VAD_THRESHOLD = 0.15
VAD_MIN_SPEECH_DURATION = 0.15
# 开场冷启动：前 N 秒用更短静音切段，减少首句被截碎
VAD_COLDSTART_SEC = 15.0
VAD_COLDSTART_MIN_SILENCE = 0.40

# ===== ASR =====
ASR_NUM_THREADS = 4
ASR_MIN_RMS = 0.0035
ASR_MIN_RAW_RMS = 0.002
# Timesintelli 4ch@16k：目标音量适中，避免 max_gain 过大把底噪抬成乱字
ASR_TARGET_RMS = 0.10
ASR_MAX_GAIN = 12.0
ASR_MAX_SEGMENT_SEC = 15.0
ASR_MIN_SEC = 0.35
ASR_DEDUPE_MAX = 200
ASR_QUEUE_MAX = 12
# 开录前预滚进 VAD，避免第一句被从半截开始
ASR_PREROLL_SEC = 1.2
# 短句合并：过短段暂存，与下一段拼接再送 ASR（稍长更利于 Paraformer）
ASR_SHORT_MERGE_MAX_SEC = 0.90
ASR_SHORT_MERGE_GAP_SEC = 1.20
# 多麦：按能量选最佳通道 / 能量加权混合，避免固定 ch0 过弱
ASR_ADAPTIVE_PRIMARY = True
ASR_USE_BEST_CHANNEL = True
# 默认只选最强通道，避免弱通道/零增益通道掺进来拉低 SNR
ASR_CHANNEL_MIX = False
# 去直流；预加重对 Paraformer 原模型常有害，默认关
ASR_REMOVE_DC = True
ASR_PREEMPHASIS = 0.0
# 会议常见同音/错词后处理
ASR_POST_CORRECT = True
# recognizer 重建间隔（过大/过频都不好；0=从不）
ASR_RECREATE_EVERY = 120

DEBUG = False

# 路径
DATA_MEETINGS = ROOT / "data" / "recordings"
OUTPUT_DIR = ROOT / "output"
OUTPUT_LOG = OUTPUT_DIR / "meeting_transcript.txt"
OUTPUT_AUDIOS_DIR = OUTPUT_DIR / "audios"
OUTPUT_TRANSCRIPTS_DIR = OUTPUT_DIR / "transcripts"
SPEAKER_ACTIVITY_LOG = OUTPUT_DIR / "speaker_activity.txt"
