# DOA 声源定位独立模块

与 `src/` 下的麦克风采集、ASR、说话人识别**完全隔离**，可单独运行，互不影响。

## 目录结构

```
doa/
  config_doa.py       # 独立配置（阵列几何、广播频率、阈值）
  doa_types.py        # PDF JSON / DoaResult_t 数据结构
  gcc_phat.py         # GCC-PHAT 时延估计
  array_geometry.py   # 环形阵列几何 + 角度解算
  vad_simple.py       # vad_active 能量检测
  mic_array.py        # 4 通道独立采集（不依赖 src/audio_capture.py）
  engine.py           # DOA 引擎
  broadcaster.py      # 100ms JSON 广播（控制台 / JSONL / TCP）
  run.py              # 主入口
  radar_demo.py       # ASCII 雷达 UI 演示
  consumer_example.py # UI/摄像头消费端示例
```

输出目录：`output/doa/doa_stream.jsonl`

## 快速开始

### 1. 环境（与主项目共用依赖）

在 `speech/` 目录下执行：

```bash
python3 -m pip install --user -r requirements-py38.txt
```

### 2. 无硬件联调（模拟模式，推荐第一步）

```bash
python doa/run.py --simulate --azimuth 125.5 --tcp 9900
```

另开终端查看雷达 Demo：

```bash
python doa/radar_demo.py --port 9900
```

### 3. 真实 4 通道阵列麦

确认设备编号：

```bash
python src/list_devices.py
```

编辑 `doa/config_doa.py`：

```python
INPUT_DEVICE_INDEX = 0   # 改成你的设备号
INPUT_CHANNELS = 4
```

启动：

```bash
python doa/run.py
```

按 `Ctrl+C` 结束。日志写入 `output/doa/doa_stream.jsonl`。

### 4. 仅输出 JSON（供其他进程管道读取）

```bash
python doa/run.py --json-only
```

### 5. UI / 摄像头消费端对接

参考 `doa/consumer_example.py`：

```python
from doa.consumer_example import handle_doa_packet

action = handle_doa_packet(json_line)
# vad_active=false → idle
# confidence>=0.6  → track，含 pan/tilt/beam_index
```

或通过 TCP 订阅：

```bash
python doa/run.py --tcp 9900
# UI 进程 connect 127.0.0.1:9900，按行读 JSON
```

## JSON 输出格式（与 PDF 一致）

```json
{
  "timestamp": 1684321098765,
  "vad_active": true,
  "source_data": {
    "azimuth": 125.5,
    "elevation": 15.0,
    "confidence": 0.85,
    "beam_index": 3
  }
}
```

| 字段 | 说明 |
|------|------|
| `timestamp` | 毫秒时间戳 |
| `vad_active` | 是否有人说话（开关） |
| `azimuth` | 水平方位角 0–360°（量化到 45° 步进） |
| `elevation` | 俯仰角 0–45° |
| `confidence` | 置信度 0–1，建议 >0.6 再驱动摄像头 |
| `beam_index` | 波束编号 0–7，供 ASR 选路 |

## 与原有麦克风模块的关系

| 模块 | 路径 | 用途 |
|------|------|------|
| 会议识别 | `src/meeting_realtime.py` | 单通道 ASR + 说话人 |
| 声源定位 | `doa/run.py` | 4 通道 TDOA + 角度 |

两者使用**不同的配置、不同的采集类**，不要同时占用同一麦克风设备；若必须并行，请使用不同 `INPUT_DEVICE_INDEX` 或分时运行。

## 开发路线（PDF 建议）

1. **第一步**：先跑通 `vad_active`（已实现）
2. **第二步**：8 方向 `azimuth` + 雷达 Demo（已实现）
3. **第三步**：对接真实 PTZ 摄像头 / 触摸屏 UI（用 `consumer_example.py` 扩展）

## 常见问题

- **ImportError**：请在项目根目录运行，或使用 `python doa/run.py`。
- **麦克风被占用**：先关闭 `meeting_realtime.py`，再启动 DOA。
- **角度乱跳**：调高 `CONFIDENCE_MIN_FOR_ACTION` 或增大 `ANALYSIS_WINDOW_SEC`。
