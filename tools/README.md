# 辅助工具索引

本作品的辅助与调试脚本集中在工程树内，未单独拆分：

| 工具 | 位置（相对 `edge_computing/`） | 用途 |
|------|-------------------------------|------|
| 环境安装 | `speech/scripts/install_env.sh` | apt + pip 依赖一键安装（清华源） |
| 模型下载 | `speech/scripts/download_model.py` | 下载 sherpa-onnx ASR / 声纹 / VAD 模型 |
| 声卡列举 | `speech/src/list_devices.py` | 列出可用声卡与通道 |
| 环境自检 | `speech/src/check_env.py` | 依赖与模型完整性检查 |
| 通道诊断 | `speech/doa/diagnose_channels.py` | 多麦阵列通道质量诊断 |
| DOA 演示 | `speech/doa/radar_demo.py` / `run.py` | GCC-PHAT 声源方位雷达演示 |
| 评估工具 | `speech/scripts/`（其余） | 语音链路评估脚本 |

声纹注册工具：`speech/src/speaker_enroll.py`（支持中文姓名，注册档案写入 `speech/data/enrolled/<姓名>.json`，该目录下的个人声纹数据不入库）。
