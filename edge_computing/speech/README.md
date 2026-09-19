# SC171 实时会议识别

**输出格式（无时间戳）：**

```
说话人1: 今天我们来讨论项目进度
说话人2: 好的我先汇报一下
```

- 直接使用系统 **Python 3.8.10**，**无需虚拟环境**
- 无需预注册，说完一句立即输出

## 快速开始

```bash
cd speech
bash scripts/install_env.sh          # 安装库到系统 Python
python3 scripts/download_model.py
python3 src/check_env.py
python3 src/meeting_realtime.py      # Ctrl+C 结束
```

## 主要脚本

| 脚本 | 用途 |
|------|------|
| **`src/meeting_realtime.py`** | 实时会议（主程序） |
| `src/analyze_file.py` | 分析已有 wav |
| `scripts/install_env.sh` | 安装依赖（无 venv） |
| `scripts/run_demo.py` | 一键 Demo：启动实时会议识别 30 秒 |
| `scripts/real_mic_pipeline_check.py` | 真机采集 + 引擎短时跑通检查（约 12 秒） |
| `scripts/test_split_preference.py` | 验证不同声纹不会被粘成同一标签 |

工程总览与板端部署步骤：`../README.md`
运行时组件与部署细节：`../ai_runtime/README.md`
