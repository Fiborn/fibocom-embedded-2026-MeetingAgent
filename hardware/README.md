# 硬件清单与环境要求

本作品为纯边缘侧系统，无自制 PCB / 结构件，硬件为成品开发板 + 外设组合。

## 硬件组成

| 部件 | 型号/要求 | 说明 |
|------|-----------|------|
| 主控板卡 | 广和通 SC171（Qualcomm QCS6490） | 板端 Linux，系统 Python 3.8.10 |
| 麦克风 | USB 四麦克风阵列（Timesintelli / DOV） | 自动探测声卡与通道数；普通 USB 声卡亦可运行（降级为单/双通道） |
| 显示 | DSI / HDMI 实体屏 | 桌面 UI 全屏 Kiosk；或任意设备浏览器访问 Web UI（`http://<板卡IP>:8787`） |
| License | `/home/fibo/qcom_6490_license/` | `key1.pem`、`key2.pem`、`key3.pem`、`license.bin` 四个文件，QNN DSP 模型授权必需 |

## 板端环境

| 项目 | 要求 |
|------|------|
| SDK | `fiboaisdk`（`/usr/local/lib/python3.8/dist-packages/fiboaisdk`） |
| Python | 系统 Python 3.8.10，**不创建虚拟环境** |
| 部署路径 | `/home/fibo/MeetingAgent`（License 目录与工程同级） |
| 运行位置 | 必须在板端本机终端执行（沙箱/容器会隔离 `/dev/snd` 音频设备） |

依赖安装与自检见 `edge_computing/speech/scripts/`（`install_env.sh`、`check_env.py`）。
