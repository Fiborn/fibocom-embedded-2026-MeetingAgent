# 端侧模型仓库说明

本目录存放工程所需的端侧模型。**模型权重文件（`*.fmodel` / `*.onnx` / `*.dlc`）体积合计约 3.1 GB，不随 Git 仓库分发**，部署时按下表自行放置；仓库内保留了各模型目录的配置文件、词典与调用代码（`vendor_reference.py`、`tts.py` 等）。

## 模型清单与获取方式

| 目录 | 权重文件 | 体积 | 用途 | 获取方式 |
|------|----------|------|------|----------|
| `asr_models/fiboasr/` | `fiboasr_base_v1_0711_qcom_6490_8550_snpe_2_29_dsp_*.fmodel` | 227 MB | 实时 ASR（QNN/SNPE DSP） | 广和通 SC171 SDK / fiboaisdk 附带 |
| `llm_models/Qwen3-0.6B/` | `qwen3-0.6b_1.0.0_qcom_6490_qnn_2.28_dsp_*.fmodel` | 897 MB | 纪要精炼主力 LLM（QNN DSP） | 广和通 SC171 SDK 附带 |
| `llm_models/deepseek-r1-distill-qwen-1.5b/` | `deepseek-r1-qwen-1.5b_1.0.0_all_all_mnn_3.0.5_cpu_*.fmodel` | 1.4 GB | 备选 LLM（MNN CPU） | 广和通 SC171 SDK 附带 |
| `tts_models/` | `fibotts_1.0.0_qcom_6490-8550_qnn_2.26_dsp_*.fmodel` | 124 MB | 语音播报（QNN DSP） | 广和通 SC171 SDK 附带 |
| `embedding_models/bge-small-zh-v1.5/` | `model.onnx` / `model.dlc` | 91 MB / 90 MB | 记忆检索向量后端 | HuggingFace [`BAAI/bge-small-zh-v1.5`](https://huggingface.co/BAAI/bge-small-zh-v1.5)（ONNX 导出；DLC 为 Qualcomm 转换版） |

`speech/models/` 下的三个模型（sherpa-onnx ASR `model.onnx` 232 MB、声纹 `embedding.onnx` 38 MB、`silero_vad.onnx` 0.6 MB）由脚本自动下载：

```bash
python3 speech/scripts/download_model.py
```

## 放置校验

权重放好后，`models/asr/tokens.txt` 等已入库的小文件位置不要改动；运行自检确认：

```bash
python3 speech/src/check_env.py     # 环境与模型自检
```

各模型目录结构与文件名需与上表一致（代码内为固定路径引用），完整部署说明见仓库根 README「快速开始」与 `docs/工程README.md`。
