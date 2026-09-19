# TTS 语音合成

基于 Fibo AI 语音 SDK 的文本转语音（TTS）工程，支持中文、英文语音合成。  
板卡工作目录：`/home/fibo/project/voice/`

## 文件功能
- `tts.py`：TTS 合成主脚本，将指定文本转为 WAV 音频文件
- `fibotts_...`：TTS 模型文件
- `s2t_map.bin`、`t2s_map.bin`：音素映射辅助文件
- `output/`：合成音频输出目录（自动创建）

## 板卡端部署与运行
1. 将所有文件传至 `/home/fibo/project/voice/`
2. 确保许可证文件存在于 `/home/fibo/qcom_6490_license/` 目录下（`key1.pem`, `key2.pem`, `key3.pem`, `license.bin`）
3. 修改 `tts.py` 中 `TEXT_TO_SPEAK` 为要合成的文本，可调整语言（`zh` / `en`）
4. 运行程序：python3 tts.py