from fiboaisdk.api_aisdk_py import api_audio_py as au_api
from fiboaisdk.api_aisdk_py import license_py as license_api

model_path="/home/fibo/model/whisper_tiny_cpu_1.0.0_onnx_all_0bc78d5f7ef9f78362960cbf5ca755fc.fmodel"

def read_file(path):
    with open(path, 'r' if path.endswith('.pem') else 'rb') as f:
        return f.read()

if __name__ == "__main__":
    # 假设四个文件路径如下，请根据实际情况修改
    key1_path = "./qcom_6490_license/key1.pem"
    key2_path = "./qcom_6490_license/key2.pem"
    key3_path = "./qcom_6490_license/key3.pem"
    license_bin_path = "./qcom_6490_license/license1.bin"

    # 读取文件内容
    license_key1 = read_file(key1_path)
    license_key2 = read_file(key2_path)
    license_key3 = read_file(key3_path)
    license_data = read_file(license_bin_path)

    # 调用License初始化
    ret = license_api.Init(license_key1, license_key2, license_key3, license_data)
    print(f"License 初始化返回值: {ret}")

    # 创建 FiboAudio 对象
    audio = au_api.FiboAudio()
    audio.audio_sample_rate = 16000
    audio.audio_channel = 1
    audio.audio_format = au_api.FiboAudioFormat.FIBO_AUDIO_FORMAT_WAV
    audio.audio_path = "output.wav"
    audio.extra_params = ""  # 新增字段：模型运行超参

    # 创建 AUAPI 对象
    api = au_api.AudioAPI()
    api.Init(model_path)

    print("=== 音频转录示例 ===")
    # 同步转录
    result = au_api.ResultNlpAudio()
    api.TranscribeSync(audio, result, 10)
    print(f"转录结果: {result.speech_text}")

    # 释放资源
    api.Release()
    print("API资源已释放")
