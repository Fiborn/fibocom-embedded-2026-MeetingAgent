#!/usr/bin/env python3
"""
TTS语音合成示例 
"""

from fiboaisdk.api_aisdk_py import api_audio_py as au_api
from fiboaisdk.api_aisdk_py import license_py as license_api
import json
import os

# ==================== 配置区域 ====================
# 所有路径和参数在这里统一设置

# 项目根目录
PROJECT_ROOT = "/home/fibo/project/voice"

# 许可证文件路径（统一标准路径）
LICENSE_CONFIG = {
    "key1_path": "/home/fibo/qcom_6490_license/key1.pem",
    "key2_path": "/home/fibo/qcom_6490_license/key2.pem",
    "key3_path": "/home/fibo/qcom_6490_license/key3.pem",
    "license_bin_path": "/home/fibo/qcom_6490_license/license.bin"
}

# TTS模型配置
TTS_CONFIG = {
    "model_path": f"{PROJECT_ROOT}/TTS",           # TTS模型目录路径
    "output_dir": f"{PROJECT_ROOT}/output",        # 输出目录
    "output_filename": "tts_output.wav",           # 输出文件名
    "language": "zh"                               # 默认语言: zh-中文, en-英文
}

# 音频参数配置
AUDIO_CONFIG = {
    "sample_rate": 16000,                          # 采样率
    "channels": 1,                                 # 声道数
    "format": au_api.FiboAudioFormat.FIBO_AUDIO_FORMAT_WAV  # 音频格式
}

# 要合成的文本
TEXT_TO_SPEAK = "你好，欢迎使用Fibocom TTS语音合成系统。"

# ==================== 工具函数 ====================

def read_file(path):
    """读取文件内容"""
    try:
        with open(path, 'r' if path.endswith('.pem') else 'rb') as f:
            return f.read()
    except Exception as e:
        print(f"读取文件失败 {path}: {e}")
        return None

def check_paths():
    """检查所有配置路径是否存在"""
    
    # 检查许可证文件
    license_files = [
        (LICENSE_CONFIG["key1_path"], "许可证密钥1"),
        (LICENSE_CONFIG["key2_path"], "许可证密钥2"),
        (LICENSE_CONFIG["key3_path"], "许可证密钥3"),
        (LICENSE_CONFIG["license_bin_path"], "许可证文件")
    ]
    
    all_ok = True
    for path, description in license_files:
        if not os.path.exists(path):
            print(f"错误: {description}不存在 - {path}")
            all_ok = False
    
    # 检查TTS模型路径
    if not os.path.exists(TTS_CONFIG["model_path"]):
        print(f"错误: TTS模型目录不存在 - {TTS_CONFIG['model_path']}")
        all_ok = False
    
    # 创建输出目录（如果不存在）
    if not os.path.exists(TTS_CONFIG["output_dir"]):
        try:
            os.makedirs(TTS_CONFIG["output_dir"], exist_ok=True)
            print(f"提示: 已创建输出目录 - {TTS_CONFIG['output_dir']}")
        except Exception as e:
            print(f"错误: 无法创建输出目录 - {TTS_CONFIG['output_dir']}: {e}")
            all_ok = False
    
    return all_ok

# ==================== 主程序 ====================

def main():
    """主程序"""
    
    # 1. 检查路径
    if not check_paths():
        print("错误: 路径检查失败，请检查配置文件")
        return
    
    # 2. 初始化许可证
    print("初始化许可证...")
    
    license_key1 = read_file(LICENSE_CONFIG["key1_path"])
    license_key2 = read_file(LICENSE_CONFIG["key2_path"])
    license_key3 = read_file(LICENSE_CONFIG["key3_path"])
    license_data = read_file(LICENSE_CONFIG["license_bin_path"])
    
    if None in [license_key1, license_key2, license_key3, license_data]:
        print("错误: 无法读取许可证文件")
        return
    
    ret = license_api.Init(license_key1, license_key2, license_key3, license_data)
    print(f"License 初始化返回值: {ret}")
    
    # 3. 创建 AudioAPI 对象
    api = au_api.AudioAPI()
    
    try:
        # 4. 初始化TTS模型
        print(f"初始化TTS模型...")
        ret = api.Init(TTS_CONFIG["model_path"])
        print(f"TTS模型初始化返回值: {ret}")
        
        if ret != 0:
            print(f"错误: TTS模型初始化失败，错误码: {ret}")
            return
        
        # 5. 准备输出文件路径
        output_path = os.path.join(TTS_CONFIG["output_dir"], TTS_CONFIG["output_filename"])
        print(f"合成文本: {TEXT_TO_SPEAK}")
        
        # 6. 创建 FiboAudio 对象（用于输出）
        output_audio = au_api.FiboAudio()
        output_audio.audio_sample_rate = AUDIO_CONFIG["sample_rate"]
        output_audio.audio_channel = AUDIO_CONFIG["channels"]
        output_audio.audio_format = AUDIO_CONFIG["format"]
        output_audio.audio_path = output_path
        output_audio.extra_params = json.dumps({"language": TTS_CONFIG["language"]})
        
        # 7. 执行TTS合成
        print("正在合成语音...")
        status = api.SpeechSynthesisSync(TEXT_TO_SPEAK, output_audio)
        print(f"TTS合成返回值: {status}")
        
        # 8. 检查结果
        if status == 0:
            if os.path.exists(output_path):
                file_size = os.path.getsize(output_path)
                print(f"成功: TTS合成完成")
                print(f"文件: {output_path}")
                print(f"大小: {file_size} 字节")
            else:
                print(f"警告: 合成成功但文件未找到 - {output_path}")
        else:
            print(f"错误: TTS合成失败，错误码: {status}")
            
    except Exception as e:
        print(f"异常: 程序执行出错 - {e}")
        
    finally:
        # 9. 释放资源
        api.Release()
        print("API资源已释放")

if __name__ == "__main__":
    main()