#!/usr/bin/env python3
"""
千问3-0.6B 极简终端对话程序
"""

import sys
import os
from fiboaisdk.api_aisdk_py import api_nlp_py as nlp_api
from fiboaisdk.api_aisdk_py import license_py as license_api


def read_file(path):
    """读取文件"""
    with open(path, 'r' if path.endswith('.pem') else 'rb') as f:
        return f.read()

def main():
    # 初始化许可证
    try:
        license_key1 = read_file("./qcom_6490_license/key1.pem")
        license_key2 = read_file("./qcom_6490_license/key2.pem")
        license_key3 = read_file("./qcom_6490_license/key3.pem")
        license_data = read_file("./qcom_6490_license/license.bin")
        
        ret = license_api.Init(license_key1, license_key2, license_key3, license_data)
        print(f"许可证初始化返回值: {ret}")
    except Exception as e:
        print(f"许可证初始化失败: {e}")
        sys.exit(1)
    
    # 初始化模型
    print("正在初始化千问3-0.6B模型...")
    api = None
    try:
        api = nlp_api.NLPAPI()
        # 使用您提供的路径
        api.Init("/home/fibo/qwen3/qwen3", "")
        print("千问3-0.6B模型初始化成功")
    except Exception as e:
        print(f"模型初始化失败: {e}")
        sys.exit(1)
    
    print("\n" + "="*60)
    print("千问3-0.6B 终端对话系统")
    print("输入 'quit' 或 'exit' 或 'q' 退出程序")
    print("输入 'clear' 清屏")
    print("="*60)
    
    # 简单的对话循环
    try:
        while True:
            # 获取用户输入
            user_input = input("\n你: ").strip()
            
            # 退出命令
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("\n再见！")
                break
            
            # 清屏命令
            if user_input.lower() == 'clear':
                print("\n" * 50)
                continue
            
            # 空输入处理
            if not user_input:
                continue
            
            # 构建最简单的上下文
            # 使用千问3的格式
            context = f"<|im_start|>system\n你是一个有帮助的助手。<|im_end|>\n"
            context += f"<|im_start|>user\n{user_input}<|im_end|>\n"
            context += f"<|im_start|>assistant\n"
            
            try:
                # 同步生成回答
                result = nlp_api.ResultNlpText()
                api.GenerateSync(context, result)
                
                # 打印回答
                response = result.text.strip()
                print(f"\n助手: {response}")
                
            except Exception as e:
                print(f"\n生成回答时出错: {e}")
                
    except KeyboardInterrupt:
        print("\n\n程序被用户中断")
    except Exception as e:
        print(f"\n程序运行出错: {e}")
    finally:
        # 释放资源
        if api:
            api.Release()
            print("模型资源已释放")

if __name__ == "__main__":
    main()