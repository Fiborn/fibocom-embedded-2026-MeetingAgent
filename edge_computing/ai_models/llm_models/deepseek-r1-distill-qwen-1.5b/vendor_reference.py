
import os
import sys
import re
import time

os.environ['MALLOC_ARENA_MAX'] = '2'

from fiboaisdk.api_aisdk_py import api_nlp_py as nlp_api
from fiboaisdk.api_aisdk_py import license_py as license_api

def build_fixed_context(user_input):
    """
    使用固定的提示模板格式
    模板: "<|begin_of_text|>你是一个专业的AI助手。{user_input}<｜Assistant｜>"
    """
    # 移除用户输入中可能包含的模板标记，避免冲突
    cleaned_input = user_input.replace('<|begin_of_text|>', '').replace('<｜Assistant｜>', '')
    
    # 返回固定格式的上下文
    return f"<|begin_of_text|>你是一个专业的AI助手。{cleaned_input}<｜Assistant｜>"

def extract_response_fixed_format(raw_text, user_input=""):
    """
    针对固定格式提取响应
    """
    if not raw_text:
        return "没有生成有效内容"
    
    # 1. 首先找到并移除完整的提示模板部分
    template_start = "<|begin_of_text|>你是一个专业的AI助手。"
    template_end = "<｜Assistant｜>"
    
    # 查找模板结束位置
    end_pos = raw_text.find(template_end)
    if end_pos != -1:
        # 从模板结束标记之后开始提取
        response_start = end_pos + len(template_end)
        raw_text = raw_text[response_start:]
    
    # 2. 如果找不到结束标记，尝试其他方法
    else:
        # 尝试移除用户输入部分
        if user_input and user_input in raw_text:
            user_input_start = raw_text.find(user_input)
            if user_input_start != -1:
                # 在用户输入后查找"<｜Assistant｜>"
                potential_end = raw_text.find("<｜Assistant｜>", user_input_start)
                if potential_end != -1:
                    raw_text = raw_text[potential_end + len("<｜Assistant｜>"):]
    
    # 3. 应用停止条件
    stop_conditions = [
        # 下一个用户输入开始
        '<|begin_of_text|>',
        '<｜User｜>',
        '用户：',
        '问题：',
        '下一个问题',
        # 模型特殊标记
        '<|endoftext|>',
        # 明显的重复开始
        '<｜问题',
        '。问题',
        # 常见的对话结束
        '\n\n\n',
    ]
    
    # 找到最早的停止点
    earliest_stop = len(raw_text)
    for stop in stop_conditions:
        pos = raw_text.find(stop)
        if pos != -1 and pos < earliest_stop:
            earliest_stop = pos
    
    # 截取到停止点
    if earliest_stop < len(raw_text):
        raw_text = raw_text[:earliest_stop]
    
    # 4. 清理文本
    raw_text = raw_text.strip()
    
    # 移除开头的标点
    while raw_text and raw_text[0] in ['，', '。', '；', ':', '：', ' ', '\n', '\t']:
        raw_text = raw_text[1:]
    
    # 5. 重复检测和清理
    # 检测重复句子
    sentences = re.split(r'[。！？；]', raw_text)
    if len(sentences) > 4:
        # 检查是否有连续重复的句子
        for i in range(len(sentences) - 2):
            if sentences[i] and len(sentences[i]) > 3:
                if sentences[i] == sentences[i+1] == sentences[i+2]:
                    # 找到第一个重复句子的位置
                    first_pos = raw_text.find(sentences[i])
                    if first_pos != -1:
                        raw_text = raw_text[:first_pos + len(sentences[i])]
                        break
    
    # 6. 检测关键词重复
    repeat_keywords = ['问题', '用户', '助手', 'AI', '回答']
    for keyword in repeat_keywords:
        if raw_text.count(keyword) > 8:
            # 关键词出现太多，可能有问题
            # 保留到第5次出现的位置
            positions = [m.start() for m in re.finditer(keyword, raw_text)]
            if len(positions) >= 5:
                raw_text = raw_text[:positions[4] + 20]
                break
    
    # 7. 限制长度
    max_length = 800
    if len(raw_text) > max_length:
        # 尝试在句号处截断
        last_period = raw_text[:max_length].rfind('。')
        if last_period > max_length * 0.6:
            raw_text = raw_text[:last_period + 1]
        else:
            raw_text = raw_text[:max_length] + "..."
    
    # 8. 最终清理
    raw_text = re.sub(r'\s+', ' ', raw_text)
    raw_text = raw_text.strip()
    
    if not raw_text or len(raw_text) < 2:
        if user_input:
            return f"已生成关于'{user_input}'的回答。"
        else:
            return "已生成回答。"
    
    return raw_text

def debug_raw_response(raw_text):
    """
    调试函数：分析原始响应
    """
    print("\n=== 调试原始响应 ===")
    print(f"总长度: {len(raw_text)}")
    
    # 查找关键标记的位置
    markers = [
        '<|begin_of_text|>',
        '<｜Assistant｜>',
        '<｜User｜>',
        '<|endoftext|>',
        '用户：',
        '问题：',
    ]
    
    for marker in markers:
        pos = raw_text.find(marker)
        if pos != -1:
            print(f"找到 '{marker}' 在位置 {pos}")
            # 显示上下文
            start = max(0, pos - 20)
            end = min(len(raw_text), pos + 20)
            print(f"  上下文: ...{raw_text[start:end]}...")
    
    # 显示前200个字符和后200个字符
    print(f"\n前200字符: {raw_text[:200]}")
    print(f"\n后200字符: {raw_text[-200:] if len(raw_text) > 200 else raw_text}")
    print("=== 调试结束 ===\n")

def main():
    print("千问3-0.6B 固定格式版")
    print("使用格式: <|begin_of_text|>你是一个专业的AI助手。{用户输入}<｜Assistant｜>")
    
    def read_file(path):
        with open(path, 'r' if path.endswith('.pem') else 'rb') as f:
            return f.read()
    
    try:
        license_key1 = read_file("./qcom_6490_license/key1.pem")
        license_key2 = read_file("./qcom_6490_license/key2.pem")
        license_key3 = read_file("./qcom_6490_license/key3.pem")
        license_data = read_file("./qcom_6490_license/license.bin")
        
        ret = license_api.Init(license_key1, license_key2, license_key3, license_data)
        if ret != 0:
            print(f"许可证失败: {ret}")
            return
    except Exception as e:
        print(f"许可证错误: {e}")
        return
    
    try:
        api = nlp_api.NLPAPI()
        model_path = "/home/fibo/qwen3/deepseek-r1-qwen-1.5b_1.0.0_all_all_mnn_3.0.5_cpu_206fb9a374c94c4d4d52bf0e873c739c.fmodel"
        api.Init(model_path, "")
        print("模型加载成功")
    except Exception as e:
        print(f"模型错误: {e}")
        return
    
    print("\n使用固定格式: <|begin_of_text|>你是一个专业的AI助手。[用户问题]<｜Assistant｜>")
    print("输入 'quit' 退出")
    print("输入 'test' 运行测试")
    print("输入 'raw' 显示原始输出")
    print("输入 'debug' 调试模式")
    print("-" * 50)
    
    show_raw = False
    debug_mode = False
    
    while True:
        try:
            user_input = input("\n你: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                break
            
            if user_input.lower() == 'raw':
                show_raw = not show_raw
                print(f"{'显示' if show_raw else '隐藏'}原始输出")
                continue
            
            if user_input.lower() == 'debug':
                debug_mode = not debug_mode
                print(f"{'开启' if debug_mode else '关闭'}调试模式")
                continue
            
            if user_input.lower() == 'test':
                print("\n运行测试...")
                test_cases = [
                    "请介绍一下你自己",
                    "深圳是哪个国家的城市？",
                    "1+1等于多少？",
                    "如何学习编程？",
                ]
                
                for question in test_cases:
                    print(f"\n测试: {question}")
                    context = build_fixed_context(question)
                    print(f"[上下文: {context}]")
                    
                    result = nlp_api.ResultNlpText()
                    api.GenerateSync(context, result)
                    
                    raw_response = result.text.strip()
                    response = extract_response_fixed_format(raw_response, question)
                    
                    print(f"回答: {response}")
                    
                    if show_raw:
                        print(f"[原始: {raw_response[:150]}...]")
                    
                    time.sleep(1)
                
                continue
            
            if not user_input:
                continue
            
            # 构建固定格式的上下文
            context = build_fixed_context(user_input)
            print(f"[使用的上下文: {context}]")
            
            # 生成回答
            print("生成中...", end="", flush=True)
            start_time = time.time()
            
            result = nlp_api.ResultNlpText()
            api.GenerateSync(context, result)
            
            elapsed = time.time() - start_time
            print(f"\r生成时间: {elapsed:.1f}s", end="")
            print("\r" + " " * 30 + "\r", end="")
            
            raw_response = result.text.strip()
            
            if debug_mode:
                debug_raw_response(raw_response)
            
            response = extract_response_fixed_format(raw_response, user_input)
            
            print(f"助手: {response}")
            
            if show_raw:
                print(f"[原始输出: {raw_response[:250]}...]")
                print(f"[原始长度: {len(raw_response)}字符]")
            
        except KeyboardInterrupt:
            print("\n退出")
            break
        except Exception as e:
            print(f"错误: {e}")
    
    api.Release()
    print("结束")

if __name__ == "__main__":
    main()