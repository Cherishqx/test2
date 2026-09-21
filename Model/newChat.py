import os
import torch
import time
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer
import json
from typing import Tuple, Dict, Any


def load_model_components(model_name: str) -> Tuple[Any, Any, Any]:
    """初始化运行环境
    功能：设置CUDA环境变量并清理GPU显存
    """
    # 设置环境变量提升CUDA运行稳定性
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

    # 清理GPU缓存
    torch.cuda.empty_cache()
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Environment initialized")


    """加载模型组件
    功能：加载LLM模型、tokenizer并创建streamer
    参数：
        model_name: 模型路径/名称
    返回值：
        tuple(模型对象, tokenizer对象, streamer对象)
    """
    # 组件加载计时
    start_time = time.time()

    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,  # 使用半精度节省显存
        device_map="cuda:0"  # 指定GPU设备
    )

    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # 创建文本流式输出器
    streamer = TextStreamer(
        tokenizer,
        skip_prompt=True,
        skip_special_tokens=True
    )

    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Components loaded in {time.time() - start_time:.2f}s")
    return model, tokenizer, streamer


def generate_chat_prompt(user_prompt: str) -> str:
    """生成对话提示模板
    功能：构造符合模型要求的对话格式
    参数：
        user_prompt: 用户原始输入
    返回值：
        格式化后的完整prompt
    """
    messages = [
        {
            "role": "system",
            "content": f"""
                Task: Solve the given math problem step-by-step, 
                providing a detailed and logical reasoning process for each step. 
                Ensure that the solution is clear, accurate, and easy to follow.

                Guidelines:
                    Focus on Problem Solving: Do not rely on external databases, articles,
                    or pre-existing solutions. 
                    Instead, focus on applying mathematical knowledge and logical reasoning to solve the problem directly.
                    Understand the Problem: Carefully read and understand the problem statement.
                    Identify what is given and what needs to be found.
                    Plan the Solution: Break down the problem into smaller, manageable parts.
                    Use mathematical concepts, formulas, or methods that are appropriate for solving it.
                    Execute Step-by-Step:
                        Step 1: Start with the first part of the problem.
                        Explain the reasoning behind each step clearly.
                        Step 2: Continue with the next part, building on the previous steps.
                        Ensure each step logically follows from the previous one.
                        Step 3: Repeat the process until the problem is fully solved.
                    Verify the Solution: Check the final answer to ensure it is reasonable and correct.
                    Provide a brief explanation of why the solution makes sense.
                
                Note: Prioritize solving the problem using your existing mathematical knowledge and logical reasoning. 
                Avoid unnecessary retrieval of external information unless explicitly required by the problem.
            """
        },
        {
            "role": "user",
            "content": user_prompt
        }
    ]
    return messages


def run_model_inference(
        model: Any,
        tokenizer: Any,
        streamer: Any,
        prompt: str,
        device: str = "cuda",
        max_new_tokens: int = 512
) -> Tuple[str, Dict, torch.Tensor]:
    """执行模型推理
    功能：处理输入并生成模型响应
    参数：
        model: 加载的模型对象
        tokenizer: 加载的分词器
        streamer: 文本流式输出器
        prompt: 用户原始输入
        device: 运行设备
        max_new_tokens: 最大生成token数
    返回值：
        tuple(生成的响应文本, 模型输入张量, 完整输出张量)
    """
    start_time = time.time()

    # 构造对话模板
    messages = generate_chat_prompt(prompt)
    formatted_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    # 准备模型输入
    model_inputs = tokenizer(
        [formatted_text],
        return_tensors="pt"
    ).to(device)

    # 执行生成
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=max_new_tokens,
        streamer=streamer,
    )
    print("\n" + "=" * 40)
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Response generated")
    # 解码输出
    response = tokenizer.decode(
        generated_ids[0][model_inputs.input_ids.shape[-1]:],
        skip_special_tokens=True
    )
    print("\n" + "=" * 40)
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Decode finished! Generated in {time.time() - start_time:.2f}s")
    return response, model_inputs, generated_ids


def format_output(response, model_inputs, generated_ids, model_name = "Qwen2.5-1.5B-Instruct"):
    start_time = time.time()
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Formatting output...")
    prompt_tokens = model_inputs.input_ids.shape[-1]
    completion_tokens = generated_ids.shape[-1] - model_inputs.input_ids.shape[-1]
    total_tokens = prompt_tokens + completion_tokens

    output = {
        "id": f"chatcmpl-{int(time.time())}-{total_tokens}-{prompt_tokens}-{completion_tokens}-{os.urandom(8).hex()}",
        "choices": [
            {
                "finish_reason": "stop",
                "index": 0,
                "logprobs": None,
                "message": {
                    "content": response,
                    "role": "assistant",
                    "function_call": None,
                    "tool_calls": None
                }
            }
        ],
        "created": int(time.time()),
        "model": model_name,
        "object": "chat.completion",
        "system_fingerprint": None,
        "usage": {
            "completion_tokens": completion_tokens,
            "prompt_tokens": prompt_tokens,
            "total_tokens": total_tokens,
            "prompt_tokens_details": {
                "cached_tokens": 0
            }
        }
    }

    print(
        f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Output formatted. Time taken: {time.time() - start_time:.2f} seconds")
    return output


def calculate_token_usage(model_inputs: Dict, generated_ids: torch.Tensor) -> Dict:
    """计算token使用统计
    功能：统计prompt和response的token数量
    参数：
        model_inputs: 模型输入张量
        generated_ids: 完整输出张量
    返回值：
        token统计字典
    """
    prompt_tokens = model_inputs.input_ids.shape[-1]
    completion_tokens = generated_ids.shape[-1] - prompt_tokens
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens
    }


def format_api_output(
        response: str,
        model_name: str,
        token_usage: Dict
) -> Dict:
    """构造API格式输出
    功能：生成OpenAI兼容的响应格式
    参数：
        response: 模型生成的原始响应
        model_name: 使用的模型名称
        token_usage: token统计信息
    返回值：
        标准化输出字典
    """
    timestamp = int(time.time())
    return {
        "id": f"chatcmpl-{timestamp}-{os.urandom(8).hex()}",
        "choices": [{
            "message": {
                "role": "assistant",
                "content": response
            },
            "finish_reason": "stop",
            "index": 0
        }],
        "created": timestamp,
        "model": model_name,
        "object": "chat.completion",
        "usage": token_usage
    }


def main():
    # 环境初始化,加载组件
    model, tokenizer, streamer = load_model_components("../Qwen/Qwen2.5-1.5B-Instruct")
    # 用户输入
    user_input = "Find the value of $x$ that satisfies the equation $4x+5 = 6x+7$."

    # 执行推理
    response, model_inputs, generated_ids = run_model_inference(
        model=model,
        tokenizer=tokenizer,
        streamer=None,
        prompt=user_input,
        device="cuda"
    )
    print("\n" + "=" * 40)
    print("Raw Response:")
    print(response)

    formatted_output = format_output(response, model_inputs, generated_ids)

    # 结果展示
    print("\n" + "=" * 40)
    print("Formatted Output:")
    print(json.dumps(formatted_output, indent=2, ensure_ascii=False))




if __name__ == '__main__':
    main()