import os
import torch
import time
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer
import json


def set_environment():
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"


def clear_gpu_memory():
    torch.cuda.empty_cache()


def load_model_and_tokenizer(model_name, device):
    start_time = time.time()
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="cuda:0"
    )
    print(
        f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Model loaded successfully. Time taken: {time.time() - start_time:.2f} seconds")

    start_time = time.time()
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    print(
        f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Tokenizer loaded successfully. Time taken: {time.time() - start_time:.2f} seconds")

    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    return model, tokenizer, streamer


def _call(model, tokenizer, streamer, prompt, device, max_new_tokens=512):
    start_time = time.time()
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Generating response...")
    messages = [
        {"role": "system",
         "content": "Please integrate natural language reasoning with programs to solve the problem above, and put your final answer within \\boxed{}."},
        {"role": "user", "content": prompt}
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    model_inputs = tokenizer([text], return_tensors="pt").to(device)

    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=max_new_tokens,
        streamer=streamer,
    )

    extract_generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    response = tokenizer.batch_decode(extract_generated_ids, skip_special_tokens=True)[0]

    print(
        f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Response generated. Time taken: {time.time() - start_time:.2f} seconds")
    return response, model_inputs, generated_ids


def format_output(response, model_inputs, generated_ids, model_name):
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


def main():
    set_environment()
    clear_gpu_memory()

    model_name = "../Qwen/Qwen2.5-1.5B-Instruct"
    device = "cuda"

    model, tokenizer, streamer = load_model_and_tokenizer(model_name, device)

    prompt = "Find the value of $x$ that satisfies the equation $4x+5 = 6x+7$."

    response, model_inputs, generated_ids = _call(model, tokenizer, streamer, prompt, device)

    output = format_output(response, model_inputs, generated_ids, model_name)

    print(json.dumps(output, indent=4, ensure_ascii=False))
    print('---' * 20)
    print(response)


if __name__ == '__main__':
    main()
