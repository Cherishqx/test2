import re

from modelscope import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
import importlib
import os

from modelscope import AutoTokenizer, AutoProcessor
# from transformers import Qwen2VLForConditionalGeneration
# from modelscope.models.nlp.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info
import torch
import sys

# from prompt.vlprompt import get_vl_prompt

current_dir = os.path.dirname(os.path.abspath(__file__))
prompt_path = os.path.join(current_dir, "..", "prompt", "vlprompt.py")
# Dynamically load the get_vl_prompt function
spec = importlib.util.spec_from_file_location("vlprompt", prompt_path)
vlprompt = importlib.util.module_from_spec(spec)
spec.loader.exec_module(vlprompt)
get_vl_prompt = vlprompt.get_vl_prompt


def load_model(use_flash_attention=False):
    """
    加载模型和处理器

    Args:
        use_flash_attention (bool): 是否使用 Flash Attention

    Returns:
        tuple: 模型和处理器
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_dir, "..", "Qwen", "Qwen2.5-VL-3B-Instruct")
    # model_path = "../Qwen/Qwen2.5-VL-3B-Instruct"

    try:
        if use_flash_attention:
            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16,
                attn_implementation="flash_attention_2",
                device_map="auto"
            )
        else:
            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_path, torch_dtype="auto", device_map="auto"
            )
    except ImportError:
        print("Flash Attention is not available. Falling back to default attention implementation.")
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path, torch_dtype="auto", device_map="auto"
        )

    min_pixels = 64 * 28 * 28
    max_pixels = 512 * 28 * 28
    processor = AutoProcessor.from_pretrained(model_path, min_pixels=min_pixels, max_pixels=max_pixels)

    return model, processor


def call_model(model, processor, img_path, max_new_tokens=512):
    """
    调用模型进行推理

    Args:
        model: 已加载的模型
        processor: 已加载的处理器
        img_path (str): 图片的路径
        max_new_tokens (int): 最大生成的新 token 数量

    Returns:
        list: 生成的文本
    """
    # 读取图片并生成提示词
    messages = get_vl_prompt(img_path)

    # 准备推理输入
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to("cuda")

    # 推理生成输出
    generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )

    # 取第一个生成结果（如果你只关心第一条）
    output_text = output_text[0] if isinstance(output_text, list) and output_text else ""

    # 用正则从字符串里提取
    match = re.search(r'"text"\s*:\s*"([^"]*)"', output_text)
    if match:
        result = match.group(1)
    else:
        # 如果没匹配到，就返回原始字符串
        result = output_text

    return result


# 示例用法
if __name__ == "__main__":

    current_dir = os.path.dirname(os.path.abspath(__file__))
    img_path = os.path.join(current_dir, "..","DataBase", "imgs", "latex.png")

    # img_path = "../DataBase/imgs/latex.jpg"

    # 加载模型和处理器
    model, processor = load_model(use_flash_attention=False)

    # 调用模型进行推理
    output_text = call_model(model, processor, img_path)

    # 打印结果
    print(output_text)
