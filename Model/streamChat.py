from queue import Queue
from threading import Thread
from transformers import TextStreamer
from typing import Any

from Model.newChat import generate_chat_prompt


class FlaskStreamer(TextStreamer):
    """自定义流式处理器"""

    def __init__(self, queue: Queue, tokenizer: Any, *args, **kwargs):
        super().__init__(tokenizer, *args, **kwargs)
        self.queue = queue  # 线程安全队列

    def on_finalized_text(self, text: str, stream_end: bool = False):
        """重写输出方法"""
        self.queue.put(text)  # 将生成的文本块放入队列


def stream_model_inference(
        model: Any,
        tokenizer: Any,
        prompt: str,
        device: str = "cuda",
        max_new_tokens: int = 1024
) -> Queue:
    """流式推理函数
    返回：包含生成文本块的队列
    """
    # 创建共享队列
    text_queue = Queue()
    # 创建自定义流式处理器
    streamer = FlaskStreamer(
        queue=text_queue,
        tokenizer=tokenizer,
        skip_prompt=True,
        skip_special_tokens=True
    )

    # 在独立线程中运行生成
    def generate_thread():
        messages = generate_chat_prompt(prompt)
        formatted_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        model_inputs = tokenizer([formatted_text], return_tensors="pt").to(device)

        model.generate(
            **model_inputs,
            max_new_tokens=max_new_tokens,
            streamer=streamer,
        )
        text_queue.put("[DONE]")  # 结束标记

    Thread(target=generate_thread).start()
    return text_queue
