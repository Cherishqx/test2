# -*- coding: utf-8 -*-
"""
RAG.py 自动化测试 —— 全局夹具（fixture）配置

设计说明：
1. 按需求【不使用 mock】：所有用例均加载并调用真实组件——
   Qwen2.5-0.5B-Instruct 模型、bge-reranker-base、ChromaDB(Math_ShuZhi 集合)。
2. 模型与 reranker 均为 session 级夹具，整套测试只加载一次，控制开销。
3. 若运行环境无 CUDA GPU，依赖 LLM 推理的用例自动 skip（而非报错），
   仅依赖 reranker/检索的用例仍可执行。
"""
import os
import sys

import pytest

# ------------------------------------------------------------------ 路径准备
# 保证无论从哪个目录启动 pytest，都能找到项目内各真实模块
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # 项目根目录
RAG_DIR = os.path.join(ROOT, "RAG")
TESTS_DIR = os.path.dirname(os.path.abspath(__file__))

for _p in (TESTS_DIR, RAG_DIR, ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from RAG import rag                             # noqa: E402  被测主函数（RAG/RAG.py）
from rerankerBge import LoadReranker            # noqa: E402  真实 bge-reranker 加载器
from Model.newChat import load_model_components  # noqa: E402 真实模型加载器


# ------------------------------------------------------------------ 夹具定义
@pytest.fixture(scope="session")
def qwen_components():
    """加载真实 Qwen2.5-0.5B-Instruct（session 级，只加载一次）。

    返回 (model, tokenizer, streamer) 三元组。
    无 CUDA 环境时跳过所有依赖该夹具的用例。
    """
    import torch

    if not torch.cuda.is_available():
        pytest.skip("当前环境无 CUDA GPU，跳过需要加载 Qwen 模型的用例")

    model_name = os.path.join(ROOT, "Qwen", "Qwen2.5-0.5B-Instruct")
    return load_model_components(model_name)


@pytest.fixture(scope="session")
def bge_reranker():
    """加载真实 bge-reranker-base（session 级，只加载一次）"""
    return LoadReranker()


@pytest.fixture(scope="session")
def run_rag(qwen_components, bge_reranker):
    """提供对被测函数 RAG.rag() 的快捷调用入口。

    用法：
        entire_json, response, knowledge = run_rag(
            query, history=None, collection_name="Math_ShuZhi")
    """
    model, tokenizer, streamer = qwen_components

    def _run(query, history=None, collection_name="Math_ShuZhi"):
        return rag(history, model, tokenizer, streamer, query,
                   bge_reranker, collection_name)

    return _run
