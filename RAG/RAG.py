import json
import os
import re
import sys

# 修复导入路径：无论从哪个目录启动脚本，都能正确找到 Model/Modules/prompt 等模块
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(CURRENT_DIR)
sys.path.insert(0, PARENT_DIR)
sys.path.insert(0, CURRENT_DIR)

from Model.newChat import run_model_inference, load_model_components, format_output
# rerankerBge 与本文件同目录，且 sys.path[0] 已是 RAG 目录；用裸导入避免
# "RAG" 被解析为 RAG.py 自身（不是包）导致的 ModuleNotFoundError
from rerankerBge import LoadReranker, Reranker
from chromaRetrieval import chromaRetrieval
from prompt.prompt import get_RAG_prompt


# ---------------------- 安全性/鲁棒性加固配置（由测试套件驱动） ----------------------
MAX_QUERY_CHARS = 8000   # 查询长度上限：超长输入主动截断，防止提示词膨胀与 DoS
MAX_HISTORY_TURNS = 50   # 对话历史上限：仅保留最近 N 轮，防止上下文溢出
# 集合名白名单：3~63 位 [A-Za-z0-9._-]，首尾为字母数字（与 chromadb 命名规则对齐）
_COLLECTION_NAME_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{1,61}[A-Za-z0-9]")

# 明确的角色覆盖命令：不单独按 DAN 等名词拦截，避免把普通概念提问当成攻击。
_ROLE_OVERRIDE_RE = re.compile(
    r"(?:进入|切换到|启用).{0,16}(?:DAN|无限制|无约束)模式"
    r"|(?:忽略|无视|绕过|解除|取消).{0,16}(?:指令|规则|限制)"
    r"|(?:不再受|不受).{0,16}(?:规则|限制)"
    r"|\b(?:enter|enable|switch\s+to)\s+(?:the\s+)?(?:DAN|unrestricted)\s+mode\b"
    r"|\b(?:ignore|bypass|remove|disable)\b.{0,40}\b(?:instructions|rules|restrictions)\b",
    re.IGNORECASE,
)
ROLE_OVERRIDE_REFUSAL = "我不能改变助手身份或解除规则限制。请提出具体的数学问题。"


def _guard_role_override(query, response):
    """对已知角色覆盖命令强制拒绝；不把小模型的提示词遵循当作安全保证。

    此启发式只覆盖明确命令，可能误拦引用此类命令的文本，不保证识别所有攻击。
    """
    if _ROLE_OVERRIDE_RE.search(query):
        return ROLE_OVERRIDE_REFUSAL
    return response


def _sanitize_input(history, query, collection_name):
    """输入加固：类型规范化 + 长度截断 + 集合名白名单校验。

    - query 为 None → 规范化为空字符串（走既有空输入路径，而非裸抛 TypeError）；
    - query 非字符串（int/float 等）→ 安全转为字符串；
    - 超长 query / history → 截断，保证提示词规模可控；
    - 非法集合名 → 快速失败（抛 ValueError），不触达数据层。
    返回 (history, query)。
    """
    if query is None:
        query = ""
    elif not isinstance(query, str):
        query = str(query)
    if len(query) > MAX_QUERY_CHARS:
        query = query[:MAX_QUERY_CHARS] + "……[输入过长，已自动截断]"

    if isinstance(history, list) and len(history) > MAX_HISTORY_TURNS:
        history = history[-MAX_HISTORY_TURNS:]

    if not _COLLECTION_NAME_RE.fullmatch(collection_name or ""):
        raise ValueError(
            "Illegal collection name (3-63 chars of [A-Za-z0-9._-], "
            f"start/end with alphanumeric): {collection_name!r}")
    return history, query


def _safe_error_text(exc):
    """生成脱敏后的错误描述：抹去本机路径/依赖目录等敏感信息，避免信息泄露。"""
    text = f"{type(exc).__name__}: {exc}"
    text = re.sub(r"[A-Za-z]:\\[^\s\"']+", "[路径已隐藏]", text)  # Windows 路径
    text = re.sub(r"(?:/home|/Users|/usr|/opt|/var|/mnt)/[^\s\"']+", "[路径已隐藏]", text)
    text = text.replace("site-packages", "[依赖目录]")
    return text[:300]


def rag(history, model, tokenizer, streamer, query, reranker, collection_name="Math_ShuZhi"):
    # 输入加固（安全性/鲁棒性优化）：类型规范化、长度截断、集合名白名单校验
    history, query = _sanitize_input(history, query, collection_name)

    relative_knowledge_rerank = []
    try:
        # 检索与重排同样纳入异常保护：任一环节失败都走受控错误路径，不再裸抛
        relative_knowledge = chromaRetrieval(query, collection_name)
        # chromaRetrieval 返回 dict 列表，reranker 与 prompt 拼接需要字符串
        relative_knowledge = [
            k if isinstance(k, str) else json.dumps(k, ensure_ascii=False)
            for k in relative_knowledge
        ]
        relative_knowledge_rerank = Reranker(reranker, query, relative_knowledge)

        if history is not None:
            prompt = get_RAG_prompt(relative_knowledge_rerank, None, history, query)
        else:
            prompt = get_RAG_prompt(relative_knowledge_rerank, None, None, query)

        # 被拦截请求不流式暴露原始生成内容，避免最终替换前先把不安全回答打印出去。
        inference_streamer = None if _ROLE_OVERRIDE_RE.search(query) else streamer
        response, model_inputs, generated_ids = run_model_inference(model, tokenizer, inference_streamer, prompt)
        response = _guard_role_override(query, response)
        output = format_output(response, model_inputs, generated_ids)
        entire_json = json.dumps(output, indent=4, ensure_ascii=False)
    except Exception as e:
        response = "error occurrence!"
        error_info = {
            "error": _safe_error_text(e),  # 脱敏：不泄露内部路径/环境细节
            "message": "An error occurred during the RAG pipeline."
        }
        entire_json = json.dumps(error_info, indent=4, ensure_ascii=False)
    return entire_json, response, relative_knowledge_rerank


if __name__ == '__main__':
    qwen_root = os.path.join(PARENT_DIR, "Qwen")
    model_name = os.path.join(qwen_root, "Qwen2.5-0.5B-Instruct")
    model, tokenizer, streamer = load_model_components(model_name)
    reranker = LoadReranker()

    query = "二次方程"
    history = []
    entire_json, response, relative_knowledge_rerank = rag(
        history, model, tokenizer, streamer, query, reranker
    )
    print(entire_json)
    print(response)
    print(relative_knowledge_rerank)
