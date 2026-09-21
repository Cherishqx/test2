import json
import os
import sys

# 修复导入路径：无论从哪个目录启动脚本，都能正确找到 Model/Modules/prompt 等模块
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(CURRENT_DIR)
sys.path.insert(0, PARENT_DIR)
sys.path.insert(0, CURRENT_DIR)

from Model.newChat import run_model_inference, load_model_components, format_output
from Modules.TF_IDF import extract_keywords
from Modules.Wikipedia import get_wikipedia_full_texts
from Modules.arxiv_search import ArxivSearch
# rerankerBge 与本文件同目录，且 sys.path[0] 已是 RAG 目录；用裸导入避免
# "RAG" 被解析为 RAG.py 自身（不是包）导致的 ModuleNotFoundError
from rerankerBge import LoadReranker, Reranker
from tavilyTest import tavilySearch
from chromaRetrieval import chromaRetrieval
from prompt.prompt import get_RAG_prompt


def rag(history, model, tokenizer, streamer, query, reranker, collection_name="Math_ShuZhi"):
    relative_knowledge = chromaRetrieval(query, collection_name)
    # chromaRetrieval 返回 dict 列表，reranker 与 prompt 拼接需要字符串
    relative_knowledge = [
        k if isinstance(k, str) else json.dumps(k, ensure_ascii=False)
        for k in relative_knowledge
    ]
    relative_knowledge_rerank = Reranker(reranker, query, relative_knowledge)

    # # tavily检索内容
    # tavily_search = tavilySearch(query)

    # # wikipedia检索内容
    # keywords = extract_keywords(query)
    # wiki_content = get_wikipedia_full_texts(keywords)

    # # arxiv检索内容
    # arxiv_search = ArxivSearch()
    # keywords = arxiv_search.fetch_EN_keywords_from_conversation(query)
    # xml_response = arxiv_search.search_arxiv_papers(keywords)
    # arxiv_articles = arxiv_search.parse_arxiv_response(xml_response)

    if history is not None:
        prompt = get_RAG_prompt(relative_knowledge_rerank, None, history, query)
    else:
        prompt = get_RAG_prompt(relative_knowledge_rerank, None, None, query)

    try:
        response, model_inputs, generated_ids = run_model_inference(model, tokenizer, streamer, prompt)
        output = format_output(response, model_inputs, generated_ids)
        entire_json = json.dumps(output, indent=4, ensure_ascii=False)
    except Exception as e:
        response = "error occurrence!"
        error_info = {
            "error": str(e),
            "message": "An error occurred during the model call."
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