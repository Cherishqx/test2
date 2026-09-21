from RAG.jsonBM25Retrieval import JSONRetriever
from RAG.chromaRetrieval import chromaRetrieval


def hybrid_search_rrf(query, top_k=10, k=60):
    # 获取两种检索结果
    json_retriever = JSONRetriever('../Database/questions_answers.json')

    dense_docs = chromaRetrieval(query)
    sparse_docs = [doc for _, doc in json_retriever.search(query, top_k=100)]

    # 构建排名字典
    rank_dict = {}
    for idx, doc in enumerate(dense_docs):
        rank_dict[doc] = rank_dict.get(doc, 0) + 1 / (k + idx + 1)

    for idx, doc in enumerate(sparse_docs):
        rank_dict[doc] = rank_dict.get(doc, 0) + 1 / (k + idx + 1)

    # 排序并返回
    sorted_docs = sorted(rank_dict.items(), key=lambda x: -x[1])
    return [doc for doc, _ in sorted_docs[:top_k]]



if __name__ == '__main__':
    query = "一元二次方程"