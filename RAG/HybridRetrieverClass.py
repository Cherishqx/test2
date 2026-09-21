import concurrent.futures

from typing import Union

from chromaRetrieval import chromaRetrieval, rerank
from jsonBM25Retrieval import JSONRetriever
from txtBM25Retrieval import ChunkRetriever


class HybridRetriever:
    def __init__(self,
                 json_retriever: JSONRetriever,
                 chunk_retriever: ChunkRetriever = None,
                 use_chunk: bool = False):
        """
        :param json_retriever: JSON格式检索器实例
        :param chunk_retriever: 分块文档检索器实例（可选）
        :param use_chunk: 是否使用分块检索模式
        """
        self.json_retriever = json_retriever
        self.chunk_retriever = chunk_retriever
        self.use_chunk = use_chunk

    def _parallel_search(self, query: str) -> tuple:
        """并行执行两种检索"""
        with concurrent.futures.ThreadPoolExecutor() as executor:
            # 提交稠密检索任务
            dense_future = executor.submit(chromaRetrieval, query)

            # 提交稀疏检索任务
            if self.use_chunk and self.chunk_retriever:
                sparse_future = executor.submit(self.chunk_retriever.search, query, 100)
            else:
                sparse_future = executor.submit(self.json_retriever.search, query, 100)

            return dense_future.result(), sparse_future.result()

    def hybrid_search(self, query: str, top_k: int = 10) -> list:
        """
        混合检索入口
        :param query: 查询文本
        :param top_k: 返回结果数量
        :return: 格式化的结果列表
        """
        # 并行获取结果
        dense_results, sparse_results = self._parallel_search(query)

        # 处理稀疏检索结果格式
        if self.use_chunk:
            # 分块模式直接返回文档列表
            sparse_docs = sparse_results
        else:
            # JSON模式返回的是纯文档列表（已去分数）
            sparse_docs = sparse_results

        # 合并结果（去重+排序）
        seen = set()
        combined = []

        # 优先保留稠密检索结果
        for doc in dense_results:
            if doc not in seen:
                seen.add(doc)
                combined.append(("[DENSE]", doc))

        # 补充稀疏检索结果
        for doc in sparse_docs:
            if doc not in seen:
                seen.add(doc)
                combined.append(("[SPARSE]", doc))

        return combined[:top_k]

if __name__ == '__main__':
    # json检索
    json_retriever = JSONRetriever('../Database/questions_answers.json')
    hybrid = HybridRetriever(json_retriever, use_chunk=False)

    # 执行混合检索
    query = "分式方程增根"
    results = hybrid.hybrid_search(query, top_k=100)
    results = [item[1] for item in results]
    print(results)
    relative_knowledge_rerank = rerank(query, results)
    # 打印结果
    print(relative_knowledge_rerank)


    # chunk检索
    long_documents = [
        "这里是一个很长的文档内容...",
        "另一个需要分块的文档内容..."
    ]
    chunk_retriever = ChunkRetriever(long_documents, 500, 300)
    hybrid = HybridRetriever(None, chunk_retriever, use_chunk=True)

    # 执行混合检索
    results = hybrid.hybrid_search("非线性方程求解", top_k=100)
    results = [item[1] for item in results]
    relative_knowledge_rerank = rerank(query, results)
    # 打印结果
    print(relative_knowledge_rerank)

