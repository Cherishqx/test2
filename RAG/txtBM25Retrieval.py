import re
import jieba
from rank_bm25 import BM25Okapi


class ChunkRetriever:
    def __init__(self, documents, window_size=500, step_size=300):
        self.raw_docs = documents
        self.window_size = window_size
        self.step_size = step_size
        self.corpus, self.formatted_chunks = self._chunk_documents()
        self.bm25 = BM25Okapi(self.corpus)

    def sliding_window_chunk(self, text):
        """改进版分块方法"""
        chunks = []
        start = 0
        while start < len(text):
            end = min(start + self.window_size, len(text))
            # 确保不截断句子
            while end < len(text) and text[end] not in {'。', '！', '？', '\n'}:
                end += 1
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            start += self.step_size
        return chunks

    def _preprocess(self, text):
        """带格式保留的预处理"""
        text = re.sub(r'(?<!\\)\$.*?(?<!\\)\$', '[MATH]', text)  # 数学公式标记
        text = re.sub(r'\s+', ' ', text)
        return jieba.lcut(text)

    def _chunk_documents(self):
        corpus = []
        formatted = []
        for doc in self.raw_docs:
            chunks = self.sliding_window_chunk(doc)
            for chunk in chunks:
                processed = self._preprocess(chunk)
                corpus.append(processed)
                formatted.append(chunk)
        return corpus, formatted

    def search(self, query, top_k=50):
        query_terms = self._preprocess(query)
        scores = self.bm25.get_scores(query_terms)
        scored = sorted(zip(scores, self.formatted_chunks), key=lambda x: -x[0])
        return [doc for _, doc in scored[:top_k]]

if __name__ == '__main__':
    # 分块文档检索示例
    long_documents = [
        "这里是一个很长的文档内容...",
        "另一个需要分块的文档内容..."
    ]

    chunk_retriever = ChunkRetriever(
        documents=long_documents,
        window_size=500,  # 字符数
        step_size=300
    )
    results = chunk_retriever.search("非线性方程求解")
    print(results)
    for result in results:
        print(result)