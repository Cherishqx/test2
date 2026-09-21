import json
import re
import jieba
from rank_bm25 import BM25Okapi

class JSONRetriever:
    def __init__(self, json_path):
        self.data = self._load_json(json_path)
        self.corpus, self.formatted_chunks = self._process_entries()
        self.bm25 = BM25Okapi(self.corpus)

    @staticmethod
    def _load_json(path):
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def _preprocess(self, text):
        """轻量级预处理保留原始格式"""
        text = re.sub(r'\s+', ' ', text)  # 合并多余空格
        return jieba.lcut(text.strip())

    def _process_entries(self):
        corpus = []
        formatted = []
        for item in self.data:
            # 直接拼接question和answer为完整段落
            full_text = f"{item['question']} {item['answer']}"
            # 预处理用于检索
            processed = self._preprocess(full_text)
            corpus.append(processed)
            formatted.append(full_text)
        return corpus, formatted

    def search(self, query, top_k=50):
        query_terms = self._preprocess(query)
        scores = self.bm25.get_scores(query_terms)
        scored = sorted(zip(scores, self.formatted_chunks), key=lambda x: -x[0])
        return [doc for _, doc in scored[:top_k]]

if __name__ == '__main__':
    # 原始JSON数据检索
    json_retriever = JSONRetriever('../Database/questions_answers.json')
    results = json_retriever.search("分式方程增根")
    print(results)
    for result in results:
        print(result)
        print('-' * 50)