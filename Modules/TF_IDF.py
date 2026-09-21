from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd


def extract_keywords(document, top_n=5):
    """
    使用 TF-IDF 提取单个文档的关键词。

    参数:
    - document: str，单个英文文本段。
    - top_n: int，提取的关键词数量，默认值为 5。

    返回:
    - keywords: list，包含提取的关键词。
    """
    try:
        # 检查输入类型
        if not isinstance(document, str):
            raise ValueError("Input document must be a string.")
        if not isinstance(top_n, int) or top_n <= 0:
            raise ValueError("top_n must be a positive integer.")

        # 将单个字符串转换为列表，以便使用 TfidfVectorizer
        documents = [document]

        # 使用 TfidfVectorizer 并应用内置的英文停用词库
        vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2))
        tfidf_matrix = vectorizer.fit_transform(documents)
        feature_names = vectorizer.get_feature_names_out()

        # 将 TF-IDF 矩阵转换为 DataFrame
        df_tfidf = pd.DataFrame(tfidf_matrix.toarray(), columns=feature_names)

        # 提取文档的关键词
        tfidf_scores = df_tfidf.iloc[0]
        top_keywords = tfidf_scores.nlargest(top_n).index.tolist()  # 提取 TF-IDF 得分最高的 top_n 个词

        return top_keywords

    except Exception as e:
        print(f"An error occurred: {e}")
        return []


if __name__ == "__main__":
    document = "Evaluating the safety of an autonomous vehicle (AV) depends on the behavior of surrounding agents which can be heavily influenced by factors such as environmental context and informally-defined driving etiquette."

    # 这里可以测试不同的 top_n 值
    keywords = extract_keywords(document, top_n=5)
    print("Keywords:", keywords)
