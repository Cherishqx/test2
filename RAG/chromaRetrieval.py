import os
import sys
import time
from dotenv import load_dotenv
import cohere
sys.path.append("..")
import json
import chromadb
from sentence_transformers import SentenceTransformer
import json


def chromaRetrieval(query, collection_name="Math_ShuZhi"):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    chroma_path = os.path.join(current_dir, "..", "Math_Chromadb_data_new")
    client = chromadb.PersistentClient(chroma_path)
    collection = client.get_or_create_collection(name=collection_name)

    results = collection.query(
        query_texts=[query],
        n_results=50
    )

    # 获取原始文档列表
    raw_docs = results.get('documents', [[]])[0]

    # 将每个JSON字符串解析为字典
    parsed_docs = [json.loads(doc) for doc in raw_docs]

    return parsed_docs  # 现在返回字典列表而不是字符串列表


def insertChroma():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    chroma_path = os.path.join(current_dir, "..", "Math_Chromadb_data_new")
    client = chromadb.PersistentClient(chroma_path)

    collection_name = "Math_ShuZhi"
    collection = client.get_or_create_collection(name=collection_name)

    model = SentenceTransformer('all-MiniLM-L6-v2')

    json_file_path = ".\\Database\\questions_answers_new.json"
    with open(json_file_path, "r", encoding="utf-8") as file:
        data = json.load(file)

    for item in data:
        question = item["question"]
        answer = item["answer"]
        # 修改点：直接拼接question和answer
        chunk = f"{question} {answer}"  # 去掉了Question/Answer标签
        id = str(item["id"])
        print(chunk)

        embedding = model.encode(chunk)

        collection.add(
            embeddings=[embedding],
            metadatas=[{"source": "Math_ShuZhi_file"}],
            documents=[chunk],
            ids=[id]
        )

    print(f"数据已成功导入到集合 '{collection_name}' 中。")

def insertChroma_new():
    client = chromadb.PersistentClient(path="./Math_Chromadb_data_new")
    collection_name = "Math_ShuZhi"
    collection = client.get_or_create_collection(name=collection_name)

    model = SentenceTransformer('all-MiniLM-L6-v2')

    json_file_path = ".\\Database\\questions_answers_new.json"
    with open(json_file_path, "r", encoding="utf-8") as file:
        data = json.load(file)

    for item in data:
        # 构建结构化JSON文档
        structured_doc = {
            "question": item["question"],
            "answer": item["answer"]
        }
        # 转换为JSON字符串
        chunk = json.dumps(structured_doc, ensure_ascii=False)
        id = str(item["id"])

        embedding = model.encode(chunk)

        collection.add(
            embeddings=[embedding],
            metadatas=[{"source": "Math_ShuZhi_file"}],
            documents=[chunk],  # 存入结构化JSON字符串
            ids=[id]
        )

    print(f"数据已成功导入到集合 '{collection_name}' 中。")


# chromaRetrieval函数无需修改，因为查询结果已经是直接拼接的文本


def sliding_window_chunk(text, window_size, step_size):
    """
    使用滑动窗口方法对文本进行分块

    :param text: 需要分块的原始文本
    :param window_size: 每个块的大小（字符数）
    :param step_size: 窗口的步长（字符数）

    :return: 包含所有分块的列表
    """
    # 存储所有分块
    chunks = []

    # 当前的起始位置
    start = 0

    # 循环直到文本的末尾
    while start < len(text):
        # 计算结束位置，确保不超出文本长度
        end = start + window_size
        chunk = text[start:end]

        # 添加分块到列表
        chunks.append(chunk)

        # 移动起始位置
        start += step_size

    return chunks


def insertChromaFileBySlidingWindow(data, file_name, window_size=500, step_size=200):
    client = chromadb.PersistentClient(path="./Math_Chromadb_data_new")

    # Create a unique collection name using the file name
    collection_name = file_name.split("/")[-1].split(".")[0]
    collection = client.get_or_create_collection(name=collection_name)

    model = SentenceTransformer('all-MiniLM-L6-v2')

    # Split the text into chunks based on sliding window
    chunks = sliding_window_chunk(data, window_size, step_size)
    print(f"Processing file: {file_name}, with {len(chunks)} chunks.")

    # Add each chunk to the collection
    for idx, chunk in enumerate(chunks):
        embedding = model.encode(chunk)

        # Use a combination of file name and chunk index as the document ID
        doc_id = f"{file_name}_{idx}"

        collection.add(
            embeddings=[embedding],
            metadatas=[{"source": file_name}],
            documents=[chunk],
            ids=[doc_id]
        )

    print(f"数据已成功导入到集合 '{collection_name}' 中，文件: {file_name}，包含 {len(chunks)} 个块。")


def delete_collection(collection_name):
    # 初始化 PersistentClient
    client = chromadb.PersistentClient(path="./Math_Chromadb_data_new")

    # 尝试获取并删除指定的 collection
    try:
        # 删除指定名字的 collection
        client.delete_collection(name=collection_name)
        print(f"Collection '{collection_name}' 已成功删除。")
    except Exception as e:
        print(f"删除 Collection '{collection_name}' 失败: {e}")

def rerank(query, docs):
    results = []
    load_dotenv()
    cohere_api = os.getenv('Cohere_API')
    co = cohere.Client(api_key=cohere_api)

    response = co.rerank(
        model="rerank-v3.5",
        query=query,
        documents=docs,
        top_n=5,
    )
    documents_dict = {}

    print(response)

    # 遍历 rerank_results 列表
    for i, result in enumerate(response):
        # 假设你想要以文档的索引作为键
        key = f"Document_{i}"
        print(result)
        # 提取每个 RerankResult 对象中的 document['text'] 部分作为值
        value = result.document['text']

        # 将键值对添加到字典中
        documents_dict[key] = value

    for value in documents_dict.values():
        results.append(value)
    return (results)


def rerank_new(query, docs):
    load_dotenv()
    cohere_api = os.getenv('Cohere_API')
    co = cohere.Client(api_key=cohere_api)

    # 建立映射关系：拼接文本 -> 原始文档字典
    text_to_doc = {}
    parsed_texts = []

    for doc in docs:
        # 直接访问字典字段
        combined_text = f"{doc['question']} {doc['answer']}"
        text_to_doc[combined_text] = doc
        parsed_texts.append(combined_text)

    response = co.rerank(
        model="rerank-v3.5",
        query=query,
        documents=parsed_texts,
        top_n=5,
    )

    # 返回原始文档字典列表
    return [text_to_doc[result.document['text']] for result in response]

# 示例用法
if __name__ == '__main__':
    query = "一元二次函数"

    # 获取解析后的字典列表
    relative_knowledge = chromaRetrieval(query)

    # 转换为标准JSON字符串
    json_output = json.dumps(relative_knowledge, ensure_ascii=False, indent=2)
    print(json_output)

    # Rerank处理
    start = time.time()
    relative_knowledge_rerank = rerank_new(query, relative_knowledge)
    end = time.time()

    # 输出rerank后的JSON
    print(json.dumps(relative_knowledge_rerank, ensure_ascii=False, indent=2))
    print(f"耗时: {end - start}秒")
