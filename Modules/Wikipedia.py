import requests
from Modules.TF_IDF import extract_keywords

"""
wikipedia禁止中国大陆访问
"""


# def fetch_CN_keywords_from_conversation(query, model, tokenizer, streamer) -> list:
#     prompt = f"""
#     根据问题，提取2-3个关键词，用于后续的 Wikipedia 查询。
#     返回格式仅包含关键词，每个关键词不超过五个汉字。
#     关键词应简洁明了，应易于理解，不要使用复杂的数学词汇。
#     生成的关键词必须简洁！
#     格式如下：
#     关键词1, 关键词2, 关键词3
#     以下为问题：
#     {query}
#     输出一定不要有引号！！
#     """
#     keywords = run_model_inference(model, tokenizer, streamer, prompt)[0]
#
#     print("1Keywords: ", keywords)
#     return keywords


def get_wikipedia_full_text(page_title: str, lang: str = "zh"):
    # 构建请求 URL 和参数
    url = f"https://{lang}.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "titles": page_title,
        "prop": "revisions|info",
        "rvprop": "content",
        "rvslots": "main",
        "inprop": "url",
        "redirects": 1,
        "format": "json"
    }

    try:
        # 发送请求并检查是否成功
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()

        # 初始化结果
        result = {"title": "页面未找到或无法获取内容", "url": "", "content": ""}

        # 解析返回的数据
        if "query" in data and "pages" in data["query"]:
            pages = data["query"]["pages"]
            for page_id, page in pages.items():
                if "revisions" in page:
                    full_text = page["revisions"][0]["slots"]["main"]["*"]
                    page_title = page.get("title", "未知标题")
                    page_url = page.get("fullurl", "未知 URL")
                    result = {
                        "title": page_title,
                        "url": page_url,
                        "content": full_text
                    }
                    break

        return result

    except requests.RequestException as e:  # 捕获网络请求相关异常
        return {}
    except ValueError as e:  # 捕获 JSON 解析异常
        return {}
    except Exception as e:  # 捕获其他异常
        return {}


def get_wikipedia_full_texts(page_titles: list, lang: str = "zh"):
    """
    获取多个维基百科页面的全文内容。

    参数:
        page_titles (list): 要获取的维基百科页面标题列表。
        lang (str): 维基百科的语言代码（默认为中文 "zh"）。

    返回:
        list: 包含每个页面的标题、URL 和内容的字典列表。
    """
    results = []
    for title in page_titles:
        result = get_wikipedia_full_text(title, lang)
        if result:  # 如果获取到有效内容
            results.append(result)
    return results


# 示例用法
if __name__ == '__main__':
    query = "余弦函数"
    keywords = extract_keywords(query)

    # 如果没有提取到有效的关键词，返回空列表
    if not keywords:
        keywords = []

    content = get_wikipedia_full_texts(keywords)
    print(content)
