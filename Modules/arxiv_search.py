import sys
from Modules.TF_IDF import extract_keywords
import requests
from xml.etree import ElementTree as ET
import re
from deep_translator import GoogleTranslator

sys.path.append('..')


# def simplify_keywords(input_text):
#     # 去掉 LaTeX 格式（如 \boxed{}）
#     cleaned_text = re.sub(r"\\boxed\{", "", input_text)  # 去掉 \boxed{
#     cleaned_text = re.sub(r"\}", "", cleaned_text)       # 去掉 }
#     cleaned_text = cleaned_text.strip()                  # 去掉首尾空格
#     return cleaned_text


def is_english(string):
    return bool(re.search("[a-zA-Z]", string))


def translate_to_english(string):
    try:
        translation = GoogleTranslator(source='auto', target='en').translate(string)
        return translation
    except Exception as e:
        print(f"翻译失败：{e}")
        return string  # 如果翻译失败，返回原始字符串


def process_string(input_string):
    try:
        if is_english(input_string):
            return input_string
        else:
            return translate_to_english(input_string)
    except Exception as e:
        print(f"处理字符串时发生错误：{e}")
        return input_string  # 如果处理失败，返回原始字符串


class ArxivSearch:

    # def fetch_EN_keywords_from_conversation(self, query, model, tokenizer, streamer) -> list:
    #     prompt = f"""
    #     根据问题，提取2-3个关键词，用于后续的 arXiv 查询。
    #     返回格式仅包含关键词，每个关键词不超过两个英文单词。
    #     关键词应简洁明了，应易于理解，不要使用复杂的数学词汇。
    #     生成的英文单词必须简洁！
    #     格式如下：
    #     keyword1, keyword2, keyword3
    #     以下为问题：
    #     注意这里返回的关键词必须是英文！
    #     输出的关键词必须是英文！
    #     {query}
    #     输出不要有引号
    #     """
    #     keywords = run_model_inference(model, tokenizer, streamer, prompt)[0]
    #
    #     # 去除latex格式（如果输出一切正常则不需要去除特殊格式）
    #     keywords = simplify_keywords(keywords)
    #
    #     print("Keywords: ", keywords)
    #     return [kw.strip() for kw in keywords.strip().split(',')]

    def fetch_EN_keywords_from_conversation(self, query):
        try:
            query = process_string(query)
            keywords = extract_keywords(query)
            return keywords
        except Exception as e:
            print(f"提取关键词失败：{e}")
            return []

    def search_arxiv_papers(self, keywords: list) -> str:
        """
        使用关键词搜索 arXiv 论文。
        """
        try:
            base_url = "http://export.arxiv.org/api/query?"
            search_query = '+OR+'.join([f'all:{keyword}' for keyword in keywords])
            query = f"search_query={search_query}&start=0&max_results=20"
            response = requests.get(base_url + query)
            response.raise_for_status()  # 检查请求是否成功
            return response.text
        except requests.RequestException as e:
            print(f"请求 arXiv 时发生错误：{e}")
            return ""
        except Exception as e:
            print(f"搜索 arXiv 论文时发生未知错误：{e}")
            return ""

    def parse_arxiv_response(self, xml_response: str) -> list:
        """
        解析 arXiv 的 XML 响应，提取论文信息。
        """
        try:
            root = ET.fromstring(xml_response)
            ns = {'atom': 'http://www.w3.org/2005/Atom'}

            articles = []
            for entry in root.findall('atom:entry', ns)[:10]:
                title = entry.find('atom:title', ns).text
                url = entry.find('atom:id', ns).text
                summary = entry.find('atom:summary', ns).text
                summary = summary.split('.')[0] + '.' if '.' in summary else summary
                keywords = extract_keywords(summary)
                articles.append({
                    'title': title,
                    'url': url,
                    'keywords': keywords,
                    'summary': summary
                })
            return articles
        except ET.ParseError as e:
            print(f"XML解析失败：{e}")
            return []
        except Exception as e:
            print(f"解析 arXiv 响应时发生未知错误：{e}")
            return []


# 示例用法
if __name__ == '__main__':
    query = "高次方程的解法"

    arxiv_search = ArxivSearch()
    keywords = arxiv_search.fetch_EN_keywords_from_conversation(query)
    print(keywords)
    xml_response = arxiv_search.search_arxiv_papers(keywords)
    articles = arxiv_search.parse_arxiv_response(xml_response)
    print(articles)
