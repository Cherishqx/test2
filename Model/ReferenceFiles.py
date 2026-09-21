from Modules.TF_IDF import extract_keywords
from Modules.Wikipedia import get_wikipedia_full_texts
from Modules.arxiv_search import ArxivSearch
import json

from RAG.tavilyTest import tavilySearch


def get_question_and_answer(id):
    try:
        with open('./Database/questions_answers.json', 'r', encoding='utf-8') as file:
            data = json.load(file)
    except FileNotFoundError:
        return "Error: JSON file not found.", None
    except json.JSONDecodeError:
        return "Error: JSON file is not valid.", None

    for item in data:
        if item.get('id') == id:
            question = item.get('question')
            answer = item.get('answer')

            # tavily检索内容
            tavily_search = tavilySearch(question)

            # arxiv检索
            arxiv_search = ArxivSearch()
            keywords = arxiv_search.fetch_EN_keywords_from_conversation(question)
            xml_response = arxiv_search.search_arxiv_papers(keywords)
            arxiv_articles = arxiv_search.parse_arxiv_response(xml_response)

            # wiki检索
            keywords = extract_keywords(question)
            wiki_articles = get_wikipedia_full_texts(keywords)
            return question, answer, tavily_search, arxiv_articles, wiki_articles

    print("111")

    return "No matching ID found.", None


# 示例调用
if __name__ == "__main__":

    id_to_search = int(input("Enter the ID to search: "))
    question, answer, tavily_search, arxiv_articles, wiki_articles = get_question_and_answer(id_to_search)
    print(question)
    print(answer)
    print(tavily_search)
    print(arxiv_articles)
    print(wiki_articles)
