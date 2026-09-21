import os
from dotenv import load_dotenv
from tavily import TavilyClient


def tavilySearch(query):
    load_dotenv()
    tavily_api = os.getenv('Tavily_API')
    tavily_client = TavilyClient(api_key=tavily_api)
    response = tavily_client.search(query)
    return response


if __name__ =="__main__":
    query = "一元二次方程"
    print(tavilySearch(query))
