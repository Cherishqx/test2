import re

def replace_text(text: str) -> str:
    # text = "测试文本，包含/）和/)的字符组合。"
    # 使用正则表达式替换
    #print("替换前:", text)
    if not isinstance(text, str):
        return text
    pattern = r'/[（）()]'  # 匹配 /） 或 /) 或 /（ 或 /(
    result = re.sub(pattern, '$', text)
    #print("替换后:", result)
    return result
