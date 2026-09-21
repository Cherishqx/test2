def get_vl_prompt(img):
    prompt = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": img,
            },
            {
                "type": "text",
                "text":'''先判断数学题目图片中是否包含图形图表，提取所有文字内容，若有图片则进行分析，若没有则image为None。按照json格式返回结果，不要无关信息：
                 {"text": "提取的文字内容，使用Latex 格式输出，包括题目描述、数学表达式、单位等。","image": "识别的图形图像信息，如几何图形、函数图像、统计图表等。"}'''
            },
                        ],
                    }
                ]
    return prompt