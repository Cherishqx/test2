import json

from Model.newChat import run_model_inference, load_model_components


def get_relative_question(conversation_history: list):
    example_json = """
    {
    'relative_questions':[
     'Question_1',
     'Question_2',
     'Question_3',
    ]
    }
    """
    System_Prompt = f"""
        你是一位经验丰富的数学研究者，专注于数学领域的深度研究多年，具备丰富的数学知识，熟悉数学经典著作，
        擅长解决复杂的数学问题，并且有丰富的教学和研究实践经验！
        同时你擅长判断用户意图！
        你需要根据用户当前的问题和系统给出的回答大纲，提出进一步追问的问题。
        注意理解用户问题，判断用户意图，你提出的问题需要与用户问题高度相关，并起到启发作用，帮助用户深入探索！
        用户问题: {conversation_history[-1]}
        你需要提出三个相关问题！
        你需要输出一个 json 格式的数组。
        输出例子: {example_json}
    """
    return System_Prompt

if __name__ == "__main__":
    model, tokenizer, streamer = load_model_components("../Qwen/Qwen2.5-1.5B-Instruct")

    conversation_history = ['求一元一次方程的解']
    prompt = get_relative_question(conversation_history)
    relative_question, model_inputs, generated_ids = run_model_inference(model, tokenizer, None, prompt)
    print(relative_question)
    print(type(relative_question))

    if isinstance(relative_question, dict):
        print("Relative question is a dictionary")
        relative_questions = relative_question.get('relative_questions', [])

    elif isinstance(relative_question, tuple):
        print("Relative question is a tuple")
        data = json.loads(relative_question[0])
        # 提取relative_questions
        relative_questions = data.get('relative_questions', [])

    elif isinstance(relative_question, str):
        # Handle the case where relative_question is a JSON string
        print("Relative question is a string (JSON format)")
        data_dict = json.loads(relative_question.strip('```json\n').strip('```'))  # Parse JSON
        relative_questions = data_dict.get('relative_questions', [])

    # 打印或使用提取的数据
    print(relative_questions)
