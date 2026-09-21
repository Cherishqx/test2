from flask import Flask, Response, request
from flask_cors import CORS
from streamChat import stream_model_inference

app = Flask(__name__)
CORS(app)  # 允许跨域


@app.route('/api/stream', methods=['POST'])

def stream_api(model, tokenizer):
    # 获取请求参数
    data = request.json
    prompt = data.get("prompt", "")

    # 执行流式推理
    queue = stream_model_inference(
        model=model,  # 预加载的模型对象
        tokenizer=tokenizer,  # 预加载的分词器
        prompt=prompt
    )

    # 流式响应生成器
    def event_stream():
        while True:
            text = queue.get()
            if text == "[DONE]":
                break
            yield f"data: {json.dumps({'text': text})}\n\n"

    return Response(event_stream(), mimetype="text/event-stream")