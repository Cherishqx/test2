import json
import time
import random
import string
import uuid
import pdfplumber
from docx import Document
from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime, timedelta
import uuid
from flask_cors import CORS
from functools import wraps
from dotenv import load_dotenv
import pytz
from pptx import Presentation
from sqlalchemy import or_
import tempfile
from Model.ReferenceFiles import get_question_and_answer
from Model.RelativeQuestion import get_relative_question
from Model.newChat import load_model_components, run_model_inference
from Model.vlChat import load_model, call_model
from Modules.TF_IDF import extract_keywords
from Modules.Wikipedia import get_wikipedia_full_texts
from Modules.arxiv_search import ArxivSearch
from Modules.utils import replace_text
from Model.streamChat import stream_model_inference  # 导入流式推理函数
from RAG.chromaRetrieval import chromaRetrieval, insertChromaFileBySlidingWindow, delete_collection, rerank, rerank_new
from RAG.rerankerBge import LoadReranker, Reranker
from RAG.tavilyTest import tavilySearch
from prompt.prompt import get_RAG_prompt
import os
import io
from flask import request, jsonify, Response, stream_with_context
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import quote

load_dotenv()

app = Flask(__name__)
CORS(app)  # 启用 CORS

db_user = os.getenv('DB_USER')
db_password = os.getenv('DB_PASSWORD')
app.config['SQLALCHEMY_DATABASE_URI'] = f'mysql+pymysql://{db_user}:{db_password}@localhost/math_rag'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)
china_tz = pytz.timezone('Asia/Shanghai')
utc = pytz.utc

model_name = "./Qwen/Qwen2.5-1.5B-Instruct"
model, tokenizer, streamer = load_model_components("./Qwen/Qwen2.5-1.5B-Instruct")

# reranker = LoadReranker()
vl_model, processor = load_model(use_flash_attention=False)

app.config['CHROMADB_NAME'] = "Math_ShuZhi"

executor = ThreadPoolExecutor(max_workers=5)  # 控制线程数，根据需要调整

def get_china_time():
    return datetime.now()

# 数据库模型定义
class User(db.Model):
    __tablename__ = 'user'
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(50), unique=True, nullable=False)
    realname = db.Column(db.String(255))
    password = db.Column(db.String(255), nullable=False)
    email = db.Column(db.String(100), unique=True)
    created_at = db.Column(db.DateTime, default=get_china_time())
    updated_at = db.Column(db.DateTime, default=get_china_time(), onupdate=get_china_time())

class UserToken(db.Model):
    __tablename__ = 'user_token'
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id', ondelete='CASCADE'), nullable=False)
    token = db.Column(db.String(512), nullable=False)
    created_at = db.Column(db.DateTime, default=get_china_time())
    expires_at = db.Column(db.DateTime, nullable=False)
    user = db.relationship('User')

class Chat(db.Model):
    __tablename__ = 'chat'
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id', ondelete='CASCADE'), nullable=False)
    chat_name = db.Column(db.String(100))
    created_at = db.Column(db.DateTime, default=get_china_time())

class Message(db.Model):
    __tablename__ = 'message'
    id = db.Column(db.Integer, primary_key=True)
    sender_id = db.Column(db.Integer, db.ForeignKey('user.id', ondelete='CASCADE'), nullable=False)
    chat_id = db.Column(db.Integer, db.ForeignKey('chat.id', ondelete='CASCADE'), nullable=False)
    sender_identity = db.Column(db.String(255)) #是谁的消息
    content = db.Column(db.Text, nullable=False)
    sent_at = db.Column(db.DateTime, default=get_china_time())

class Knowledge(db.Model):
    __tablename__ = 'knowledge'
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, nullable=False)
    title = db.Column(db.String(255), nullable=False)
    description = db.Column(db.Text)
    chromadb_title = db.Column(db.String(512), nullable=False)
    created_at = db.Column(db.DateTime, default=get_china_time())
    updated_at = db.Column(db.DateTime, default=get_china_time(), onupdate=get_china_time())
    is_public = db.Column(db.Boolean, default=False)  # New column added

# 令牌验证装饰器
def token_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        token = request.headers.get('Authorization')
        print(token)
        if not token:
            return jsonify({'message': 'Token is missing!'}), 401

        # 提取 token，移除 "Bearer " 部分
        token = token.split(" ")[1] if "Bearer " in token else None

        user_token = UserToken.query.filter_by(token=token).first()
        if not user_token or user_token.expires_at < get_china_time():
            return jsonify({'message': 'Token is invalid or expired!'}), 401

        # 将当前用户挂载到请求中，方便后续使用
        request.user = user_token.user
        return f(*args, **kwargs)

    return decorated

# 用户注册接口
@app.route('/api/register', methods=['POST'])
def register():
    data = request.get_json()
    username = data.get('username')#用户名，必填
    email = data.get('email')#邮箱/手机号
    password = data.get('password')#密码，必填
    print(username)
    print(email)
    print(password)
    if not username or not password:
        return jsonify({'message': 'Username and password are required!'}), 400

    if User.query.filter_by(email=email).first():
        return jsonify({'message': 'Email already exists!'}), 400

    new_user = User(username=username,password=password, email=email)
    db.session.add(new_user)
    db.session.commit()
    return jsonify({'message': 'User registered successfully!'}), 201

# 用户登录接口
@app.route('/api/login', methods=['POST'])
def login():
    data = request.get_json()
    email = data.get('email')
    password = data.get('password')
    user = User.query.filter_by(email=email, password=password).first()
    if not user:
        return jsonify({'message': 'Invalid credentials!'}), 401

    # 先查询数据库是否存有该用户的token，如果有先删除再储存
    existing_token = UserToken.query.filter_by(user_id=user.id).first()
    if existing_token:
        db.session.delete(existing_token)
        db.session.commit()

    token_str = str(uuid.uuid4())
    utc_now = get_china_time()
    expires_at = utc_now + timedelta(hours=360)  # 令牌有效期 360 小时
    user_token = UserToken(user_id=user.id, token=token_str, expires_at=expires_at)
    db.session.add(user_token)
    db.session.commit()
    return jsonify({'token': token_str}), 200
#    return jsonify({'token': token_str, 'expires_at': expires_at.isoformat()}), 200

# 获取用户信息
@app.route('/api/user_info', methods=['GET'])
@token_required
def get_user_info():
    user = request.user
    return jsonify({
        'username': user.username,
        'realname': user.realname,
        'email': user.email,
        'created_at': user.created_at.isoformat(),
        'updated_at': user.updated_at.isoformat()
    }), 200

#更新用户信息
@app.route('/api/update', methods=['POST'])
@token_required
def update():
    data = request.get_json()
    old_password = data.get('old_password')
    new_password = data.get('new_password')
    username = data.get('username')

    user = request.user

    if user.password != old_password:
        return jsonify({'message': 'Old password is incorrect!'}), 401

    user.password = new_password
    user.username = username
    db.session.commit()

    return jsonify({'message': 'Password updated successfully!'}), 200

# 查询所有对话
@app.route('/api/chats', methods=['GET'])
@token_required
def get_user_chats():
    user = request.user
    chats = Chat.query.filter_by(user_id=user.id).all()
    chat_list = [{'chat_id': chat.id, 'chat_name': chat.chat_name, 'created_at': chat.created_at.isoformat()} for chat
                 in chats]

    return jsonify({'chats': chat_list}), 200

# 查询某个对话所有消息
@app.route('/api/messages/<int:chat_id>', methods=['GET'])
@token_required
def get_chat_messages(chat_id):
    user = request.user
    chat = Chat.query.filter_by(id=chat_id, user_id=user.id).first()

    if not chat:
        return jsonify({'message': 'Chat not found or does not belong to you!'}), 404

    messages = Message.query.filter_by(chat_id=chat.id).order_by(Message.sent_at).all()
    message_list = [
        {'message_id': msg.id, 'sender': msg.sender_identity, 'content': msg.content,
         'sent_at': msg.sent_at.isoformat()}
        for msg in messages
    ]

    return jsonify({'chat_id': chat_id, 'messages': message_list}), 200


# @app.route('/v1/chat/upload_image', methods=['POST'])
# @token_required
# def upload_image():
#     if 'image' not in request.files:
#         return 'No file part'
#
#     file = request.files['image']
#
#     # image = Image.open(file)
#     file_content = file.read()
#     print(file_content)
#
#     return f'File {file.filename} uploaded successfully!', 200
@app.route('/v1/chat/upload_image', methods=['POST'])
@token_required
def upload_image():
    if 'image' not in request.files:
        return 'No file part', 400

    file = request.files['image']

    if file.filename == '':
        return 'No selected file', 400

    try:
        # 创建临时文件并保存图片内容
        temp_dir = tempfile.gettempdir()
        temp_path = os.path.join(temp_dir, file.filename)
        file.save(temp_path)

        text = call_model(vl_model, processor, temp_path)

        print(f"Image saved to temporary file: {temp_path}")

        return jsonify({'text': text}), 200
    except Exception as e:
        return f'Error saving file: {str(e)}', 500


# 问答
@app.route('/v1/chat/completions', methods=['POST'])
@token_required
def ask():
    data = request.get_json()
    question = data.get('question')
    chat_id = data.get('chat_id')
    chat_name = data.get('chat_name')
    maxtokens = data.get('maxtokens')
    temperature = data.get('temperature')
    model2 = data.get('model')
    use_network = data.get('use_network', False)

    if not question:
        return jsonify({'message': 'Question is required!'}), 400

    # 查询或创建 Chat 会话
    chat = Chat.query.filter_by(id=chat_id, user_id=request.user.id).first()
    if not chat:
        chat = Chat(user_id=request.user.id, chat_name=chat_name)
        db.session.add(chat)
        db.session.commit()
        history_texts = []
    else:
        history = Message.query.filter_by(chat_id=chat.id).order_by(Message.sent_at).all()
        history_texts = [(msg.sender_identity, msg.content) for msg in history if msg.sender_identity == "user"][-5:]

    wiki_content = []
    arxiv_articles = []
    if use_network:
        tavily_search = tavilySearch(question)

        keywords = extract_keywords(question)  # 提取关键词

        wiki_content = get_wikipedia_full_texts(keywords)  # 获取 Wikipedia 内容

        arxiv_search = ArxivSearch()
        arxiv_keywords = arxiv_search.fetch_EN_keywords_from_conversation(question)  # 获取 arXiv 关键词
        xml_response = arxiv_search.search_arxiv_papers(arxiv_keywords)  # 搜索 arXiv 论文
        arxiv_articles = arxiv_search.parse_arxiv_response(xml_response)  # 解析 arXiv 响应


    if (app.config['CHROMADB_NAME'] == 'Math_ShuZhi'):
        relative_knowledge = chromaRetrieval(question)
    else:
        relative_knowledge = chromaRetrieval(question, collection_name=app.config['CHROMADB_NAME'])

    relative_knowledge_rerank = rerank_new(question, relative_knowledge)
    relative_out = json.dumps(relative_knowledge_rerank, ensure_ascii=False, indent=2)
    # 生成 AI 回答
    prompt = get_RAG_prompt(relative_knowledge_rerank, tavily_search, history_texts, question)  # 生成提示
    text_queue = stream_model_inference(model, tokenizer, prompt)  # 使用流式推理

    # 保存用户问题
    user_message = Message(sender_id=request.user.id, chat_id=chat.id, content=question, sender_identity="user")
    db.session.add(user_message)

    def generate():
        final_answer = ""

        while True:
            text = text_queue.get()
            if text == "[DONE]":
                break
            if text and text.strip() and text != '(empty)':
                replaced_text = replace_text(text)
                final_answer += replaced_text
                yield f"{replaced_text}"

        # 保存 AI 回答
        ai_message = Message(chat_id=chat.id, content=final_answer.strip(), sender_identity="system")
        db.session.add(ai_message)
        db.session.commit()

    return Response(stream_with_context(generate()), mimetype='text/event-stream', headers={
        "chat_id": str(chat.id),
        "relative_knowledge": quote(str(relative_out), encoding="utf-8"),
        "wiki_content": quote(str(wiki_content), encoding="utf-8"),
        "arxiv_articles": quote(str(arxiv_articles), encoding="utf-8"),
        "tavily_search": quote(str(tavily_search), encoding="utf-8"),
        "Access-Control-Expose-Headers": "chat_id, relative_knowledge, wiki_content, arxiv_articles, tavily_search"
    })


@app.route("/api/delete_chat", methods=["POST"])
@token_required
def delete_chat():
    data = request.json
    chat_id = data.get("chat_id")

    if not chat_id:
        return jsonify({"error": "Chat ID is required"}), 400  # 检查是否提供 chat_id

    # 查找当前用户的对话
    chat = Chat.query.filter_by(id=chat_id, user_id=request.user.id).first()

    if not chat:
        return jsonify({"error": "Chat not found or permission denied"}), 404  # 确保用户只能删除自己的对话

    try:
        db.session.delete(chat)  # 删除对话
        db.session.commit()  # 提交更改
        return jsonify({"message": "Chat deleted successfully"}), 200
    except Exception as e:
        db.session.rollback()  # 发生错误时回滚
        return jsonify({"error": f"Failed to delete chat: {str(e)}"}), 500


@app.route("/api/change_name",methods=["POST"])
@token_required
def change_name():
    data = request.json
    chat_id = data.get("chat_id")
    chat_name = data.get("chat_name")
    # 查询对应的对话
    chat = Chat.query.filter_by(id=chat_id, user_id=request.user.id).first()

    if not chat:
        return jsonify({"error": "Chat not found"}), 404  # 如果未找到对话，返回 404 错误

    chat.chat_name = chat_name  # 更新对话名称

    db.session.commit()  # 提交更改
    return jsonify({"message": "Chat name updated successfully"}), 200

@app.route("/v1/related_questions", methods=["POST"])
@token_required
def related_questions():
    data = request.get_json()
    question = data.get('question')
    chat_id = data.get('chat_id')

    if not question:
        return jsonify({'message': 'Question is required!'}), 400

    history_texts = [question]
    print(history_texts)

    conversation_history = history_texts
    prompt = get_relative_question(conversation_history)
    relative_question, model_inputs, generated_ids = run_model_inference(model, tokenizer, None, prompt)
    print(f"relative_question: {relative_question}")


    data_dict = json.loads(relative_question.strip('```json\n').strip('```'))  # Parse JSON
    relative_questions = data_dict.get('relative_questions', [])


    print(f"Relative questions: {relative_questions}")

    response = {
        "related_questions": relative_questions
    }
    return jsonify(response)


@app.route("/v1/reference_files", methods=["GET"])
@token_required
def reference_files():
    question_id = request.args.get("question_id")
    if not question_id:
        return jsonify({"error": "Missing required parameter 'question_id'"}), 400

    try:
        question_id = int(question_id)  # 确保 question_id 是整数
    except (TypeError, ValueError):
        return jsonify({"error": "Invalid 'question_id'. It must be an integer."}), 400

    try:
        question, answer,tavily_search, arxiv_articles, wiki_articles = get_question_and_answer(question_id)
    except Exception as e:
        return jsonify({"error": f"Failed to retrieve data: {str(e)}"}), 500  # 服务器内部错误

    response = {
        "reference_files": [f"File related to question {question_id}"],
        "status": "success",
        "question": question,
        "answer": answer,
        "arxiv_articles": arxiv_articles,
        "wiki_articles": wiki_articles,
        "tavily_search": tavily_search
    }
    return jsonify(response), 200

###################知识库####################
@app.route('/api/knowledge/select', methods=['POST'])
@token_required
def select():
    data = request.get_json()
    title = data.get('title')

    user = request.user

    # if title == "Math_ShuZhi":
    #     app.config['CHROMADB_NAME'] = "Math_ShuZhi"
    # else:
    #     knowledge = Knowledge.query.filter_by(user_id=user.id, title=title).first()
    #
    #     print(app.config['CHROMADB_NAME'])
    #     app.config['CHROMADB_NAME'] = knowledge.chromadb_title

    knowledge = Knowledge.query.filter(
        Knowledge.title == title,
        or_(Knowledge.user_id == user.id, Knowledge.is_public == True)
    ).order_by(Knowledge.user_id != user.id).first()

    print(app.config['CHROMADB_NAME'])
    app.config['CHROMADB_NAME'] = knowledge.chromadb_title
    print(app.config['CHROMADB_NAME'])
    return jsonify({'message': 'Knowledge updated successfully!'}), 200

@app.route('/api/upload',methods = ['POST'])
@token_required
def file_receive():
    # 获取文件对象
    file = request.files['file']
    filename = file.filename
    user = request.user

    title = request.args.get('Stock_name')
    print(f"title: {title}")

    _, ext = os.path.splitext(filename)

    text = ""
    if ext == '.txt':
        # 解析 txt 文本文件
        text = file.read().decode('utf-8', errors='ignore')  # 忽略解码错误
        print(f"text{text}")
    elif ext == '.docx':
        # 解析 Word 文档
        doc = Document(io.BytesIO(file.read()))
        text = '\n'.join([para.text for para in doc.paragraphs])
        print(f"text{text}")
    elif ext == '.pdf':
        with pdfplumber.open(file) as pdf:
            text = ""
            # 遍历每一页，提取文本
            for page in pdf.pages:
                text += page.extract_text()  # 提取页面文本
            print(f"text{text}")
    elif ext == '.pptx':
        presentation = Presentation(file)
        text = ""
        for slide in presentation.slides:
            for shape in slide.shapes:
                if hasattr(shape, "text"):
                    text += shape.text + "\n"
        print(f"text{text}")

    prefix = random.choice(string.ascii_lowercase)  # 随机选择一个小写字母
    unique_id = str(uuid.uuid4().hex)  # 生成UUID
    chromadb_title = prefix + unique_id
    print(f"unique_id: {chromadb_title}")

    insertChromaFileBySlidingWindow(text, chromadb_title)

    knowledge = Knowledge(
        user_id=user.id,
        title=title,
        chromadb_title=chromadb_title
    )
    db.session.add(knowledge)
    db.session.commit()

    return jsonify({"message": "Chat name updated successfully"}), 200

@app.route('/api/knowledge/user', methods=['GET'])
@token_required
def get_user_knowledge():
    user = request.user
    # knowledge_list = Knowledge.query.filter_by(user_id=user.id).all()

    # 获取当前用户的知识和所有 public 知识
    knowledge_list = Knowledge.query.filter(or_(Knowledge.user_id == user.id, Knowledge.is_public == True)).all()

    knowledge = Knowledge.query.filter_by(chromadb_title=app.config['CHROMADB_NAME']).first()
    print(f"knowledge: {app.config['CHROMADB_NAME']}")

    if knowledge is None:
        select_id = None  # Handle the case where no record is found, e.g., return None or handle accordingly
    else:
        select_id = knowledge.id
        print(knowledge.id)

    result = [{
        'id': k.id,
        'title': k.title,
        'description': k.description,
        'selected': 1 if k.id == select_id else 0,
    } for k in knowledge_list]

    return jsonify({'knowledge': result})

# 删除知识（通过 id）
@app.route('/api/knowledge/delete/<int:knowledge_id>', methods=['DELETE'])
@token_required
def delete_knowledge(knowledge_id):
    knowledge = Knowledge.query.get(knowledge_id)
    if not knowledge:
        return jsonify({'error': '知识项不存在'}), 404

    delete_collection(knowledge.chromadb_title)

    db.session.delete(knowledge)
    db.session.commit()
    return jsonify({'message': '删除成功'})

@app.route('/api/knowledge/update/<int:knowledge_id>', methods=['PUT'])
@token_required
def update_knowledge(knowledge_id):
    data = request.json
    new_title = data.get('title')
    # new_description = data.get('description')

    knowledge = Knowledge.query.get(knowledge_id)
    if not knowledge:
        return jsonify({'error': '知识项不存在'}), 404

    if new_title:
        knowledge.title = new_title

    db.session.commit()
    return jsonify({'message': '知识项已更新'})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
