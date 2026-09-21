# 数学 RAG 智能问答系统

基于 **Qwen2.5-0.5B-Instruct + ChromaDB 向量检索 + BGE-Reranker 重排序** 的数学问答 RAG 系统。

> 已在 Windows 11 + RTX 3060 (6GB) + CUDA 12.7 驱动 环境下验证可行。
> 全部磁盘占用约 **3.5GB**（模型 ~2.1 GB + Python 环境 ~1.5 GB）。

## 1. 项目结构与原理

```
test2/
├── RAG/
│   ├── RAG.py                  # 主入口：检索 → 重排 → LLM 生成
│   ├── chromaRetrieval.py      # ChromaDB 向量召回（Top 50）
│   ├── rerankerBge.py          # bge-reranker-v2-m3 重排序（取 Top 5）
│   ├── jsonBM25Retrieval.py    # BM25 检索（备用）
│   └── tavilyTest.py           # Tavily 联网搜索（备用）
├── Model/
│   └── newChat.py              # Qwen2.5 模型加载与推理
├── Modules/                    # TF-IDF / Wikipedia / arXiv 检索
├── prompt/
│   └── prompt.py               # RAG 提示词模板
├── Math_Chromadb_data_new/     # 数学知识点向量库（已内置，免重建）
├── Database/
│   └── questions_answers.json  # 原始知识数据
├── Qwen/                       # ★ 模型目录（首次使用需下载，见第 5 步）
│   ├── Qwen2.5-0.5B-Instruct/  #   生成模型 ~1.0 GB
│   └── bge-reranker-base/     #   重排序模型 ~1.1 GB
├── requirements.txt            # Python 依赖清单
└── README.md
```

**执行流程**：`用户问题` → ChromaDB 向量召回 Top50 → BGE 重排序取 Top5 → 拼接进 Prompt → Qwen2.5 生成回答。

## 2. 环境要求

| 项目     | 要求                                                |
| -------- | --------------------------------------------------- |
| 操作系统 | Windows 10/11（Linux/macOS 同理）                   |
| GPU      | NVIDIA 显卡，显存 ≥ 6 GB（实测 RTX 3060 6GB 可跑） |
| 显卡驱动 | 支持 CUDA 12.1 及以上                               |
| 包管理器 | Miniconda / Anaconda                                |
| Python   | 3.10（在 conda 环境内指定）                         |

## 3. 创建 conda 环境并安装依赖

### 3.1 创建环境（Python 3.10）

```powershell
conda create -n rag python=3.10 -y
conda activate rag
```

### 3.2 安装PyTorch

```powershell
# 方式 A：PyTorch
conda install pytorch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 pytorch-cuda=12.1 -c pytorch -c nvidia
```

### 3.3 安装其余 Python 依赖

```powershell
pip install -r requirements.txt
```

## 5. 下载模型

共需下载 2 个模型到 `Qwen/` 目录（目录名必须一致，代码按此路径加载）：

```powershell
pip install modelscope -i https://pypi.tuna.tsinghua.edu.cn/simple

modelscope download --model Qwen/Qwen2.5-0.5B-Instruct --local_dir Qwen/Qwen2.5-0.5B-Instruct
# modelscope download --model BAAI/bge-reranker-v2-m3 --local_dir Qwen/bge-reranker-v2-m3
modelscope download --model BAAI/bge-reranker-base --local_dir Qwen/bge-reranker-base
```

### 5.3 ChromaDB 内置 ONNX 嵌入模型（首次运行自动下载）

`chromaRetrieval.py` 查询时，ChromaDB 默认嵌入函数会**首次自动**下载 ONNX 版
`all-MiniLM-L6-v2`（~80 MB）到缓存目录：

```
C:\Users\<用户名>\.cache\chroma\onnx_models\all-MiniLM-L6-v2\
```

下载源为 `https://chroma-onnx-models.s3.amazonaws.com/all-MiniLM-L6-v2/onnx.tar.gz`。
若网络不通导致首次查询卡住/报错，可手动下载该文件放到上述目录中（保持文件名
`onnx.tar.gz`，无需解压），程序会自动识别并解压。

## 6. 运行

```powershell
conda activate rag
cd d:\project\code\test2\RAG
python RAG.py
```

**正常输出**依次为：

1. 环境初始化与模型加载日志（`Components loaded in xx s`）
2. reranker 加载 + 重排序 Top5 知识片段（`Knowledge: {...}` × 5）
3. 模型流式生成的回答
4. OpenAI 格式的 JSON（`chat.completion`）与回答文本
