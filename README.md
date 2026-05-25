# RAG智能问答系统

基于本地知识库的RAG（检索增强生成）智能问答系统，使用Ollama本地大模型、LangChain框架和Streamlit构建。

## 功能特点

- 📚 支持PDF、DOCX、TXT多种文档格式
- 🔍 基于Chroma向量数据库的智能检索
- 💬 支持多轮对话，具有会话记忆功能
- 🚀 本地化部署，无需联网即可运行
- 📊 可视化Web界面，操作简单直观

## 技术栈

- **框架**: LangChain
- **前端**: Streamlit
- **向量数据库**: Chroma
- **大模型**: Ollama (DeepSeek-R1)
- **嵌入模型**: nomic-embed-text

## 环境要求

- Python 3.10+
- Ollama
- 至少8GB内存（建议16GB以上）

## 安装步骤

### 1. 安装Ollama

访问 [Ollama官网](https://ollama.com/) 下载并安装Ollama。

### 2. 下载模型

```bash
ollama pull deepseek-r1:7b
ollama pull nomic-embed-text
```

### 3. 创建虚拟环境

```bash
python -m venv venv
```

### 4. 激活虚拟环境

**Windows**:
```bash
venv\Scripts\activate
```

**Linux/Mac**:
```bash
source venv/bin/activate
```

### 5. 安装依赖

```bash
pip install -r requirements.txt
```

## 使用说明

### 运行Web应用

```bash
streamlit run app.py
```

### 使用步骤

1. 在浏览器中打开显示的URL（通常是 http://localhost:8501）
2. 点击左侧"上传文档"按钮，选择PDF、DOCX或TXT文件
3. 点击"构建知识库"按钮，等待文档处理完成
4. 在问答交互区输入问题，点击"提问"按钮获取答案
5. 支持多轮对话，可以继续提问

### 命令行测试

```bash
python cli_rag.py
```

## RAG流程说明

1. **文档加载**: 读取本地PDF/DOCX/TXT文档
2. **文本分块**: 使用RecursiveCharacterTextSplitter进行分块（chunk_size=1000, chunk_overlap=200）
3. **向量化**: 使用Ollama的nomic-embed-text模型生成嵌入向量
4. **存储**: 将向量存入Chroma向量数据库
5. **检索**: 接收用户查询，进行相似性检索，返回最相关的3个文本块
6. **生成**: 将检索结果作为上下文，调用大模型生成回答

## 项目结构

```
├── app.py              # Streamlit Web应用主入口
├── cli_rag.py          # 命令行版本测试脚本
├── document_processor.py # 文档处理模块
├── vector_store.py     # 向量存储管理模块
├── rag_chain.py        # RAG问答链模块
├── test_ollama.py      # Ollama API测试脚本
├── requirements.txt    # 依赖列表
├── .gitignore          # Git忽略配置
└── documents/          # 示例文档目录
    ├── attention_mechanism.txt
    ├── bert.txt
    ├── nlp_intro.txt
    ├── transformer.txt
    └── word_embedding.txt
```

## 已知问题与改进方向

- 大模型响应速度较慢，建议使用性能更好的硬件
- 文档解析对复杂格式的PDF支持有限
- 可添加文档预览和管理功能
- 可添加夜间模式
- 可支持更多文档格式（如Markdown）

## License

MIT License