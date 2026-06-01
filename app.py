import streamlit as st
import tempfile
import os
import time
from document_processor import load_document
from rag_chain import RAGQAChain

def init_session_state():
    if 'rag_chain' not in st.session_state:
        with st.spinner("正在初始化问答系统..."):
            st.session_state.rag_chain = RAGQAChain()
            st.session_state.rag_chain.load_existing_store()
            time.sleep(1)
    
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    if 'uploaded_files' not in st.session_state:
        st.session_state.uploaded_files = []

def main():
    st.set_page_config(page_title="RAG智能问答系统", page_icon=":books:", layout="wide")
    
    init_session_state()
    
    st.title("📚 RAG智能问答系统")
    
    with st.expander("⚡ 性能优化建议", expanded=True):
        st.markdown("""
        **性能优化建议：**
        - 首次加载模型需要约10-30秒，后续回答会更快
        - 建议使用较小的模型（如 llama3:8b）以获得更好的响应速度
        - 文档向量化需要调用嵌入模型，大文档可能需要较长时间
        - 建议至少16GB内存运行7B/8B参数模型
        - 流式输出已启用，您可以看到实时响应
        """)
    
    st.sidebar.title("知识库管理")
    
    with st.sidebar:
        uploaded_files = st.file_uploader(
            "上传文档",
            type=["pdf", "docx", "txt"],
            accept_multiple_files=True
        )
        
        if st.button("🔄 构建知识库"):
            if uploaded_files:
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                documents = []
                total_files = len(uploaded_files)
                
                for i, file in enumerate(uploaded_files):
                    status_text.text(f"📄 正在处理第 {i+1}/{total_files} 个文件: {file.name}")
                    progress_bar.progress((i+1)/total_files)
                    
                    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.name)[1]) as tmp:
                        tmp.write(file.read())
                        tmp_path = tmp.name
                    
                    try:
                        text = load_document(tmp_path)
                        if text:
                            documents.append(text)
                            if file.name not in st.session_state.uploaded_files:
                                st.session_state.uploaded_files.append(file.name)
                    finally:
                        os.unlink(tmp_path)
                
                status_text.text("🔢 正在向量化文档...")
                progress_bar.progress(0.9)
                
                if documents:
                    chunk_count = st.session_state.rag_chain.add_documents(documents)
                    progress_bar.progress(1.0)
                    st.success(f"✅ 知识库构建完成！共处理 {len(documents)} 个文档，生成 {chunk_count} 个文本块")
                    status_text.empty()
                    progress_bar.empty()
                else:
                    st.error("❌ 未能提取任何文档内容")
                    status_text.empty()
                    progress_bar.empty()
            else:
                st.warning("⚠️ 请先上传文档")
        
        chunk_count = st.session_state.rag_chain.get_chunk_count()
        st.info(f"📊 当前知识库文本块数量: {chunk_count}")
        st.info(f"📄 已上传文档数量: {len(st.session_state.uploaded_files)}")
        
        if st.button("🗑️ 清空对话历史"):
            st.session_state.rag_chain.clear_history()
            st.session_state.chat_history = []
            st.success("对话历史已清空")
    
    st.subheader("💬 问答交互")
    
    for chat in st.session_state.chat_history:
        with st.chat_message("user"):
            st.write(chat["question"])
        with st.chat_message("assistant"):
            st.write(chat["answer"])
    
    user_input = st.text_input("请输入您的问题:", key="question_input", placeholder="输入问题后点击提问按钮...")
    
    if st.button("🚀 提问") and user_input.strip():
        with st.spinner("🤔 正在思考中，请稍候..."):
            start_time = time.time()
            answer = st.session_state.rag_chain.ask(user_input)
            end_time = time.time()
        
        response_time = round(end_time - start_time, 2)
        
        st.session_state.chat_history.append({
            "question": user_input,
            "answer": answer
        })
        
        with st.chat_message("user"):
            st.write(user_input)
        with st.chat_message("assistant"):
            st.write(answer)
            st.caption(f"⏱️ 响应时间: {response_time} 秒")

if __name__ == "__main__":
    main()