import streamlit as st
import tempfile
import os
from document_processor import load_document
from rag_chain import RAGQAChain

def init_session_state():
    if 'rag_chain' not in st.session_state:
        st.session_state.rag_chain = RAGQAChain()
        st.session_state.rag_chain.load_existing_store()
    
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    if 'uploaded_files' not in st.session_state:
        st.session_state.uploaded_files = []

def main():
    st.set_page_config(page_title="RAG智能问答系统", page_icon=":books:", layout="wide")
    
    init_session_state()
    
    st.title("📚 RAG智能问答系统")
    st.sidebar.title("知识库管理")
    
    with st.sidebar:
        uploaded_files = st.file_uploader(
            "上传文档",
            type=["pdf", "docx", "txt"],
            accept_multiple_files=True
        )
        
        if st.button("构建知识库"):
            if uploaded_files:
                with st.spinner("正在处理文档..."):
                    documents = []
                    for file in uploaded_files:
                        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.name)[1]) as tmp:
                            tmp.write(file.read())
                            tmp_path = tmp.name
                        
                        try:
                            text = load_document(tmp_path)
                            documents.append(text)
                            st.session_state.uploaded_files.append(file.name)
                        finally:
                            os.unlink(tmp_path)
                    
                    if documents:
                        chunk_count = st.session_state.rag_chain.add_documents(documents)
                        st.success(f"知识库构建完成！共处理 {len(documents)} 个文档，生成 {chunk_count} 个文本块")
                    else:
                        st.error("未能提取任何文档内容")
            else:
                st.warning("请先上传文档")
        
        chunk_count = st.session_state.rag_chain.get_chunk_count()
        st.info(f"当前知识库文本块数量: {chunk_count}")
        st.info(f"已上传文档数量: {len(st.session_state.uploaded_files)}")
        
        if st.button("清空对话历史"):
            st.session_state.rag_chain.clear_history()
            st.session_state.chat_history = []
            st.success("对话历史已清空")
    
    st.subheader("问答交互")
    
    for chat in st.session_state.chat_history:
        with st.chat_message("user"):
            st.write(chat["question"])
        with st.chat_message("assistant"):
            st.write(chat["answer"])
    
    user_input = st.text_input("请输入您的问题:", key="question_input")
    
    if st.button("提问") and user_input.strip():
        with st.spinner("正在思考..."):
            answer = st.session_state.rag_chain.ask(user_input)
        
        st.session_state.chat_history.append({
            "question": user_input,
            "answer": answer
        })
        
        with st.chat_message("user"):
            st.write(user_input)
        with st.chat_message("assistant"):
            st.write(answer)

if __name__ == "__main__":
    main()