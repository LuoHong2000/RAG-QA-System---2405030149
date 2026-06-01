from langchain_community.chat_models import ChatOllama
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from vector_store import VectorStoreManager
from typing import List, Dict, Generator

class RAGQAChain:
    def __init__(self, model_name: str = "llama3:8b"):
        self.llm = ChatOllama(model=model_name, temperature=0.1, max_tokens=512)
        self.vector_store_manager = VectorStoreManager()
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
        self.chain = None
        
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", "你是一个基于知识库的智能问答助手。请严格根据提供的参考文档内容回答问题。如果文档中没有相关信息，请明确回答'文档中未找到相关答案'。不要编造答案。"),
            MessagesPlaceholder("chat_history"),
            ("human", "参考文档：\n{context}\n\n问题：{input}"),
        ])
    
    def setup_chain(self):
        retriever = self.vector_store_manager.vector_store.as_retriever(
            search_kwargs={"k": 2}
        )
        document_chain = create_stuff_documents_chain(self.llm, self.prompt)
        self.chain = create_retrieval_chain(retriever, document_chain)
    
    def add_documents(self, documents: List[str]) -> int:
        chunk_count = self.vector_store_manager.create_vector_store(documents)
        if chunk_count > 0:
            self.setup_chain()
        return chunk_count
    
    def load_existing_store(self) -> bool:
        success = self.vector_store_manager.load_vector_store()
        if success:
            self.setup_chain()
        return success
    
    def ask(self, question: str) -> str:
        if self.chain is None:
            return "知识库尚未构建，请先上传文档并构建知识库。"
        
        try:
            response = self.chain.invoke({"input": question, "chat_history": self.memory.load_memory_variables({})["chat_history"]})
            answer = response.get("answer", "").strip()
            
            if answer:
                self.memory.save_context({"input": question}, {"output": answer})
            
            if not answer or "没有找到" in answer or "未找到" in answer or not answer.strip():
                return "文档中未找到相关答案"
            
            return answer
        except Exception as e:
            return f"问答过程中出现错误: {str(e)}"
    
    def ask_stream(self, question: str) -> Generator[str, None, None]:
        if self.chain is None:
            yield "知识库尚未构建，请先上传文档并构建知识库。"
            return
        
        try:
            full_answer = ""
            for chunk in self.chain.stream({"input": question, "chat_history": self.memory.load_memory_variables({})["chat_history"]}):
                if "answer" in chunk:
                    delta = chunk["answer"]
                    full_answer += delta
                    yield delta
            
            if full_answer:
                self.memory.save_context({"input": question}, {"output": full_answer})
            
            if not full_answer.strip() or "没有找到" in full_answer or "未找到" in full_answer:
                yield "\n文档中未找到相关答案"
                
        except Exception as e:
            yield f"\n问答过程中出现错误: {str(e)}"
    
    def get_chat_history(self) -> List[Dict[str, str]]:
        messages = self.memory.load_memory_variables({}).get("chat_history", [])
        history = []
        for i in range(0, len(messages), 2):
            if i + 1 < len(messages):
                history.append({
                    "question": messages[i].content,
                    "answer": messages[i + 1].content
                })
        return history
    
    def clear_history(self):
        self.memory.clear()
    
    def get_chunk_count(self) -> int:
        return self.vector_store_manager.get_chunk_count()