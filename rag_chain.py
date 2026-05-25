from langchain_community.llms import Ollama
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from vector_store import VectorStoreManager
from typing import List, Dict

class RAGQAChain:
    def __init__(self, model_name: str = "deepseek-r1:7b"):
        self.llm = Ollama(model=model_name)
        self.vector_store_manager = VectorStoreManager()
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="answer"
        )
        self.chain = None
        
        template = """
        你是一个基于知识库的智能问答助手。
        请严格根据提供的参考文档内容回答问题。
        如果文档中没有相关信息，请明确回答"文档中未找到相关答案"。
        不要编造答案。

        参考文档：
        {context}

        问题：{question}

        回答：
        """
        self.prompt = PromptTemplate(
            template=template,
            input_variables=["context", "question"]
        )
    
    def setup_chain(self):
        retriever = self.vector_store_manager.vector_store.as_retriever(
            search_kwargs={"k": 3}
        )
        self.chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=retriever,
            memory=self.memory,
            combine_docs_chain_kwargs={"prompt": self.prompt},
            return_source_documents=True
        )
    
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
            response = self.chain({"question": question})
            answer = response.get("answer", "").strip()
            
            if not answer or "没有找到" in answer or "未找到" in answer:
                return "文档中未找到相关答案"
            
            return answer
        except Exception as e:
            return f"问答过程中出现错误: {str(e)}"
    
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