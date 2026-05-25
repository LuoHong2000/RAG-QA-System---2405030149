from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from typing import List, Optional

class VectorStoreManager:
    def __init__(self, persist_directory: str = "./chroma_db"):
        self.persist_directory = persist_directory
        self.embeddings = OllamaEmbeddings(model="nomic-embed-text")
        self.vector_store = None
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
    
    def create_vector_store(self, documents: List[str]):
        all_chunks = []
        for doc in documents:
            chunks = self.text_splitter.split_text(doc)
            all_chunks.extend(chunks)
        
        if all_chunks:
            self.vector_store = Chroma.from_texts(
                texts=all_chunks,
                embedding=self.embeddings,
                persist_directory=self.persist_directory
            )
            self.vector_store.persist()
            return len(all_chunks)
        return 0
    
    def load_vector_store(self):
        if self.vector_store is None:
            try:
                self.vector_store = Chroma(
                    persist_directory=self.persist_directory,
                    embedding_function=self.embeddings
                )
            except Exception:
                pass
        return self.vector_store is not None
    
    def similarity_search(self, query: str, k: int = 3) -> List[str]:
        if self.vector_store is None:
            return []
        results = self.vector_store.similarity_search(query, k=k)
        return [doc.page_content for doc in results]
    
    def get_chunk_count(self) -> int:
        if self.vector_store is None:
            return 0
        return self.vector_store._collection.count()