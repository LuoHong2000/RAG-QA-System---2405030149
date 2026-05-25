import os
from document_processor import load_documents_from_folder
from rag_chain import RAGQAChain

def main():
    rag_chain = RAGQAChain()
    
    docs_folder = "./documents"
    if os.path.exists(docs_folder):
        documents = load_documents_from_folder(docs_folder)
        if documents:
            print(f"找到 {len(documents)} 个文档")
            chunk_count = rag_chain.add_documents(documents)
            print(f"知识库构建完成，共 {chunk_count} 个文本块")
        else:
            print("文件夹中没有找到有效的文档")
    else:
        print(f"文档文件夹 {docs_folder} 不存在")
        return
    
    test_questions = [
        "什么是自然语言处理？",
        "Transformer模型的主要特点是什么？",
        "BERT模型是如何预训练的？",
        "词嵌入的作用是什么？",
        "什么是注意力机制？",
        "量子计算的基本原理是什么？",
        "如何制作蛋糕？"
    ]
    
    print("\n开始测试问答效果...")
    for i, question in enumerate(test_questions, 1):
        print(f"\n问题 {i}: {question}")
        answer = rag_chain.ask(question)
        print(f"回答: {answer}")

if __name__ == "__main__":
    main()