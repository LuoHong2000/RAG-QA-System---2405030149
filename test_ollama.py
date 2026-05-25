from langchain.llms import Ollama

def test_ollama():
    try:
        llm = Ollama(model="deepseek-r1:7b")
        response = llm("Hello, how are you?")
        print("Ollama API测试成功!")
        print("响应:", response)
        return True
    except Exception as e:
        print(f"Ollama API测试失败: {e}")
        return False

if __name__ == "__main__":
    test_ollama()