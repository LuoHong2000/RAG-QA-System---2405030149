import sys
sys.path.insert(0, '.')

print("Testing RAGQAChain module...")

try:
    from rag_chain import RAGQAChain
    print("✓ Import successful")
    
    rag = RAGQAChain()
    methods = [m for m in dir(rag) if not m.startswith('_')]
    print(f"Available methods: {methods}")
    
    if 'ask_stream' in methods:
        print("✓ ask_stream method FOUND")
    else:
        print("✗ ask_stream method NOT FOUND")
        
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()

print("\nTesting app.py imports...")
try:
    from rag_chain import RAGQAChain as ImportTest
    print("✓ App imports successful")
except Exception as e:
    print(f"✗ App import error: {e}")