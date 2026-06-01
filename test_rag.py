import sys
try:
    from rag_chain import RAGQAChain
    print("Import successful")
    
    # Check if ask_stream method exists
    rag = RAGQAChain()
    methods = [m for m in dir(rag) if not m.startswith('_')]
    print(f"Methods available: {methods}")
    
    if 'ask_stream' in methods:
        print("✓ ask_stream method found")
    else:
        print("✗ ask_stream method NOT found")
        
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()