import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[3]
DB_DIR = BASE_DIR / "src" / "rag" / "chroma_db"

def get_retriever(db_dir = "./chroma_db"):
    if not os.path.exists(db_dir):
        raise FileNotFoundError(f"Error: '{db_dir}' folder not found! first run embedding.py.")

    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    vector_store = Chroma(
        persist_directory=str(DB_DIR), 
        embedding_function=embedding_model
    )

    retriever = vector_store.as_retriever(search_kwargs={"k": 4})
    return retriever


def query_pdf(question, db_directory="./chroma_db"):
   
    print(f"\n🔍 Searching database for: '{question}'")
    
    try:
        retriever = get_retriever(db_directory)
        
        matching_docs = retriever.invoke(question)
        
        print(f"📚 Total {len(matching_docs)} relevant chunks found:\n")
        
        for index, doc in enumerate(matching_docs, start=1):
            page_num = doc.metadata.get("page", 0) + 1  
            
            print(f"--- MATCHING CHUNK {index} (Page: {page_num}) ---")
            print(doc.page_content[:400]) 
            print("-" * 40 + "\n")
            
        return matching_docs

    except Exception as e:
        print(f"unexpected error: {e}")
        return []

if __name__ == "__main__":
    userquery = "what is the multihead machenism?"
    query_pdf(userquery)