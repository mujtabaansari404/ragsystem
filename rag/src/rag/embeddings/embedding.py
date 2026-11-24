import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from rag.chunking.text_splitter import split_text_documents
from rag.loaders.loader import loadFile
import time

def create_vector_embeddings(chunks, persist_directory = "./chroma_db"):
    if not chunks:
        print("chunks not found")
        return None

    print("\n 🧠: Embedding Model loading... ")
    embedding_model = HuggingFaceEmbeddings(model_name = "sentence-transformers/all-MiniLM-L6-v2")

    print("\n ⚙️: Generating Vectors....")

    print("\n 💾: Saving Vectors to 'persist directory' folder. ")
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=persist_directory
    )

    print("🎉 Success! vector database created.")
    return vector_store


if __name__ == "__main__":
    pdf_path = "../data/doc/attentionisallyouneed.pdf"
    db_folder = "./chroma_db"
    
    
    if os.path.exists(pdf_path):
        print("--- Step 1: Loading PDF ---")
        loaded_docs = loadFile(pdf_path)
        
        print("\n--- Step 2: Chunking Text ---")
        final_chunks = split_text_documents(loaded_docs)
        
        print("\n--- Step 3: Creating Embeddings ---")
        db = create_vector_embeddings(final_chunks, persist_directory=db_folder)
    else:
        print(f"❌ Error: pdf not found: {pdf_path}")