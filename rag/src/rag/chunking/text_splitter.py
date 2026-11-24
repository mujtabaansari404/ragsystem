from langchain_text_splitters import RecursiveCharacterTextSplitter
from rag.loaders.loader import loadFile


def split_text_documents(documents):
    textSplitter = RecursiveCharacterTextSplitter(
       chunk_size = 1000,
       chunk_overlap = 200
    )

    chunks = textSplitter.split_documents(documents)

    print(f"loaded: #{len(documents)} ")
    print(f"created {len(chunks)} chunks!")

    for i, chunk in enumerate(chunks, start=1):
        print(f"\n--- CHUNK {i} TEXT ---")
        print(chunk.page_content)

        print(f"--- CHUNK {i} METADATA ---")
        print(chunk.metadata)

    return chunks


if __name__ == "__main__":
    path = "../data/doc/attentionisallyouneed.pdf"
    loadedDocs = loadFile(path)
    finalChunks =  split_text_documents(loadedDocs)