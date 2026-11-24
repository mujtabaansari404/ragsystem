from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from rag.retrievel.retrivel import get_retriever
from rag.llm.model import get_llm, create_prompt 

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def get_sources(docs):
    sources = []

    for doc in docs:
        source = doc.metadata.get("source", "Unknown")
        page = doc.metadata.get("page", 0) + 1

        sources.append(f"{source} - Page {page}")

    return list(dict.fromkeys(sources))

def create_rag_chain():
    retriever = get_retriever()
    llm = get_llm()
    prompt = create_prompt()

    rag_chain = (
        {
            "context": retriever | RunnableLambda(format_docs),
            "question": RunnablePassthrough()
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain

if __name__ == "__main__":
    retriever = get_retriever()
    chain = create_rag_chain()

    question = "What is multi-head attention?"

    docs = retriever.invoke(question)
    answer = chain.invoke(question)

    print(f"bot: {answer}")

    print("\nSources:")
    for source in get_sources(docs):
        print(f"📄 {source}")
   