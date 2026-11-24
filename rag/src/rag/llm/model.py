import os

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate


load_dotenv()


def get_llm():
    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        raise ValueError("GROQ_API_KEY not found in .env")

    llm = ChatGroq(
        model="openai/gpt-oss-120b",
        temperature=0,
        api_key=api_key,
    )

    return llm


def create_prompt():
    prompt = ChatPromptTemplate.from_template(
        """
        You are a helpful assistant answering questions based only on
        the provided context.

        If the answer cannot be found in the context, say:
        "I don't know based on the provided document."

        Context:
        {context}

        Question:
        {question}

        Answer:
        """
    )

    return prompt