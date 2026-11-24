from langchain_community.document_loaders import PyPDFLoader

def loadFile(path):
    loader = PyPDFLoader(path)
    documents = loader.load()
    return documents



