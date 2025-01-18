from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# 1. load raw pdf
DATA_PATH = "data/"
def load_pdf_file(data):
    loader= DirectoryLoader(data,
                            glob="*.pdf",
                            loader_cls=PyPDFLoader)

    documents=loader.load()
    return documents
documents = load_pdf_file(data=DATA_PATH)
print("Length of pdf pages: ", len(documents))

# 2. create chunks


# 3. create vector embeddings
# 4. store embeddings in FAISS

