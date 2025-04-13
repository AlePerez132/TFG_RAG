import os
from dotenv import load_dotenv

load_dotenv()

from langchain_huggingface import HuggingFaceEmbeddings


dir="./pdf"

from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
loader = DirectoryLoader(
    dir,
    glob="**/*.pdf",
    loader_cls=PyPDFLoader,
)

docs = loader.load()

from langchain.text_splitter import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=20)
split_docs = splitter.split_documents(docs)

# Extraer solo el texto de los documentos
texts = [doc.page_content for doc in split_docs]

# Create embeddings using Hugginface
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

from langchain_community.vectorstores import FAISS
db = FAISS.from_texts(texts, embeddings)
db.save_local("faiss_index")