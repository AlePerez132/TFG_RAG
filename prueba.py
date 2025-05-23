from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from def_clase import RAG

loader = DirectoryLoader(
    path="./pdf_",
    glob="**/*.pdf",
    loader_cls=PyPDFLoader,
)
docs = loader.load()

contenido = ""
for doc in docs:
    contenido += doc.page_content

print("EN BRUTO")
print(contenido)

rag=RAG()

print(rag.optimizar_texto_estudio(contenido))
