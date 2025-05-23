from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from def_clase import RAG

loader = DirectoryLoader(
    path="./pdf_",
    glob="**/*.pdf",
    loader_cls=PyPDFLoader,
)
docs = loader.load()

rag=RAG()

print("\nEN BRUTO:\n")
print(docs[0].page_content)

texto_limpio=rag.limpiar_texto_pubmed(docs[0].page_content.__str__)

print("\nPROCESADO:\n")
print(texto_limpio)