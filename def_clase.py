from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_community.vectorstores import FAISS
from langchain.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
import torch
import chainlit as cl
from langchain.schema import StrOutputParser
import re
from unidecode import unidecode


modelo = ChatOllama(
    base_url="http://localhost:11434",
    model="mistral",
    temperature=0.0,
    max_tokens=500
)

prompt=( "Eres un asistente experto en temas médicos que responde siempre en español. "
            "Debes usar únicamente la información proporcionada en el contexto para responder. "
            "NO menciones artículos, fuentes, documentos ni autores. "
            "NO digas frases como 'el objetivo de este artículo' o 'según el documento'. "
            "No digas frases como 'dentro del contexto proporcionado.' "
            "Responde de forma directa, como si el conocimiento fuera tuyo. "
            "Si no tienes información suficiente en el contexto, responde: 'No tengo información sobre su pregunta'.\n\n"
         )

system_prompt_with_context = prompt + "Contexto:\n{relevant_doc}" #lo guardo así porque si no no puede acceder a la clave relevant_doc

prompt_template_ragas = ChatPromptTemplate.from_messages([
            ("system", system_prompt_with_context),
            ("human", "\nUsuario: {query}")
        ])

prompt_template_chainlit = ChatPromptTemplate.from_messages([
            ("system", system_prompt_with_context), 
            ("human", "{history}\nUsuario: {query}")
        ])

class RAG:
    def __init__(self):
        self.llm = modelo
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"device": "cuda" if torch.cuda.is_available() else "cpu"}
        )
        self.doc_embeddings = None
        self.docs = None
        self.retriever = None
        self.prompt_template_ragas=prompt_template_ragas
        self.prompt_template_chainlit=prompt_template_chainlit

    def limpiar_texto_pubmed(texto):
    # Normalización básica
        texto = unidecode(texto)

    # Remueve emails, DOI, orcid, URLs
        texto = re.sub(r'\S+@\S+', '', texto)
        texto = re.sub(r'https?://\S+', '', texto)
        texto = re.sub(r'doi:?\s*\S+', '', texto, flags=re.I)

    # Remueve encabezados académicos comunes
        patrones = [
            r'(?i)(received|accepted|published|correspondence|editor).*?\n',
            r'(?i)(abstract|introduction|resumen)\s*:',  # mantén estos si usas títulos
            r'(?i)(keywords|palabras clave)\s*:',  # opcional: mantener o eliminar
            r'(?i)(conflict of interest|funding|references|bibliography|agradecimientos|acknowledg).*',
            r'(?i)(author|autores|editor|revisado por):?.*\n',
            r'\n{2,}',  # múltiples saltos de línea
        ]

        for patron in patrones:
            texto = re.sub(patron, '\n', texto)

    # Elimina líneas muy cortas (nombres, títulos)
        texto = '\n'.join([l for l in texto.split('\n') if len(l.strip()) > 30])

        return texto.strip()

    def generate_embeddings(self):
        loader = DirectoryLoader(
            path="./pdf",
            glob="**/*.pdf",
            loader_cls=PyPDFLoader,
        )
        docs = loader.load()

        splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=20)
        split_docs = splitter.split_documents(docs)

        texts = [doc.page_content for doc in split_docs]
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

        db = FAISS.from_texts(texts, embeddings)
        db.save_local("faiss_index")

    def load_retriever(self):
        db = FAISS.load_local("./faiss_index", self.embeddings, allow_dangerous_deserialization=True)
        self.retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 4})

    def get_most_relevant_docs_string(self, query):
        docs = self.retriever.invoke(query)
        context = "\n\n".join([doc.page_content for doc in docs])
        return context

    def get_most_relevant_docs_list(self, query):
        docs = self.retriever.invoke(query)
        context_list = [doc.page_content for doc in docs]
        return context_list

    def generate_answer_ragas(self, query, relevant_doc):
        prompt_template = self.prompt_template_ragas

        messages = prompt_template.format_messages(query=query, relevant_doc=relevant_doc)
        ai_msg = self.llm.invoke(messages)
        return ai_msg.content

    def chat_start_chainlit(self):
        cl.user_session.set("history", [])
        runnable = self.prompt_template_chainlit | self.llm | StrOutputParser()
        self.runnable=runnable
        cl.user_session.set("runnable", runnable)
        return runnable