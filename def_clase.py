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

    def optimizar_texto_estudio(self, texto):
    

        # Patrones de expresiones regulares para identificar secciones a eliminar o para delimitar el contenido
        patrones_a_eliminar = [
            r'The Journal of Prevention of Alzheimer’s Disease.*journal homepage: www\.elsevier\.com/locate/tjpad', # Encabezado de revista
            r'Stephen Macfarlane a ,.*Marwan N Sabbagh ab , \*', # Lista de autores y afiliaciones
            r'a The Dementia Centre, HammondCare, Melbourne, Victoria, Australia.*ab Barrow Neurological Institute, St\. Joseph’s Hospital and Medical Center, Phoenix, Arizona, USA', # Afiliaciones detalladas
            r'a r t i c l e i n f o\s*Keywords:.*a b s t r a c t', # Sección de keywords
            r'https://doi\.org/10\.1016/j\.tjpad\.2024\.100016', # DOI
            r'Available online \d{1,2} \w+ \d{4}', # Fecha de disponibilidad online
            r'\d{4}-\d{4}/©\d{4} Anavex Life Sciences Corp\. Published by Elsevier Masson SAS on behalf of SERDI Publisher\. This is an open access article under the CC BY license \( http://creativecommons\.org/licenses/by/4\.0/ \).*S\.Macfarlane,T\.Grimmer,K\.Teoetal\. TheJournalofPreventionofAlzheimer’sDisease12\(2025\)100016', # Copyright y licencias
            r'Fig\. \d+\..*Flowchart of patient screening, enrollment, discontinuation, and completion\.', # Descripción de la figura 1
            r'\[\s*\d{1,2}\s*\]', # Referencias numéricas tipo [1], [ 2], etc.
            r'\d{1,2}S\.Macfarlane,T\.Grimmer,K\.Teoetal\. TheJournalofPreventionofAlzheimer’sDisease\d{1,2}\(\d{4}\)\d{5,6}', # Encabezados de página con autores
            r'Table \d+\nDemographic characteristics of the Intent-to-Treat \(ITT\) population\..*Sex, n \(\%\).*White \d{1,3} \(?\d{1,3}\.?\d{1,3}\)?', # Tabla 1 y su contenido
            r'Results \nOf 988 participants screened, 508 were enrolled and randomized,', # Inicio de la sección de resultados
            r'Fig\. \d+\. Flowchart of patient screening, enrollment, discontinuation, and completion\.', # Pie de figura
            r'Table \d+\nDemographic characteristics of the Intent-to-Treat \(ITT\) population\.', # Encabezado de tabla 1 (si aparece solo)
        ]

    # Eliminar las secciones identificadas
        texto_procesado = texto
        for patron in patrones_a_eliminar:
            texto_procesado = re.sub(patron, '', texto_procesado, flags=re.DOTALL)

    # Identificar y extraer el Abstract y las secciones principales (Background, Objectives, Design, Setting, etc.)
    # Vamos a buscar un bloque que comience con "Background:" y termine antes de la "Introduction" detallada
    # o antes de la sección "Methods" si "Introduction" es muy corta y relevante al estudio.
    
    # Intento 1: Extraer desde "Background:" hasta "Conclusions:"
        match_abstract_and_conclusions = re.search(r'Background:.*Conclusions:.*?(?=\nIntroduction|\n\nIntroduction)', texto_procesado, re.DOTALL)
    
    # Si no encuentra el patrón anterior, intenta una extracción más amplia del Abstract
        if not match_abstract_and_conclusions:
            match_abstract_and_conclusions = re.search(r'a b s t r a c t\s*Background:.*Conclusions:.*', texto_procesado, re.DOTALL)
    
        contenido_principal = ""
        match_bg_to_methods = None
        if match_abstract_and_conclusions:
            contenido_principal = match_abstract_and_conclusions.group(0)
        else:
        # Si no se encuentra el bloque completo, podemos intentar extraer desde "Background:" hasta "Methods"
            match_bg_to_methods = re.search(r'Background:.*?(?=\nMethods)', texto_procesado, re.DOTALL)
        if match_bg_to_methods:
            contenido_principal = match_bg_to_methods.group(0)

    # Limpiar espacios en blanco excesivos y saltos de línea
        contenido_principal = re.sub(r'\n\s*\n', '\n\n', contenido_principal).strip()
    
    # También podemos limpiar el inicio del texto por si quedan fragmentos de los encabezados de página
        contenido_principal = re.sub(r'^\d{1,2}S\.Macfarlane,T\.Grimmer,K\.Teoetal\. TheJournalofPreventionofAlzheimer’sDisease\d{1,2}\(\d{4}\)\d{5,6}\n*', '', contenido_principal)


        return contenido_principal

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