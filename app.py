"""
from langchain_ollama import ChatOllama
from langchain.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
from langchain.schema.runnable import Runnable
from langchain.schema.runnable.config import RunnableConfig
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

import chainlit as cl
from typing import cast

from transformers import AutoTokenizer, AutoModel
import torch


# Configurar embeddings y vector store
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cuda" if torch.cuda.is_available() else "cpu"}
)
db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 4})

# Prompt adaptado a RAG + historial
prompt_template = ChatPromptTemplate.from_messages([
    ("system", 
        "Eres un asistente de IA que responde correctamente y en español. "
        "Si no sabes la respuesta, di 'No tengo información sobre su pregunta'. "
        "Usa la siguiente información relevante para responder la pregunta:\n\n{context}"),
    ("human", "{history}\nUsuario: {query}")
])

@cl.on_chat_start
async def on_chat_start():
    # Inicializamos el historial vacío en la sesión de usuario
    cl.user_session.set("history", [])

    model = ChatOllama(
        base_url="http://localhost:11434",
        model="mistral",
        temperature=0.0,
        max_tokens=500
    )
    runnable = prompt_template | model | StrOutputParser()
    cl.user_session.set("runnable", runnable)

@cl.on_message
async def on_message(message: cl.Message):
    runnable = cast(Runnable, cl.user_session.get("runnable"))

    # Recuperar y actualizar el historial
    history = cl.user_session.get("history", [])

    # Formatear el historial (sin incluir la última entrada del usuario, que se pasará como query)
    history_str = "\n".join(
        [f"{'Usuario' if m['role']=='user' else 'Asistente'}: {m['content']}" for m in history]
    )
    #Este código formatea el historial de la siguiente forma:
    #   Usuario: ¿Qué es la fotosíntesis?
    #   Asistente: La fotosíntesis es un proceso mediante el cual...
    #Recorriendo la lista de mensajes y determinando si pertenecen al usuario o al asistente.

    # Recuperar documentos relevantes
    query = message.content
    docs = retriever.invoke(query)
    context = "\n\n".join([doc.page_content for doc in docs])

    # Generar respuesta con streaming y capturar el texto completo
    msg = cl.Message(content="")
    response_text = ""

    async for chunk in runnable.astream(
        {"query": query, "context": context, "history": history_str},
        config=RunnableConfig(callbacks=[cl.LangchainCallbackHandler()]),
    ):
        response_text += chunk
        await msg.stream_token(chunk)
    await msg.send()

    # Guardamos la entrada del usuario en el historial
    history.append({"role": "user", "content": message.content})
    # Guardar la respuesta del asistente en el historial
    history.append({"role": "assistant", "content": response_text})
    cl.user_session.set("history", history)
    
    """
from def_clase import RAG
import chainlit as cl
from langchain.schema.runnable.config import RunnableConfig

rag=RAG()

rag.load_retriever()

@cl.on_chat_start
async def on_chat_start():
    rag.chat_start_chainlit()

@cl.on_message
async def on_message(message: cl.Message):
    runnable = rag.runnable

    # Recuperar y actualizar el historial
    history = cl.user_session.get("history", [])

    # Formatear el historial (sin incluir la última entrada del usuario, que se pasará como query)
    history_str = "\n".join(
        [f"{'Usuario' if m['role']=='user' else 'Asistente'}: {m['content']}" for m in history]
    )

    #Este código formatea el historial de la siguiente forma:
    #   Usuario: ¿Qué es la fotosíntesis?
    #   Asistente: La fotosíntesis es un proceso mediante el cual...
    #Recorriendo la lista de mensajes y determinando si pertenecen al usuario o al asistente.

    # Recuperar documentos relevantes
    query = message.content
    docs = rag.retriever.invoke(query)
    relevant_doc = "\n\n".join([doc.page_content for doc in docs])

    # Generar respuesta con streaming y capturar el texto completo
    msg = cl.Message(content="")
    response_text = ""

    async for chunk in runnable.astream(
        {"query": query, "relevant_doc": relevant_doc, "history": history_str},
        config=RunnableConfig(callbacks=[cl.LangchainCallbackHandler()]),
    ):
        response_text += chunk
        await msg.stream_token(chunk)
    await msg.send()

    # Guardamos la entrada del usuario en el historial
    history.append({"role": "user", "content": message.content})
    # Guardar la respuesta del asistente en el historial
    history.append({"role": "assistant", "content": response_text})
    cl.user_session.set("history", history)
