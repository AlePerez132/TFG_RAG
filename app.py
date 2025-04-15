from langchain_ollama import ChatOllama
from langchain.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
from langchain.schema.runnable import Runnable
from langchain.schema.runnable.config import RunnableConfig
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

import chainlit as cl
from typing import cast
from dotenv import load_dotenv

load_dotenv()

# Carga de embeddings y FAISS
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 4})

# Prompt adaptado a RAG
prompt_template = ChatPromptTemplate.from_messages([
    ("system", "Eres un asistente de IA que responde correctamente y en español. "
               "Si no sabes la respuesta, di 'No tengo información sobre su pregunta'. "
               "Usa la siguiente información relevante para responder la pregunta:\n\n"
               "{context}"),
    ("human", "{query}")
])

@cl.on_chat_start
async def on_chat_start():
    model = ChatOllama(
        base_url="http://localhost:11434",  # si estás en contenedor docker
        model="mistral",
        temperature=0.0,
        max_tokens=500
    )
    runnable = prompt_template | model | StrOutputParser()
    cl.user_session.set("runnable", runnable)

@cl.on_message
async def on_message(message: cl.Message):
    runnable = cast(Runnable, cl.user_session.get("runnable"))

    # Obtener documentos similares
    query = message.content
    docs = retriever.invoke(query)
    context = "\n\n".join([doc.page_content for doc in docs])

    # Crear el mensaje de respuesta
    msg = cl.Message(content="")

    async for chunk in runnable.astream(
        {"query": query, "context": context},
        config=RunnableConfig(callbacks=[cl.LangchainCallbackHandler()]),
    ):
        await msg.stream_token(chunk)

    await msg.send()
