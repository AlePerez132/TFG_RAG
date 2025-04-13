from langchain_ollama import ChatOllama
from langchain.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
from langchain.schema.runnable import Runnable
from langchain.schema.runnable.config import RunnableConfig
from typing import cast

import os
from dotenv import load_dotenv

load_dotenv()

import chainlit as cl
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


#embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

#db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)

#retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 4})

prompt_template = ChatPromptTemplate.from_messages([
    ("system", "Usa la siguiente información relevante para responder la pregunta:\n\n"
     
    "{context}\n\n"

    "Pregunta: {query}\n"
    "Respuesta:")
])

@cl.on_chat_start
async def on_chat_start():
    model = ChatOllama(
        model="mistral",
        temperature=0.0,
        max_tokens=500
    )
    prompt = prompt_template
    runnable = prompt | model | StrOutputParser()
    cl.user_session.set("runnable", runnable)

    #elements = [
    #  cl.Pdf(name="pdf1", display="page", path="./pdf/Memoria_TFG_RAG.pdf", page=1)
    #]

    #await cl.Message(content="Look at this local pdf1!", elements=elements).send()


@cl.on_message
async def on_message(message: cl.Message):
    runnable = cast(Runnable, cl.user_session.get("runnable"))  # type: Runnable

    msg = cl.Message(content="")
    query=message.content
    #docs = retriever.invoke(query)
    #context = "\n\n".join([doc.page_content for doc in docs])

    async for chunk in runnable.astream(
        {
        "query": message.content,
        #"context": context
        "context":"Jose trabaja de camionero"
        },
        config=RunnableConfig(callbacks=[cl.LangchainCallbackHandler()]),
    ):
        await msg.stream_token(chunk)

    await msg.send()