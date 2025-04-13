import os
from dotenv import load_dotenv

load_dotenv()

from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings()

db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 4})

query="¿Qué objetivos tienen los enfoques farmacológicos?"
docs = retriever.invoke(query)

for i, doc in enumerate(docs):
    print(f"Documento {i + 1}:")
    print(doc.page_content)
    print("\n" + "="*80 + "\n")  # Separador para claridad

# Combina los documentos recuperados en un contexto único
context = "\n\n".join([doc.page_content for doc in docs])

# Construye el prompt para el modelo
prompt = f"""
Usa la siguiente información relevante para responder la pregunta:
{context}

Pregunta: {query}
Respuesta:
"""

from openai import OpenAI
import os

client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),
)

response = client.chat.completions.create(
    model="gpt-3.5-turbo",  # Modelo a utilizar
    messages=[
        {"role": "user", "content": prompt}  # Mensaje del usuario
    ],
    temperature=0.3,          # Control de creatividad
    max_tokens=256,           # Máximo de tokens en la respuesta
    top_p=1,                  # Nucleus sampling
    frequency_penalty=0,      # Penalización por repetición de palabras
    presence_penalty=0        # Penalización por introducir nuevos temas
)

print(response.choices[0].message.content)