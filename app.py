import streamlit as st
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

# modelo
modelo = SentenceTransformer("all-MiniLM-L6-v2")

# conexión (local)
client = chromadb.Client(Settings(persist_directory="./chroma_db"))

collection = client.get_collection(name="tesis_rag_final")

st.title("Buscador Jurídico Inteligente")

query = st.text_input("Escribe tu consulta jurídica:")

if query:
    query_embedding = modelo.encode(query).tolist()

    resultados = collection.query(
        query_embeddings=[query_embedding],
        n_results=10
    )

    docs = resultados["documents"][0]

    keywords = query.lower().split()

    def score_keyword(texto):
        texto_lower = texto.lower()
        return sum(1 for k in keywords if k in texto_lower)

    docs_ordenados = sorted(
        docs,
        key=lambda x: score_keyword(x),
        reverse=True
    )

    for i, doc in enumerate(docs_ordenados[:5]):
        st.subheader(f"Resultado {i+1}")
        st.write(doc)
