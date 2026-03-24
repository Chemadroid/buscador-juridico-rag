import streamlit as st
import chromadb
from sentence_transformers import SentenceTransformer
import os
import zipfile
import requests

# 🔗 URL de tu base en Drive
URL_ZIP = "https://drive.google.com/uc?id=17MmP7pwNWa0VX1D8dKsbRwa-rUPXGrnx"

# 📦 Descargar y descomprimir base si no existe
def descargar_db():
    if not os.path.exists("chroma_db"):
        st.write("Descargando base de datos...")

        r = requests.get(URL_ZIP)
        with open("db.zip", "wb") as f:
            f.write(r.content)

        with zipfile.ZipFile("db.zip", 'r') as zip_ref:
            zip_ref.extractall()

        st.write("Base lista")

descargar_db()

# 🤖 Cargar modelo
@st.cache_resource
def cargar_modelo():
    return SentenceTransformer("all-MiniLM-L6-v2")

modelo = cargar_modelo()

# 🗂️ Conexión a Chroma (VERSIÓN CORRECTA)
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_collection(name="tesis_rag_final")

# 🎨 UI
st.title("Buscador Jurídico Inteligente")
st.write("Consulta tesis de la SCJN usando búsqueda semántica")

query = st.text_input("Escribe tu consulta jurídica:")

if query:
    with st.spinner("Buscando..."):
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
