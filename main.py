from dotenv import load_dotenv

from langchain_community import embeddings
from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import OpenAIEmbeddings

import warnings

warnings.filterwarnings(
    "ignore", category=DeprecationWarning
)  # langchain-community sunset notice

from langchain_community.embeddings import (
    FastEmbedEmbeddings,
)  # local embedding model (the one community import)
from langchain_core.vectorstores import InMemoryVectorStore

load_dotenv()

# Fase de Ingestão de Documentos
# 1. Carregar os documentos
# 2. Splitar os documentos em chunks
# 3. Criar embeddings para cada chunk
# 4. Armazenar os embeddings em um vetor store
# 5. Carregar o vetor store para uso posterior


# Fase de Consulta
# 1. Receber a consulta do usuário
# 2. Criar embeddings para a consulta
# 3. Recuperar os chunks mais relevantes do vetor store


def _filter_by_page(doc) -> Document:
    return doc.metadata.get("page") == 2


def main():
    embeddings = FastEmbedEmbeddings(
        model_name="BAAI/bge-small-en-v1.5", cache_dir="models"
    )

    vector_store = InMemoryVectorStore.load(
        path="vector_store/react_store.json", embedding=embeddings
    )
    print(len(vector_store.store), "entries")

    entry = next(iter(vector_store.store.values()))
    print(entry.keys())
    print(entry["vector"][:5], entry["text"][:80])

    # Embedando a consulta do usuário
    query = "How does ReAct combine reasoning and acting?"
    query_vector = embeddings.embed_query(query)
    print(len(query_vector), query_vector[:5])

    # Buscando os 3 chunks mais relevantes
    for doc in vector_store.similarity_search(query, k=3):
        print(f"page {doc.metadata['page']}: {doc.page_content[:150]!r}\n")

    # Avaliando a similaridade entre a consulta e os chunks recuperados (scores)
    for q in [query, "best pizza in Naples"]:
        scores = [
            round(vector_store.similarity_search(q, k=1)[0].metadata.get("score", 0), 3)
        ]
        print(f"{q!r:48} {scores}")

    # Filtrando por metadata (ex: page number)
    for doc in vector_store.similarity_search(query, k=3, filter=_filter_by_page):
        print(f"page {doc.metadata['page']}: {doc.page_content[:150]!r}\n")

    retriever = vector_store.as_retriever(search_kwargs={"k": 3})
    docs = retriever.invoke(query)
    print(f"retriever.invoke: {len(docs)} docs")
    print(f"page {docs[0].metadata['page']}: {docs[0].page_content[:150]!r}\n")

    query = "Is there a drink recipe in the document?"
    doc = vector_store.similarity_search(query, k=1)[0]
    print(doc.metadata)
    print(doc.page_content)


if __name__ == "__main__":
    main()
