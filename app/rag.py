# rag.py (Refactored)
import os
from vector_store import VectorStore
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)

# Initialize FAISS store once
store = VectorStore()
# Using the relative path as defined in your vector_store.py logic
store.load_embeddings("data/air_pollution_embeddings.pkl")

def generate_answer(question: str) -> str:
    # 1. Generate embedding for the query
    embedding_response = client.embeddings.create(
        model="text-embedding-3-small",
        input=question
    )
    query_embedding = embedding_response.data[0].embedding

    # 2. Search in FAISS vector store
    # This now returns the actual data rows from metadata
    search_results = store.search(query_embedding, top_k=3)

    # 3. Build context directly from search results
    context_list = []
    for row in search_results:
        context_list.append(
            f"City: {row.get('City')}, Country: {row.get('Country')}, PM2.5 AQI: {row.get('PM2.5 AQI Value')}"
        )

    context = "\n".join(context_list)

    # 4. Generate answer using OpenAI Chat
    prompt = f"Answer the question using the following context:\n{context}\nQuestion: {question}\nAnswer:"
    
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )

    # FIX: Access as attribute (.content), not as dict (["content"])
    return response.choices[0].message.content