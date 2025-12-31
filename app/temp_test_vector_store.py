from vector_store import VectorStore
from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def get_embedding(text):
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return response.data[0].embedding


if __name__ == "__main__":
    store = VectorStore()

    
    store.load_embeddings("data/air_pollution_embeddings.pkl")

    query = "Which cities have high PM2.5 pollution?"
    query_embedding = get_embedding(query)

    results = store.search(query_embedding, top_k=3)

    for r in results:
        print(f"{r['City']} ({r['Country']}) - PM2.5 AQI: {r['PM2.5 AQI Value']}")
