# vector_store.py
import faiss
import numpy as np
import pandas as pd
import os

EMBEDDING_DIM = 1536 

class VectorStore:
    def __init__(self):
        self.index = faiss.IndexFlatL2(EMBEDDING_DIM)
        self.metadata = []

    def load_embeddings(self, relative_path: str):
        # Finds the path relative to this file
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        full_path = os.path.join(base_dir, relative_path)

        if not os.path.exists(full_path):
            raise FileNotFoundError(f"Embedding file not found at: {full_path}")

        df = pd.read_pickle(full_path)
        embeddings = np.array(df["embedding"].tolist()).astype("float32")
        self.index.add(embeddings)
        
        # Store metadata as a list of dicts for easy retrieval
        self.metadata = df.to_dict(orient="records")
        print(f"✅ Loaded {self.index.ntotal} vectors")

    def search(self, query_embedding, top_k=5):
        query_embedding = np.array([query_embedding]).astype("float32")
        distances, indices = self.index.search(query_embedding, top_k)

        results = []
        for idx in indices[0]:
            if idx != -1: # Ensure valid index
                results.append(self.metadata[idx])
        return results