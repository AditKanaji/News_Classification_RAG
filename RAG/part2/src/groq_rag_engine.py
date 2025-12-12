import numpy as np
import os
from sentence_transformers import SentenceTransformer
from .groq_config import EMBEDDING_MODEL_NAME, CHUNK_SIZE
os.environ["TOKENIZERS_PARALLELISM"] = "false"
class RAGEngineGroq:
    def __init__(self):
        print(f"   ⏳ Loading HF Embedding Model ({EMBEDDING_MODEL_NAME})...")
        # This downloads the model locally the first time you run it
        self.model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        self.vector_store = [] 

    def _chunk_text(self, text, chunk_size=CHUNK_SIZE):
        return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

    def ingest_articles(self, articles):
        """
        Chunks articles and generates embeddings using HuggingFace.
        """
        self.vector_store = [] # Reset store
        
        for article in articles:
            chunks = self._chunk_text(article['content'])
            
            # SentenceTransformers can encode a list of strings efficiently
            embeddings = self.model.encode(chunks)
            
            for i, chunk in enumerate(chunks):
                self.vector_store.append({
                    "text": chunk,
                    "embedding": embeddings[i],
                    "source": article['title'],
                    "url": article['url']
                })
        
        # print(f"   🧠 Index built with {len(self.vector_store)} chunks.")

    def retrieve(self, query, top_k=3):
        """
        Cosine similarity search using numpy.
        """
        if not self.vector_store:
            return []

        # Embed the query
        query_vec = self.model.encode(query)

        scored_chunks = []
        for item in self.vector_store:
            doc_vec = item['embedding']
            
            # Cosine Similarity: (A . B) / (||A|| * ||B||)
            similarity = np.dot(query_vec, doc_vec) / (np.linalg.norm(query_vec) * np.linalg.norm(doc_vec))
            scored_chunks.append((similarity, item))

        # Sort and return top K
        scored_chunks.sort(key=lambda x: x[0], reverse=True)
        
        # FIX: 'item' is already the dictionary, no need to access index [1]
        return [item for score, item in scored_chunks[:top_k]]