import google.generativeai as genai
import numpy as np
from .config import GOOGLE_API_KEY, EMBEDDING_MODEL_NAME, CHUNK_SIZE

class LightweightRAGEngine:
    def __init__(self):
        genai.configure(api_key=GOOGLE_API_KEY)
        self.vector_store = [] # List of {text, embedding, source}
    
    def _chunk_text(self, text, chunk_size=CHUNK_SIZE):
        """Simple character-based chunking."""
        return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

    def ingest_articles(self, articles):
        """
        Process articles: Chunk them and generate embeddings.
        """
        print("   ⚙️ Processing and embedding articles...")
        self.vector_store = [] # Reset store
        
        for article in articles:
            chunks = self._chunk_text(article['content'])
            
            # Batch embedding could be optimized, but doing loop for simplicity/rate limits
            for chunk in chunks:
                try:
                    # Generate embedding
                    result = genai.embed_content(
                        model=EMBEDDING_MODEL_NAME,
                        content=chunk,
                        task_type="retrieval_document"
                    )
                    
                    self.vector_store.append({
                        "text": chunk,
                        "embedding": np.array(result['embedding']),
                        "source": article['title'],
                        "url": article['url']
                    })
                except Exception as e:
                    print(f"      ⚠️ Failed to embed chunk: {e}")
        
        print(f"   🧠 Built in-memory index with {len(self.vector_store)} chunks.")

    def retrieve(self, query, top_k=3):
        """
        Cosine similarity search for the query against the vector store.
        """
        if not self.vector_store:
            return []

        # 1. Embed the query
        query_embedding = genai.embed_content(
            model=EMBEDDING_MODEL_NAME,
            content=query,
            task_type="retrieval_query"
        )['embedding']
        query_vec = np.array(query_embedding)

        # 2. Calculate Cosine Similarity
        # Sim(A, B) = (A . B) / (||A|| * ||B||)
        scored_chunks = []
        for item in self.vector_store:
            doc_vec = item['embedding']
            similarity = np.dot(query_vec, doc_vec) / (np.linalg.norm(query_vec) * np.linalg.norm(doc_vec))
            scored_chunks.append((similarity, item))

        # 3. Sort and Return Top K
        scored_chunks.sort(key=lambda x: x[0], reverse=True)
        return [item for score, item in scored_chunks[:top_k]]