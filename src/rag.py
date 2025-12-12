import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import google.generativeai as genai
from src.models import configure_genai
import logging

logger = logging.getLogger(__name__)

class SimpleRAG:
    def __init__(self):
        self.documents = []
        self.embeddings = []
        configure_genai()
        self.model = genai.GenerativeModel('gemini-flash-latest')
        self.embedding_model = 'models/text-embedding-004'

    def add_documents(self, articles):
        """Adds articles to the RAG system and computes embeddings."""
        self.documents = articles
        texts = [f"{a['webTitle']} {a['bodyText'][:500]}" for a in articles]
        
        logger.info("Generating embeddings for RAG...")
        try:
            # Batching might be needed for large datasets
            results = genai.embed_content(
                model=self.embedding_model,
                content=texts,
                task_type="retrieval_document"
            )
            self.embeddings = np.array(results['embedding'])
            logger.info("Embeddings generated.")
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")

    def retrieve(self, query, k=3):
        """Retrieves top-k relevant documents."""
        if len(self.embeddings) == 0:
            return []
            
        try:
            query_embedding = genai.embed_content(
                model=self.embedding_model,
                content=query,
                task_type="retrieval_query"
            )['embedding']
            
            query_embedding = np.array(query_embedding).reshape(1, -1)
            similarities = cosine_similarity(query_embedding, self.embeddings)[0]
            
            top_k_indices = similarities.argsort()[-k:][::-1]
            return [self.documents[i] for i in top_k_indices]
        except Exception as e:
            logger.error(f"Error during retrieval: {e}")
            return []

    def generate_answer(self, query):
        """Generates an answer using retrieved context."""
        retrieved_docs = self.retrieve(query)
        if not retrieved_docs:
            return "No relevant information found."
            
        context = "\n\n".join([f"Title: {d['webTitle']}\nContent: {d['bodyText'][:1000]}" for d in retrieved_docs])
        
        prompt = f"""
        You are a helpful assistant. Use the following context to answer the user's question.
        If the answer is not in the context, say you don't know.
        
        Context:
        {context}
        
        Question: {query}
        
        Answer:
        """
        
        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            return "Error generating answer."
