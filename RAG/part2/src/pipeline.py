from .llm_client import LLMClient
from .rag_engine import RAGEngine
from .prompts import PROMPT_DECONSTRUCT, PROMPT_JUDGE

class NuanceDetector:
    def __init__(self):
        self.llm = LLMClient()
        self.rag = RAGEngine()

    def analyze_article(self, row_data):
        """
        Runs the 3-stage Nuance Detection on a single article row.
        """
        # Updated keys to match: id,title,url,section,published_date,bodyText
        title = row_data.get('title', 'No Title')
        content = row_data.get('bodyText', '')
        date = row_data.get('published_date', 'Unknown Date')
        article_id = row_data.get('id', 'Unknown ID')
        
        print(f"Processing ID {article_id}: {title}...")

        # --- STAGE 1: DECONSTRUCTION ---
        deconstruct_prompt = PROMPT_DECONSTRUCT.format(article_text=content)
        structure = self.llm.generate_json(deconstruct_prompt)
        
        if not structure:
            return {"error": "Failed to deconstruct article", "id": article_id}

        premises = structure.get("premises", [])
        conclusion = structure.get("conclusion", "")

        # --- STAGE 2: CONTEXT RETRIEVAL (RAG) ---
        context = self.rag.retrieve_context(premises, article_text=content)

        # --- STAGE 3: LOGIC JUDGE ---
        metadata_str = f"Date: {date}, Title: {title}, ID: {article_id}"
        judge_prompt = PROMPT_JUDGE.format(
            metadata=metadata_str,
            premises=str(premises),
            conclusion=conclusion,
            context=context
        )
        
        verdict = self.llm.generate_json(judge_prompt)
        
        if not verdict:
            return {"error": "Failed to generate verdict", "id": article_id}

        # Combine everything into a final report
        return {
            "id": article_id,
            "title": title,
            "published_date": date,
            "extracted_premises": premises,
            "extracted_conclusion": conclusion,
            "retrieved_context": context,
            "analysis": verdict
        }