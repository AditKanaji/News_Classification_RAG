from .groq_llm_client import LLMClientGroq
from .groq_rag_engine import RAGEngineGroq

class NuanceDetectorGroq:
    def __init__(self):
        self.llm = LLMClientGroq()
        self.rag = RAGEngineGroq()

    def judge_article(self, suspicious_article, evidence_chunks):
        """
        Constructs the prompt and gets the verdict from Groq.
        """
        evidence_text = "\n\n".join([f"REAL NEWS ({c['source']}): {c['text']}" for c in evidence_chunks])
        
        prompt = f"""
        SUSPICIOUS ARTICLE:
        Title: {suspicious_article.get('title')}
        Date: {suspicious_article.get('published_date')}
        Content Snippet: {suspicious_article.get('bodyText')[:1000]}...

        VERIFIED REAL NEWS (Ground Truth):
        {evidence_text}

        TASK:
        Compare the "SUSPICIOUS ARTICLE" against the "VERIFIED REAL NEWS".
        1. Does it contradict the real news?
        2. Does it invent events?
        3. Does it twist facts?

        OUTPUT format:
        Verdict: [SAFE / MISLEADING / FABRICATED]
        Confidence: [0-100]%
        Explanation: [Brief analysis]
        """
        
        system_instruction = "You are a Fact-Checking Judge. Analyze discrepancies between the claim and the evidence."
        
        return self.llm.generate_text(prompt, system_instruction)