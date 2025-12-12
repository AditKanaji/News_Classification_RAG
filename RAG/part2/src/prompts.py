# Stage 1: Deconstruction
# Extracts the raw claims and the intended argument.
PROMPT_DECONSTRUCT = """
You are a Logic Parser. Read the following news article carefully.
Your goal is to separate the factual claims from the author's argument.

Input Article:
"{article_text}"

Instructions:
1. Extract "premises": A list of specific, verifiable claims (statistics, quotes, dates, physical events).
2. Extract "conclusion": The main argument, opinion, or narrative the author is deriving from those facts.
3. Keep the text exact where possible.

Output JSON format:
{{
  "premises": ["claim 1", "claim 2"],
  "conclusion": "The author's main argument"
}}
"""

# Stage 3: Logic Judge
# compares the article against the retrieved "Ground Truth".
PROMPT_JUDGE = """
You are a Nuance Misinformation Detector. 
Your job is to identify "Lying with Truths" — where real facts are used to support a false or misleading conclusion.

You have three inputs:
1. ARTICLE_METADATA: {metadata}
2. AUTHOR_PREMISES: {premises}
3. AUTHOR_CONCLUSION: "{conclusion}"
4. RETRIEVED_CONTEXT (Ground Truth): "{context}"

Task:
1. Compare the AUTHOR_CONCLUSION against the RETRIEVED_CONTEXT.
2. Check for these specific Logical Fallacies:
   - Cherry Picking: Citing one true negative fact while ignoring broader positive context.
   - False Causality: Linking unrelated events.
   - Omission: Leaving out critical data that changes the meaning.
3. Assign a Deception Score (0-10), where 0 is Truthful/Safe and 10 is Highly Manipulative.

Output JSON format:
{{
  "is_misleading": boolean,
  "fallacy_type": "None" or "Cherry Picking" or "False Causality" or "Omission" or "Strawman",
  "deception_score": integer,
  "reasoning": "Brief explanation of why the score was given, citing specific missing context."
}}
"""