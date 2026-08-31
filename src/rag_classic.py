"""
Phase 3 — RAG Classique (baseline).
Pipeline fixe : question → retrieve top-k → prompt strict → LLM → réponse + sources.
Sert de point de comparaison chiffré pour mesurer l'apport de l'Agentic RAG.
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import FALLBACK_ANSWER, TOP_K
from llm import OllamaLLM
from retriever import TGRRetriever

SYSTEM_PROMPT = """Tu es l'assistant officiel du portail eServices de la Trésorerie Générale du Royaume du Maroc (TGR).

RÈGLES STRICTES :
1. Réponds UNIQUEMENT à partir des passages de documentation fournis ci-dessous.
2. Si l'information ne se trouve pas dans les passages, dis-le clairement — n'invente JAMAIS.
3. Réponds dans la LANGUE de la question (français ou arabe).
4. N'invente jamais un lien, un délai, un montant ou une procédure.
5. Sois concis, clair et poli. Structure ta réponse en étapes si c'est une procédure.
6. Ne mentionne JAMAIS les mots « passage », « Passage 1 », « documentation fournie » — l'usager ne les voit pas. Réponds directement, comme un agent au guichet.
7. Commence directement par la solution (pas de « je ne trouve pas... cependant »)."""


def build_context(passages: list[dict]) -> str:
    blocks = []
    for i, p in enumerate(passages, 1):
        src = p["fiche_id"] or p["fichier"]
        blocks.append(f"--- Passage {i} (source: {src}, catégorie: {p['categorie']}) ---\n{p['text']}")
    return "\n\n".join(blocks)


class ClassicRAG:
    def __init__(self, retriever: TGRRetriever = None, llm: OllamaLLM = None):
        self.retriever = retriever or TGRRetriever()
        self.llm = llm or OllamaLLM()

    def answer(self, question: str) -> dict:
        start = time.time()
        passages = self.retriever.search(question, k=TOP_K)

        if not passages:
            return {
                "question": question,
                "reponse": FALLBACK_ANSWER,
                "sources": [],
                "pipeline": "classic",
                "latence_s": round(time.time() - start, 1),
            }

        user_prompt = (
            f"PASSAGES DE DOCUMENTATION :\n\n{build_context(passages)}\n\n"
            f"QUESTION DE L'USAGER : {question}\n\nRéponds selon les règles."
        )
        reponse = self.llm.generate(SYSTEM_PROMPT, user_prompt)

        return {
            "question": question,
            "reponse": reponse,
            "sources": [
                {"id": p["fiche_id"] or p["fichier"], "categorie": p["categorie"],
                 "distance": p["distance"]}
                for p in passages
            ],
            "pipeline": "classic",
            "latence_s": round(time.time() - start, 1),
        }


if __name__ == "__main__":
    question = sys.argv[1] if len(sys.argv) > 1 else "J'ai perdu mon téléphone, comment me connecter ?"
    rag = ClassicRAG()
    if not rag.llm.is_available():
        print("⚠ Ollama indisponible — affichage du retrieval seul :\n")
        for p in rag.retriever.search(question):
            print(f"  [{p['fiche_id']}] dist={p['distance']} : {p['text'][:120]}")
        sys.exit(0)
    result = rag.answer(question)
    print(f"\nQ : {result['question']}")
    print(f"\nR : {result['reponse']}")
    print(f"\nSources : {result['sources']}")
    print(f"Latence : {result['latence_s']}s")
