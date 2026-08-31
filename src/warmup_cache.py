"""
Préchauffage du cache sémantique — à lancer la veille de la démo.

Certaines questions n'ont pas de fiche pré-rédigée (elles n'existent que
dans la FAQ PDF) : la 1re fois, le LLM doit rédiger (20-60 s sur CPU).
Ce script pose ces questions UNE fois hors ligne et met les réponses en
cache → le jour J, elles répondent en ~0,3 s.

Ajoutez ici toutes les questions que vous comptez montrer en démo.

Lancement :  python -X utf8 src\\warmup_cache.py
"""
import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from agent_rag import AgenticRAG
from semantic_cache import SemanticCache, empreinte_corpus

# ── Questions de la démo (complétez librement) ───────────────────────
QUESTIONS_DEMO = [
    "Je n'arrive pas à m'inscrire, on me dit que mon salaire ne correspond pas",
    "Comment payer ma taxe d'habitation en ligne ?",
    "Comment obtenir une attestation d'imposition ?",
    "Où télécharger ma quittance de paiement ?",
    "Comment consulter ma situation fiscale ?",
]


def main():
    agent = AgenticRAG()
    if not agent.llm.is_available():
        print("Ollama indisponible — lancez Ollama d'abord.")
        sys.exit(1)
    cache = SemanticCache(agent.retriever.embeddings,
                          version=empreinte_corpus(agent.retriever.vectorstore))

    for i, q in enumerate(QUESTIONS_DEMO, 1):
        print(f"\n[{i}/{len(QUESTIONS_DEMO)}] {q}")
        if cache.get(q):
            print("  → déjà en cache ⚡ (rien à faire)")
            continue
        start = time.time()
        r = agent.answer(q)
        deja_rapide = any(e.get("noeud") == "direct" for e in r.get("etapes", []))
        if r["statut"] == "SUCCESS" and not deja_rapide:
            cache.put(q, r["reponse"], r["sources"])
            print(f"  → rédigée en {time.time() - start:.0f}s et mise en cache "
                  f"(prochaine fois : ~0,3 s)")
        elif deja_rapide:
            print(f"  → déjà instantanée via fiche pré-validée ⚡ ({r['latence_s']}s)")
        else:
            print(f"  → statut {r['statut']} (non mise en cache)")

    print("\nPréchauffage terminé — toutes les questions de démo répondront en < 2 s. ⚡")


if __name__ == "__main__":
    main()
