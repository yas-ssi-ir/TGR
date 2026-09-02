"""
Vérification complète du système — une seule commande avant toute démo.

Enchaîne les contrôles qui comptent et rend un verdict PASSE / ÉCHOUE :

  1. Intégrité des données   : fiches, réponses pré-rédigées, base vectorielle
  2. Couverture voie rapide  : le consensus retrouve-t-il la bonne fiche ?
  3. Latence + justesse      : 18 questions réelles chronométrées
  4. Relecture humaine       : combien de réponses engagent encore le modèle seul

Aucun appel LLM sur les contrôles 1, 2 et 4. Le contrôle 3 en fait quelques-uns
(questions inédites) — comptez 2 à 4 minutes au total.

Lancement :  python -X utf8 src\verifier_tout.py
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (
    CONSENSUS_K, FAQ_FICHES_JSON, PRECOMPUTED_JSON, QA_FICHES_JSON,
)

# Seuils d'acceptation — c'est ici qu'on décide ce qui est « bon »
SEUIL_COUVERTURE = 90     # % de questions connues retrouvant leur fiche
SEUIL_CONFUSION = 5       # % max de mauvaise fiche désignée
SEUIL_LATENCE_MED = 2.0   # s — médiane
SEUIL_JUSTESSE = 100      # % de verdicts périmètre corrects
# Ce seuil reste à 100 alors qu'un cas limite documenté échoue (« crédit
# immobilier », voir bench_latence.py). Un contrôle rouge que l'on comprend
# vaut mieux qu'un seuil abaissé qui masque la réalité.

resultats = []


def controle(nom: str, ok: bool, detail: str):
    resultats.append((nom, ok, detail))
    print(f"  [{'OK   ' if ok else 'ECHEC'}] {nom} — {detail}")


def charger_fiches() -> list[dict]:
    fiches = []
    for chemin in (QA_FICHES_JSON, FAQ_FICHES_JSON):
        if os.path.exists(chemin):
            with open(chemin, encoding="utf-8") as f:
                fiches += json.load(f)
    return fiches


def main():
    print("=" * 92)
    print("  VÉRIFICATION COMPLÈTE — Assistant Agentic RAG TGR")
    print("=" * 92)

    # ── 1. Intégrité des données ─────────────────────────────────────
    print("\n1. INTÉGRITÉ DES DONNÉES")
    fiches = charger_fiches()
    controle("fiches chargées", len(fiches) >= 40, f"{len(fiches)} fiches")

    precalc = {}
    if os.path.exists(PRECOMPUTED_JSON):
        with open(PRECOMPUTED_JSON, encoding="utf-8") as f:
            precalc = json.load(f)
    attendues = [f["id"] for f in fiches if f["status"] == "ok"]
    manquantes = [i for i in attendues if i not in precalc]
    controle("réponses pré-rédigées", not manquantes,
             f"{len(precalc)}/{len(attendues)}" +
             (f" — manquantes : {manquantes[:6]}" if manquantes else ""))

    vides = [i for i, r in precalc.items() if not r.get("fr", "").strip()]
    controle("aucune réponse vide", not vides, f"{len(vides)} vide(s)")

    from retriever import TGRRetriever
    retriever = TGRRetriever()
    nb_chunks = retriever.vectorstore._collection.count()
    controle("base vectorielle", nb_chunks >= 100, f"{nb_chunks} chunks indexés")

    # la base doit refléter les fiches actuelles
    ids_indexes = set()
    try:
        for m in retriever.vectorstore._collection.get(include=["metadatas"])["metadatas"]:
            if m.get("fiche_id"):
                ids_indexes.add(m["fiche_id"])
    except Exception:
        pass
    absentes = [f["id"] for f in fiches if f["id"] not in ids_indexes]
    controle("base à jour avec les fiches", not absentes,
             "toutes indexées" if not absentes
             else f"{len(absentes)} fiche(s) non indexée(s) → relancer ingestion.py")

    # ── 2. Couverture de la voie rapide ──────────────────────────────
    print("\n2. COUVERTURE DE LA VOIE RAPIDE (sans LLM)")
    from agent_rag import fiche_consensus, marge_ecriture
    from lexique import construire_lexique
    lexique = construire_lexique(retriever.vectorstore)   # pour la marge translangue
    total = bons = confusions = 0
    for fiche in fiches:
        for q in [fiche["probleme"]] + fiche.get("variantes", []):
            total += 1
            c = fiche_consensus(retriever.search(q, k=CONSENSUS_K),
                                marge_ecriture(q, lexique), q)
            if c is None:
                continue
            if c["fiche_id"] == fiche["id"]:
                bons += 1
            else:
                confusions += 1
    pc_bons = 100 * bons / total if total else 0
    pc_conf = 100 * confusions / total if total else 0
    controle("bonne fiche trouvée", pc_bons >= SEUIL_COUVERTURE,
             f"{pc_bons:.0f} % (seuil {SEUIL_COUVERTURE} %)")
    controle("confusions de fiche", pc_conf <= SEUIL_CONFUSION,
             f"{pc_conf:.0f} % (max {SEUIL_CONFUSION} %)")

    # ── 3. Latence et justesse du périmètre ──────────────────────────
    print("\n3. LATENCE ET JUSTESSE (questions réelles)")
    from agent_rag import AgenticRAG
    from bench_latence import HORS_SUJET, QUESTIONS
    agent = AgenticRAG(retriever=retriever)
    if not agent.llm.is_available():
        controle("Ollama disponible", False, "non joignable — contrôle 3 ignoré")
    else:
        latences, erreurs = [], []
        for q, typ in QUESTIONS:
            t = time.time()
            r = agent.answer(q)
            latences.append(time.time() - t)
            if (r["statut"] == "OUT_OF_SCOPE") != (typ == HORS_SUJET):
                erreurs.append(q)
        latences.sort()
        mediane = latences[len(latences) // 2]
        rapides = sum(1 for lat in latences if lat <= 2)
        justesse = 100 * (len(QUESTIONS) - len(erreurs)) / len(QUESTIONS)
        controle("latence médiane", mediane <= SEUIL_LATENCE_MED,
                 f"{mediane:.2f}s | max {latences[-1]:.1f}s | {rapides}/{len(latences)} sous 2 s")
        controle("justesse du périmètre", justesse >= SEUIL_JUSTESSE,
                 f"{justesse:.0f} %" + (f" — erreurs : {erreurs}" if erreurs else ""))

    # ── 4. Relecture humaine ─────────────────────────────────────────
    print("\n4. RELECTURE HUMAINE")
    import revision
    etat = revision.liste()
    # Ce contrôle reste rouge tant que la TGR n'a pas relu : ce n'est pas un
    # défaut logiciel mais la porte d'entrée en production.
    controle("réponses relues par un agent", etat["validees"] == etat["total"],
             f"{etat['validees']}/{etat['total']} validées"
             + (f" — {etat['risque_eleve']} à risque élevé en attente" if etat["risque_eleve"] else ""))

    # ── Verdict ──────────────────────────────────────────────────────
    echecs = [n for n, ok, _ in resultats if not ok]
    print("\n" + "=" * 92)
    if not echecs:
        print("  VERDICT : PASSE — le système est prêt.")
    else:
        print(f"  VERDICT : {len(echecs)} contrôle(s) en échec — {', '.join(echecs)}")
    print("=" * 92)
    return 0 if not echecs else 1


if __name__ == "__main__":
    sys.exit(main())
