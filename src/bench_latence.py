"""
Banc d'essai latence + justesse métier.

Le chargement du modèle e5 (~20 s) est payé UNE fois au démarrage du serveur :
il n'entre donc PAS dans la latence perçue par l'usager, et reste hors
chronomètre ici, exactement comme en production.

Lancement :  python -X utf8 src\bench_latence.py
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from agent_rag import AgenticRAG

HORS_SUJET = "hors sujet"

QUESTIONS = [
    # ── Questions du portail : doivent toutes être ACCEPTÉES ──
    ("Mon mot de passe ne marche plus", "fiche"),
    ("J'ai changé de téléphone et je n'ai plus mes codes", "fiche"),
    ("Comment supprimer mon compte ?", "fiche"),
    ("Mon téléphone n'a pas le NFC", "fiche"),
    ("Le lien de réinitialisation est expiré", "fiche"),
    ("Mon compte est bloqué après plusieurs tentatives", "fiche"),
    ("Je n'arrive pas à m'inscrire, mon salaire ne correspond pas", "FAQ"),
    ("Comment obtenir une attestation d'imposition ?", "FAQ"),
    ("Où télécharger ma quittance de paiement ?", "FAQ"),
    ("Comment payer ma taxe d'habitation en ligne ?", "FAQ"),
    ("كيف أحذف حسابي؟", "arabe"),
    ("ما هي مشاكل رمز التحقق؟", "arabe"),
    ("ma9dertch ndkhol l compte dyali", "darija"),
    # ── Hors sujet : doivent toutes être REFUSÉES ──
    ("Quelle est la capitale de la France ?", HORS_SUJET),
    ("Donne-moi une recette de tajine", HORS_SUJET),
    ("Raconte-moi une blague", HORS_SUJET),
    ("Quel temps fera-t-il demain a Rabat ?", HORS_SUJET),
    # CAS LIMITE ASSUMÉ : le consensus trouve la fiche « profits immobiliers »
    # et répond que ces taxes ne relèvent pas de la TGR — la redirection est
    # correcte, mais le statut devrait être OUT_OF_SCOPE. Resserrer les seuils
    # pour rattraper ce cas casse des comportements justes (mesuré) : on garde
    # le test en échec plutôt que de maquiller le résultat.
    ("Comment obtenir un credit immobilier ?", HORS_SUJET),
]


def voie_de(r):
    if any(e.get("noeud") == "direct" for e in r["etapes"]):
        return "pre-validee"
    if r["statut"] == "OUT_OF_SCOPE":
        return "guardrail"
    return "LLM"


def main():
    agent = AgenticRAG()              # hors chronometre, comme le serveur
    agent.retriever.search("prechauffage")

    print()
    print(f"{'QUESTION':<52}{'TYPE':<11}{'LATENCE':>9}  {'VOIE':<12}{'VITESSE':<9}JUSTESSE")
    print("-" * 104)
    latences, erreurs = [], 0
    for q, typ in QUESTIONS:
        start = time.time()
        r = agent.answer(q)
        lat = time.time() - start
        latences.append(lat)
        refuse = r["statut"] == "OUT_OF_SCOPE"
        correct = refuse == (typ == HORS_SUJET)
        erreurs += 0 if correct else 1
        print(f"{q[:50]:<52}{typ:<11}{lat:>7.2f}s  {voie_de(r):<12}"
              f"{'OK' if lat <= 2 else 'LENT':<9}{'OK' if correct else 'ERREUR'}")

    rapides = sum(1 for lat in latences if lat <= 2)
    print("-" * 104)
    print(f"Latence  : moyenne {sum(latences)/len(latences):.2f}s | max {max(latences):.2f}s "
          f"| <=2s : {rapides}/{len(latences)}")
    print(f"Justesse : {len(QUESTIONS) - erreurs}/{len(QUESTIONS)} verdicts corrects")


if __name__ == "__main__":
    main()
