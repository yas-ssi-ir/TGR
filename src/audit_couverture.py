"""
Audit de couverture de la VOIE RAPIDE — sans aucun appel LLM.

Question à laquelle ce script répond : pour chaque question connue de la base
(le libellé officiel du problème ET chaque variante en langage usager), le
vote de consensus désigne-t-il la BONNE fiche ?

C'est la mesure qui compte : une fiche que le consensus ne retrouve pas est
une fiche qui retombera sur la voie lente (40-100 s) ou, pire, qui renverra la
réponse d'une autre fiche.

Lancement :  python -X utf8 src\audit_couverture.py
"""
import json
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from agent_rag import fiche_consensus, marge_ecriture
from lexique import construire_lexique
from config import (
    ASSISTANT_FICHES_JSON, CONSENSUS_K, FAQ_FICHES_JSON, PRECOMPUTED_JSON,
    QA_FICHES_JSON,
)
from retriever import TGRRetriever


def charger_fiches() -> list[dict]:
    fiches = []
    # Les trois sources, sinon l'audit ne porte que sur la moitié du corpus.
    # Les nœuds de menu du relevé de l'assistant ne sont pas indexés : les
    # auditer reviendrait à exiger qu'on retrouve une fiche absente de la base.
    for chemin in (QA_FICHES_JSON, FAQ_FICHES_JSON, ASSISTANT_FICHES_JSON):
        if os.path.exists(chemin):
            with open(chemin, encoding="utf-8") as f:
                fiches += json.load(f)
    # « menu » = nœud de navigation du relevé, exclu de la base vectorielle
    return [f for f in fiches if f.get("status") != "menu"]


def main():
    fiches = charger_fiches()
    retriever = TGRRetriever()
    lexique = construire_lexique(retriever.vectorstore)   # pour la marge translangue
    precalc = {}
    if os.path.exists(PRECOMPUTED_JSON):
        with open(PRECOMPUTED_JSON, encoding="utf-8") as f:
            precalc = json.load(f)

    total = bons = muets = confusions = 0
    problemes = []
    par_fiche = Counter()

    for fiche in fiches:
        requetes = [fiche["probleme"]] + fiche.get("variantes", [])
        for q in requetes:
            total += 1
            c = fiche_consensus(retriever.search(q, k=CONSENSUS_K),
                                marge_ecriture(q, lexique), q)
            if c is None:
                muets += 1
                problemes.append(("MUET   ", fiche["id"], q, "-"))
            elif c["fiche_id"] != fiche["id"]:
                confusions += 1
                problemes.append(("CONFUSION", fiche["id"], q, c["fiche_id"]))
            else:
                bons += 1
                par_fiche[fiche["id"]] += 1

    print("\n" + "=" * 92)
    print("AUDIT DE COUVERTURE DE LA VOIE RAPIDE (aucun appel LLM)")
    print("=" * 92)
    print(f"Fiches dans la base            : {len(fiches)}")
    print(f"Questions testées              : {total} (libellés officiels + variantes usager)")
    print(f"Bonne fiche trouvée            : {bons:3d}  ({100*bons/total:.0f} %) -> reponse en ~0,1 s")
    print(f"Aucun consensus (voie lente)   : {muets:3d}  ({100*muets/total:.0f} %) -> 40-100 s")
    print(f"Mauvaise fiche (CONFUSION)     : {confusions:3d}  ({100*confusions/total:.0f} %) -> risque metier")

    sans_precalc = [f["id"] for f in fiches
                    if f["status"] == "ok" and f["id"] not in precalc]
    jamais_trouvees = [f["id"] for f in fiches if par_fiche[f["id"]] == 0]
    print(f"\nFiches jamais retrouvées       : {len(jamais_trouvees)} {jamais_trouvees[:12]}")
    print(f"Fiches sans réponse pré-rédigée: {len(sans_precalc)} {sans_precalc[:12]}")

    if problemes:
        print("\n--- Détail des cas à corriger " + "-" * 62)
        for typ, attendu, q, obtenu in problemes[:30]:
            print(f"  {typ}  fiche {attendu:<8} obtenu {obtenu:<8} | {q[:58]}")
        if len(problemes) > 30:
            print(f"  ... et {len(problemes) - 30} autres")


if __name__ == "__main__":
    main()
