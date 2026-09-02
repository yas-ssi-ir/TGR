r"""
Pourquoi l'assistant a-t-il répondu ça ?

Affiche, pour une ou plusieurs questions, TOUT ce qui a conduit à la réponse :
les voisins trouvés, qui a le droit de voter, qui l'emporte et pourquoi, la
décision du guardrail, et la voie empruntée. Aucun appel au modèle : le
diagnostic est instantané.

Lancement :
    python -X utf8 src\diagnostic.py                       (les cas connus)
    python -X utf8 src\diagnostic.py "ma question à moi"
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (
    CONSENSUS_K, CONSENSUS_MIN_CHUNKS, DIST_CANDIDATE_MAX, DIST_DEPARTAGE,
    DIST_HORS_SUJET, DIST_SOLO_ACCEPT,
)
from agent_rag import AgenticRAG, fiche_consensus, langue_de, marge_ecriture
from lexique import contient_terme_noyau, recouvrement

# Cas ayant échoué à l'évaluation : à garder sous la main pour vérifier
# qu'une correction règle bien le problème visé sans en créer d'autres.
# « question » et, quand on la connaît, la fiche qui AURAIT dû répondre : le
# rang réel de cette fiche est l'information décisive quand la réponse est
# mauvaise sans qu'on comprenne pourquoi.
CAS_CONNUS = [
    ("Mon mot de passe est refusé alors que je suis sûr qu'il est correct, pourquoi ?", "1.1.1"),
    ("J'ai cliqué sur le lien de réinitialisation reçu par email mais il ne marche pas.", "1.1.2"),
    ("Mon code OTP est toujours invalide, il ne marche jamais.", "2.3.1"),
]

# On regarde plus loin que la fenêtre de vote pour situer la fiche attendue
PROFONDEUR_ENQUETE = 30


def situer_fiche_attendue(agent: AgenticRAG, question: str, fiche_attendue: str, plafond: float):
    """Où se classe la fiche qui aurait dû répondre ? Si elle est hors de la
    fenêtre de vote, aucun réglage de seuil ne la sauvera : c'est la recherche
    elle-même qui l'a manquée."""
    profond = agent.retriever.search(question, k=PROFONDEUR_ENQUETE)
    rangs = [(i, p) for i, p in enumerate(profond, 1) if p["fiche_id"] == fiche_attendue]
    print(f"\n  FICHE ATTENDUE : {fiche_attendue}")
    if not rangs:
        print(f"    introuvable dans les {PROFONDEUR_ENQUETE} plus proches — "
              "la recherche documentaire elle-même la manque")
        return
    for rang, p in rangs[:4]:
        vote = "vote" if p["distance"] <= plafond else "hors fenêtre de vote"
        print(f"    rang {rang:>2} · distance {p['distance']:.3f} · {vote:<21} {p['text'][:44]!r}")
    dans_fenetre = sum(1 for _, p in rangs if p["distance"] <= plafond)
    print(f"    → {dans_fenetre} chunk(s) en mesure de voter "
          f"(il en faut {CONSENSUS_MIN_CHUNKS} pour emporter la décision)")


def diagnostiquer(agent: AgenticRAG, question: str, fiche_attendue: str = ""):
    print("=" * 96)
    print(f"QUESTION : {question}")
    marge = marge_ecriture(question, agent.lexique)
    plafond = DIST_CANDIDATE_MAX + marge
    print(f"  langue={langue_de(question)}  marge translangue={marge:.2f}  "
          f"→ un chunk vote s'il est à ≤ {plafond:.2f}")
    print(f"  terme métier reconnu : {contient_terme_noyau(question)}   "
          f"mots communs avec la doc : {recouvrement(question, agent.lexique)}")

    passages = agent.retriever.search(question, k=CONSENSUS_K)
    affiches = 14      # le vote porte sur tous ; on n'en montre que le haut
    print(f"\n  VOISINS TROUVÉS (les {min(affiches, len(passages))} premiers "
          f"sur {len(passages)} examinés)")
    for p in passages[:affiches]:
        fiche = p["fiche_id"] or p["fichier"]
        if not p["fiche_id"]:
            marque = "extrait PDF — ne vote pas"
        elif p["distance"] > plafond:
            marque = "trop loin — ne vote pas"
        else:
            marque = "VOTE" + ("  [fiche sans solution]" if p["status"] == "no_answer" else "")
        print(f"    {p['distance']:.3f}  {fiche:<14} {marque:<28} {p['text'][:44]!r}")

    consensus = fiche_consensus(passages, marge, question)
    print("\n  DÉCISION")
    if consensus is None:
        print(f"    aucune fiche recevable (il faut {CONSENSUS_MIN_CHUNKS} chunks concordants,")
        print(f"    ou un seul à ≤ {DIST_SOLO_ACCEPT + marge:.2f}) → voie lente : le modèle rédige")
    else:
        print(f"    fiche retenue : {consensus['fiche_id']}  "
              f"({consensus['votes']} chunks, meilleure distance {consensus['best_distance']:.3f})")
        rivales = {}
        for p in passages:
            if p["fiche_id"] and p["distance"] <= plafond:
                d = rivales.get(p["fiche_id"])
                if d is None or p["distance"] < d:
                    rivales[p["fiche_id"]] = p["distance"]
        rivales.pop(consensus["fiche_id"], None)
        if rivales:
            detail = ", ".join(f"{f} à {d:.3f}" for f, d in sorted(rivales.items(),
                                                                   key=lambda kv: kv[1]))
            print(f"    fiches écartées : {detail}")
            print(f"    (une fiche sans solution cède la place à une solution si l'écart "
                  f"est ≤ {DIST_DEPARTAGE})")
        print(f"    réponse pré-rédigée disponible : "
              f"{consensus['fiche_id'] in agent.reponses_precalculees}")

    if fiche_attendue and (consensus is None or consensus["fiche_id"] != fiche_attendue):
        situer_fiche_attendue(agent, question, fiche_attendue, plafond)

    in_scope, methode = agent.node_guardrail(question, consensus,
                                             passages[0]["distance"] if passages else None)
    print(f"\n  PÉRIMÈTRE : {'accepté' if in_scope else 'REFUSÉ'} — {methode}")
    print(f"    (sans terme métier ni consensus, une distance ≥ "
          f"{DIST_HORS_SUJET + marge:.2f} vaut refus)")


def main():
    cas = [(q, "") for q in sys.argv[1:]] or CAS_CONNUS
    agent = AgenticRAG()
    for question, attendue in cas:
        diagnostiquer(agent, question, attendue)
    print("=" * 96)


if __name__ == "__main__":
    main()
