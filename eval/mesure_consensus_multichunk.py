"""Mesure ponctuelle (pas un script du pipeline) : distribution des distances
des consensus multi-chunks, legitimes (auto-test des fiches) vs adverses
(20 questions inedites), pour choisir un seuil de recevabilite justifie par
la mesure plutot que par supposition. Voir tache A de la delegation."""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + r"\..\src")
os.chdir(os.path.dirname(os.path.abspath(__file__)) + r"\..")

from config import ASSISTANT_FICHES_JSON, CONSENSUS_K, CONSENSUS_MIN_CHUNKS, DIST_SOLO_ACCEPT, FAQ_FICHES_JSON, QA_FICHES_JSON
from agent_rag import fiche_consensus
from lexique import construire_lexique, marge_ecriture
from retriever import TGRRetriever

ADVERSAIRES = [
    "Je viens d'etre recrute a la fonction publique, quelles demarches dois-je faire sur le portail pour commencer a toucher mon salaire ?",
    "Quelle est la difference entre la taxe d'habitation et la taxe des services communaux ?",
    "Je veux payer ma taxe sur les terrains non batis ET verifier si j'ai bien recu ma quittance apres, comment faire les deux ?",
    "Un huissier me reclame le paiement d'une amende radar datant de plus de 2 mois, que dois-je faire ?",
    "Quels articles du decret n. 2-22-431 encadrent le retrait d'une offre apres depot ?",
    "J'aimerais comprendre comment presenter mon entreprise aux administrations publiques sans passer par un appel d'offres classique",
    "Ma pension de retraite n'apparait pas sur mon espace, a qui dois-je m'adresser ?",
    "Comment un fonctionnaire peut-il consulter a la fois sa situation administrative, sa situation familiale et son dernier bulletin de paie en une seule fois ?",
    "Peut-on payer une amende radar recue par une entreprise avec le compte personnel du gerant ?",
    "Je voudrais savoir si la TGR peut m'aider a etaler le paiement d'une dette fiscale importante",
    "Est-ce que je peux deleguer a mon comptable l'acces a mon espace eServices pour qu'il paie mes taxes a ma place ?",
    "J'ai deja un compte eServices mais je veux aussi m'inscrire comme fournisseur pour repondre a des marches publics, dois-je creer un second compte ?",
    "Quel est le delai legal de traitement d'une reclamation en ligne avant de la considerer rejetee par defaut ?",
    "Je suis a l'etranger et je n'ai pas acces a une agence bancaire TGR, comment ouvrir un compte a distance ?",
    "Comment fonctionne le mecanisme de preference nationale dans l'attribution des marches publics de la TGR ?",
    "Quels sont les taux de change appliques par la TGR pour les paiements recus en devises etrangeres ?",
    "La TGR verse-t-elle des interets moratoires en cas de retard de paiement d'un fournisseur ?",
    "Comment un syndic de copropriete peut-il payer la taxe de sejour pour le compte de plusieurs proprietaires ?",
    "Quelle est la procedure de recours si un marche public est attribue a un concurrent alors que mon offre etait moins-disante ?",
    "Un fonctionnaire detache aupres d'un organisme international continue-t-il a percevoir son salaire via la TGR ?",
    "Comment obtenir un credit immobilier ?",  # cas deja documente comme echec connu
]

print("Chargement du retriever...")
retriever = TGRRetriever()
lexique = construire_lexique(retriever.vectorstore)

def fiches_toutes():
    out = []
    for chemin in (QA_FICHES_JSON, FAQ_FICHES_JSON, ASSISTANT_FICHES_JSON):
        if os.path.exists(chemin):
            out += json.load(open(chemin, encoding="utf-8"))
    return [f for f in out if f.get("status") != "menu"]

fiches = fiches_toutes()
print(f"{len(fiches)} fiches chargees\n")

# ---- 1) distribution LEGITIME : consensus multi-chunks corrects sur l'auto-test ----
print("Mesure 1/2 : auto-test de toutes les fiches (peut prendre 1-2 min)...")
t0 = time.time()
legit_multi_correct = []   # best_distance des consensus a >=2 votes ET corrects
legit_multi_incorrect = [] # best_distance des consensus a >=2 votes MAIS fiche erronee (confusions existantes)
total = bons = confusions = solo = 0
for fiche in fiches:
    for q in [fiche["probleme"]] + fiche.get("variantes", []):
        total += 1
        passages = retriever.search(q, k=CONSENSUS_K)
        marge = marge_ecriture(q, lexique)
        c = fiche_consensus(passages, marge, q)
        if c is None:
            continue
        if c["votes"] < 2:
            solo += 1
            continue
        if c["fiche_id"] == fiche["id"]:
            bons += 1
            legit_multi_correct.append(c["best_distance"])
        else:
            confusions += 1
            legit_multi_incorrect.append(c["best_distance"])
print(f"  -> {total} tests, {bons} bons, {confusions} confusions, {solo} via solo-accept "
      f"({time.time()-t0:.0f}s)")
print(f"  Couverture actuelle (reference verifier_tout.py) : {100*bons/total:.1f}% bons, "
      f"{100*confusions/total:.1f}% confusions\n")

# ---- 2) distribution ADVERSE : consensus multi-chunks sur les 20 questions inedites ----
print("Mesure 2/2 : les 21 questions adversariales...")
adv_multi = []  # (question, fiche_id, votes, best_distance)
for q in ADVERSAIRES:
    passages = retriever.search(q, k=CONSENSUS_K)
    marge = marge_ecriture(q, lexique)
    c = fiche_consensus(passages, marge, q)
    if c and c["votes"] >= 2:
        adv_multi.append((q, c["fiche_id"], c["votes"], c["best_distance"]))
        print(f"  MULTI-CHUNK : {q[:55]:55s} -> {c['fiche_id']:8s} votes={c['votes']} dist={c['best_distance']:.4f}")
    elif c:
        print(f"  solo-accept : {q[:55]:55s} -> {c['fiche_id']:8s} dist={c['best_distance']:.4f} (hors de portee de cette tache)")
    else:
        print(f"  AUCUN CONSENSUS : {q[:55]:55s} -> echapperait deja a la voie rapide")

print("\n" + "=" * 90)
print("DISTRIBUTIONS (best_distance des consensus a >=2 votes)")
print("=" * 90)

def stats(nom, xs):
    if not xs:
        print(f"{nom} : aucune donnee")
        return
    xs = sorted(xs)
    n = len(xs)
    print(f"{nom} (n={n}) : min={xs[0]:.4f} p10={xs[n//10]:.4f} p25={xs[n//4]:.4f} "
          f"median={xs[n//2]:.4f} p75={xs[3*n//4]:.4f} p90={xs[9*n//10] if n>=10 else xs[-1]:.4f} max={xs[-1]:.4f}")

stats("LEGITIME correct  (doit rester accepte)", legit_multi_correct)
stats("LEGITIME confusion (deja faux aujourd'hui, hors sujet de cette tache)", legit_multi_incorrect)
adv_dists = [d for _, _, _, d in adv_multi]
stats("ADVERSE (on veut qu'ils echappent au fast-path)", adv_dists)

print()
print("Recherche d'un seuil separant les deux distributions legitime-correct / adverse :")
for seuil in [0.28, 0.29, 0.30, 0.31, 0.32, 0.33, 0.34, 0.35, 0.36]:
    perdus_legit = sum(1 for d in legit_multi_correct if d > seuil)
    gagnes_adv = sum(1 for d in adv_dists if d > seuil)
    pc_perdus = 100*perdus_legit/len(legit_multi_correct) if legit_multi_correct else 0
    print(f"  seuil={seuil:.2f} -> perd {perdus_legit}/{len(legit_multi_correct)} legitimes "
          f"({pc_perdus:.1f}%) | exclut {gagnes_adv}/{len(adv_dists)} adverses en multi-chunk")
