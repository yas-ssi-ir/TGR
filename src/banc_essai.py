r"""
Banc d'essai — pose TOUTES les questions du corpus au véritable agent.

Différence avec audit_couverture.py : celui-ci ne mesure que le vote de
consensus, en isolation. Ici on appelle `AgenticRAG.answer()`, donc toute la
chaîne réelle — garde-fou de périmètre, voie rapide, voie lente, vérification
d'ancrage — exactement ce que reçoit un usager.

Ce que le script vérifie, pour chaque question :

  ROUTAGE   la fiche servie est-elle celle qui déclare savoir répondre ?
  PÉRIMÈTRE une question du portail est-elle acceptée, une question hors
            sujet refusée ?
  LATENCE   la réponse tient-elle sous les 2 secondes promises ?

Les questions hors sujet sont incluses : un assistant d'administration qui
invente hors de son domaine est plus dangereux qu'un assistant incomplet.

Sortie : un tableau des anomalies à l'écran + un rapport Markdown horodaté
dans eval/, à joindre à un compte rendu.

Lancement :
    python -X utf8 src\banc_essai.py                 # tout le corpus
    python -X utf8 src\banc_essai.py --langue ar     # arabe seulement
    python -X utf8 src\banc_essai.py --langue dj     # darija seulement
    python -X utf8 src\banc_essai.py --rapide        # 1 variante par fiche
"""
import json
import os
import re
import sys
import time
from collections import Counter
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from agent_rag import AgenticRAG
from config import (
    ASSISTANT_FICHES_JSON, EVAL_DIR, FAQ_FICHES_JSON, PRECOMPUTED_JSON,
    QA_FICHES_JSON,
)

ARABE = re.compile("[؀-ۿ]")
DARIJA = re.compile(
    r"\b(bghit|bghina|kifach|kifah|dyal|dyali|ndir|nsajjel|ndkhol|nmse7|makayn"
    r"|chikaya|mochkil|wa9ef|jdid|bdelt|telephoni)\b|[a-z][2379][a-z]", re.I)

SEUIL_LATENCE = 2.0        # promesse tenue à l'usager

# Questions qui doivent être REFUSÉES. Elles ne viennent pas du corpus : c'est
# le seul contrôle qu'un corpus ne peut pas fournir lui-même.
HORS_SUJET = [
    "Donne-moi une recette de tajine",
    "Quelle est la capitale de la France ?",
    "Raconte-moi une blague",
    "Quel temps fera-t-il demain à Rabat ?",
    "Qui a gagné le match hier soir ?",
    "Comment cuisiner un couscous ?",
    "bghit chi recette dyal tajine",
    "ما هو أفضل مطعم في الرباط؟",
]


def langue_de(question: str) -> str:
    if ARABE.search(question):
        return "ar"
    return "dj" if DARIJA.search(question) else "fr"


def charger_cas(rapide: bool, langue: str) -> list[dict]:
    """Un cas = une question + la fiche qui doit y répondre."""
    fiches = []
    for chemin in (QA_FICHES_JSON, FAQ_FICHES_JSON, ASSISTANT_FICHES_JSON):
        if os.path.exists(chemin):
            with open(chemin, encoding="utf-8") as f:
                fiches += json.load(f)

    precalc = {}
    if os.path.exists(PRECOMPUTED_JSON):
        with open(PRECOMPUTED_JSON, encoding="utf-8") as f:
            precalc = json.load(f)

    cas = []
    for fiche in fiches:
        # On n'interroge que ce que le système prétend savoir servir vite :
        # une fiche sans réponse pré-rédigée part sur la voie lente par
        # construction, ce n'est pas une anomalie mais une donnée manquante.
        if fiche.get("status") != "ok" or fiche["id"] not in precalc:
            continue
        variantes = fiche.get("variantes") or [fiche["probleme"]]
        for v in (variantes[:1] if rapide else variantes):
            if langue != "tout" and langue_de(v) != langue:
                continue
            cas.append({"question": v, "attendu": fiche["id"],
                        "categorie": fiche["categorie"], "langue": langue_de(v)})

    if langue == "tout" or langue == "fr":
        for q in HORS_SUJET:
            if langue == "tout" or langue_de(q) == langue:
                cas.append({"question": q, "attendu": None,
                            "categorie": "Hors périmètre", "langue": langue_de(q)})
    return cas


def juger(cas: dict, res: dict) -> tuple[str, str]:
    """(verdict, explication). Le verdict porte sur ce que voit l'usager."""
    servie = res["sources"][0]["id"] if res.get("sources") else None
    if cas["attendu"] is None:
        if res["statut"] == "OUT_OF_SCOPE":
            return "OK", "refusée comme prévu"
        return "ACCEPTE_A_TORT", f"acceptée → {servie or res['statut']}"
    if res["statut"] == "OUT_OF_SCOPE":
        return "REFUS_A_TORT", "question du portail refusée"
    if servie == cas["attendu"]:
        return "OK", ""
    if servie is None:
        return "SANS_SOURCE", f"statut {res['statut']}"
    return "MAUVAISE_FICHE", f"servie : {servie}"


def main():
    rapide = "--rapide" in sys.argv
    langue = "tout"
    if "--langue" in sys.argv:
        langue = sys.argv[sys.argv.index("--langue") + 1]

    cas = charger_cas(rapide, langue)
    print("=" * 96)
    print(f"  BANC D'ESSAI — {len(cas)} questions posées à l'agent complet")
    print("=" * 96)
    if not cas:
        print("Aucun cas : la chaîne de préparation a-t-elle été jouée ?")
        return

    agent = AgenticRAG()
    lignes, verdicts, latences = [], Counter(), []
    debut = time.time()

    for i, c in enumerate(cas, 1):
        t0 = time.time()
        res = agent.answer(c["question"])
        lat = time.time() - t0
        verdict, detail = juger(c, res)
        verdicts[verdict] += 1
        latences.append(lat)
        lignes.append({**c, "verdict": verdict, "detail": detail,
                       "latence": lat, "statut": res["statut"]})

        marque = "." if verdict == "OK" and lat <= SEUIL_LATENCE else "x"
        print(marque, end="", flush=True)
        if i % 80 == 0:
            print(f"  {i}/{len(cas)}")

    duree = time.time() - debut
    total = len(lignes)
    lentes = [x for x in lignes if x["latence"] > SEUIL_LATENCE]
    anomalies = [x for x in lignes if x["verdict"] != "OK"]
    latences.sort()

    print(f"\n\n{'-' * 96}")
    print(f"ROUTAGE   {verdicts['OK']}/{total} corrects ({100 * verdicts['OK'] / total:.1f} %)")
    for v in ("MAUVAISE_FICHE", "SANS_SOURCE", "REFUS_A_TORT", "ACCEPTE_A_TORT"):
        if verdicts[v]:
            print(f"          {verdicts[v]:>3} {v.lower().replace('_', ' ')}")
    print(f"LATENCE   médiane {latences[len(latences) // 2]:.2f}s | "
          f"p95 {latences[int(len(latences) * 0.95)]:.2f}s | max {max(latences):.2f}s")
    print(f"          {total - len(lentes)}/{total} sous {SEUIL_LATENCE}s")
    print(f"DURÉE     {duree / 60:.1f} min")

    if anomalies:
        print(f"\n{'-' * 96}\nANOMALIES DE ROUTAGE ({len(anomalies)})\n")
        for x in anomalies:
            print(f"  [{x['verdict']:<15}] {x['attendu'] or 'REFUS':<9} {x['detail']:<26} "
                  f"{x['question'][:44]}")
    if lentes:
        print(f"\n{'-' * 96}\nAU-DESSUS DE {SEUIL_LATENCE}s ({len(lentes)})\n")
        for x in sorted(lentes, key=lambda y: -y["latence"]):
            print(f"  {x['latence']:>6.1f}s  {x['attendu'] or 'REFUS':<9} {x['question'][:56]}")

    chemin = ecrire_rapport(lignes, verdicts, latences, lentes, anomalies, duree, langue)
    print(f"\nRapport écrit : {chemin}")

    parfait = not anomalies and not lentes
    print(f"\nVERDICT : {'PASSE' if parfait else 'À REGARDER'}")
    sys.exit(0 if parfait else 1)


def ecrire_rapport(lignes, verdicts, latences, lentes, anomalies, duree, langue) -> str:
    os.makedirs(EVAL_DIR, exist_ok=True)
    horodatage = datetime.now().strftime("%Y-%m-%d_%H%M")
    chemin = os.path.join(EVAL_DIR, f"banc_essai_{horodatage}.md")
    total = len(lignes)
    par_langue = Counter(x["langue"] for x in lignes)
    ok_langue = Counter(x["langue"] for x in lignes if x["verdict"] == "OK")

    with open(chemin, "w", encoding="utf-8") as f:
        f.write(f"# Banc d'essai — {datetime.now():%d/%m/%Y %H:%M}\n\n")
        f.write(f"{total} questions posées à l'agent complet "
                f"(filtre : {langue}), en {duree / 60:.1f} min.\n\n")
        f.write("| | Questions | Bonne fiche | Taux |\n|---|---|---|---|\n")
        for lg, nom in (("fr", "Français"), ("ar", "Arabe"), ("dj", "Darija")):
            if par_langue[lg]:
                f.write(f"| {nom} | {par_langue[lg]} | {ok_langue[lg]} | "
                        f"{100 * ok_langue[lg] / par_langue[lg]:.1f} % |\n")
        f.write(f"| **Total** | **{total}** | **{verdicts['OK']}** | "
                f"**{100 * verdicts['OK'] / total:.1f} %** |\n\n")
        f.write(f"Latence : médiane **{latences[len(latences) // 2]:.2f}s**, "
                f"p95 {latences[int(len(latences) * 0.95)]:.2f}s, "
                f"max {max(latences):.2f}s — {total - len(lentes)}/{total} "
                f"sous {SEUIL_LATENCE}s.\n\n")

        if anomalies:
            f.write(f"## Anomalies de routage ({len(anomalies)})\n\n")
            f.write("| Verdict | Attendu | Obtenu | Question |\n|---|---|---|---|\n")
            for x in anomalies:
                f.write(f"| {x['verdict']} | {x['attendu'] or 'refus'} | "
                        f"{x['detail']} | {x['question'][:70]} |\n")
            f.write("\n")
        if lentes:
            f.write(f"## Au-dessus de {SEUIL_LATENCE}s ({len(lentes)})\n\n")
            f.write("| Latence | Fiche | Question |\n|---|---|---|\n")
            for x in sorted(lentes, key=lambda y: -y["latence"]):
                f.write(f"| {x['latence']:.1f}s | {x['attendu'] or 'refus'} | "
                        f"{x['question'][:70]} |\n")
            f.write("\n")
        if not anomalies and not lentes:
            f.write("Aucune anomalie.\n")
    return chemin


if __name__ == "__main__":
    main()
