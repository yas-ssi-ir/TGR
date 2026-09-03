"""
Pré-rédaction HORS LIGNE des réponses officielles — à lancer UNE SEULE fois
(puis à relancer uniquement si les fiches changent).

Pourquoi : sur CPU, le LLM met 40-100 s à rédiger une réponse. En production
on ne fait jamais ça en direct pour des problèmes déjà connus. Ce script fait
rédiger au LLM, tranquillement hors ligne, la réponse officielle de CHAQUE
fiche (en français ET en arabe). Au runtime, l'assistant sert ces réponses
pré-validées instantanément (< 1 s) dès que la preuve vectorielle est forte.

Résultat : data/processed/reponses_precalculees.json
  { "1.1.1": {"fr": "...", "ar": "...", "probleme": "..."}, ... }

Reprise sur interruption : le fichier est sauvegardé après CHAQUE fiche —
on peut arrêter (Ctrl+C) et relancer, les fiches déjà rédigées sont sautées.

Lancement :  python -X utf8 src\\precompute_answers.py
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (
    ASSISTANT_FICHES_JSON, FAQ_FICHES_JSON, PRECOMPUTED_JSON, QA_FICHES_JSON,
)
from llm import OllamaLLM
from nettoyer_reponses import nettoyer
from retraduire_ar import preserver_liens, traduire
from retriever import TGRRetriever

PRECOMPUTE_SYSTEM = """Tu es l'assistant officiel du portail eServices de la Trésorerie Générale du Royaume du Maroc (TGR).
Reformule la note de résolution en une réponse claire pour l'usager.

INTERDICTIONS ABSOLUES (une réponse pré-validée est servie sans relecture) :
- Ne JAMAIS inventer une étape, un écran, un menu, un bouton, un lien, un délai
  ou un montant qui ne figure pas mot pour mot dans la documentation fournie.
- Ne JAMAIS décrire une manipulation dans un service tiers (Google, opérateur,
  banque) si la documentation ne la décrit pas.
- Si la documentation n'indique QUE la cause du problème, tu expliques la cause
  et tu orientes vers le support : tu n'inventes pas la marche à suivre.

RÈGLES DE FORME :
1. Commence directement par l'information utile — ni « Bonjour », ni signature.
2. Numérote les étapes seulement si la documentation en décrit plusieurs.
3. Ne dis jamais « passage », « fiche » ou « documentation fournie ».
4. 3 à 6 phrases, ton d'un agent au guichet : simple, poli, précis."""


def reponse_officielle(fiche: dict) -> str | None:
    """Une réponse DÉJÀ officielle ne doit pas être reformulée par un modèle :
    on la sert telle quelle. Zéro risque d'invention, zéro appel LLM.

    Deux cas :
      - AST.*  → texte réellement affiché par l'assistant en production.
                 C'est la source la plus sûre du projet : TOUJOURS verbatim,
                 quelle que soit sa longueur (un lien seul est une réponse).
      - FAQ.*  → réponse de la FAQ publiée, si elle est assez substantielle.

    Seules les notes internes laconiques du fichier des réclamations sont
    confiées au LLM.
    """
    sol = (fiche.get("solution") or "").strip()
    if not sol:
        return None
    if fiche["id"].startswith("AST."):
        return sol
    return sol if fiche["id"].startswith("FAQ.") and len(sol) >= 150 else None


def build_context_for(fiche: dict, retriever: TGRRetriever) -> str:
    """Contexte = la fiche + jusqu'à 2 extraits proches de la FAQ officielle."""
    parts = [
        f"FICHE OFFICIELLE ({fiche['categorie']})\n"
        f"Problème : {fiche['probleme']}\n"
        f"Note interne de résolution : {fiche['solution']}"
    ]
    complements = 0
    for p in retriever.search(f"{fiche['probleme']} {fiche['solution']}", k=6):
        if p["fichier"] and p["distance"] <= 0.40:
            parts.append(f"EXTRAIT DE LA FAQ OFFICIELLE ({p['fichier']}) :\n{p['text']}")
            complements += 1
            if complements >= 2:
                break
    return "\n\n".join(parts)


def main():
    fiches = []
    for chemin in (QA_FICHES_JSON, FAQ_FICHES_JSON, ASSISTANT_FICHES_JSON):
        if os.path.exists(chemin):
            with open(chemin, encoding="utf-8") as f:
                fiches += json.load(f)

    done = {}
    if os.path.exists(PRECOMPUTED_JSON):
        with open(PRECOMPUTED_JSON, encoding="utf-8") as f:
            done = json.load(f)
        print(f"Reprise : {len(done)} fiche(s) déjà rédigée(s), on continue.")

    retriever = TGRRetriever()
    llm = OllamaLLM()
    if not llm.is_available():
        print("Ollama indisponible — lancez Ollama d'abord.")
        sys.exit(1)

    # Les identifiants FAQ.* sont POSITIONNELS : ajouter une question au guide
    # décale tous les suivants. Une réponse déjà rédigée se retrouverait alors
    # attachée à un autre problème — sans bruit, et servie telle quelle aux
    # usagers. On compare donc l'intitulé mémorisé à celui de la fiche : s'ils
    # diffèrent, l'entrée est périmée et doit être refaite.
    connus = {fi["id"] for fi in fiches}
    perimees = [fid for fid, rep in done.items()
                if fid in connus and rep.get("probleme")
                and rep["probleme"] != next(f["probleme"] for f in fiches if f["id"] == fid)]
    orphelines = [fid for fid in done if fid not in connus]

    # L'intitulé ne suffit pas : réparer l'analyseur du guide raccourcit des
    # RÉPONSES sans toucher aux questions. Constaté sur FAQ.1, qui a continué
    # de servir 1399 caractères — l'inscription, puis l'erreur de salaire CNT,
    # puis le format de la date de naissance — alors que sa source officielle
    # était retombée à 288. Pour les fiches servies verbatim, la source FAIT
    # foi ; sauf si un relecteur humain est passé après, auquel cas c'est SON
    # texte qui fait foi et qu'on ne réécrit jamais.
    par_id = {f["id"]: f for f in fiches}
    # une même fiche peut relever des DEUX contrôles — la question a changé de
    # place ET son texte diffère de la source. Sans cette exclusion, elle était
    # supprimée deux fois : KeyError, et la reprise s'arrêtait net.
    deja_vues = set(perimees) | set(orphelines)
    desynchronisees = [
        fid for fid, rep in done.items()
        if fid in connus and fid not in deja_vues
        and not rep.get("modifiee") and not rep.get("validee")
        and (off := reponse_officielle(par_id[fid])) is not None
        and off != rep.get("fr", "").strip()
    ]
    for fid in perimees + orphelines + desynchronisees:
        del done[fid]
    if desynchronisees:
        print(f"{len(desynchronisees)} réponse(s) désynchronisée(s) de leur source "
              f"officielle → à refaire : {', '.join(sorted(desynchronisees))}")
    if perimees:
        print(f"{len(perimees)} réponse(s) périmée(s) (l'identifiant désigne désormais "
              f"un autre problème) → à refaire : {', '.join(sorted(perimees))}")
    if orphelines:
        print(f"{len(orphelines)} réponse(s) orpheline(s) supprimée(s) : "
              f"{', '.join(sorted(orphelines))}")

    a_faire = [fi for fi in fiches if fi["status"] == "ok" and fi["id"] not in done]
    print(f"\n{len(a_faire)} fiche(s) à rédiger (FR + AR). "
          f"Comptez ~1 min/fiche sur CPU — c'est un coût payé UNE seule fois.\n")

    for i, fiche in enumerate(a_faire, 1):
        start = time.time()
        ctx = build_context_for(fiche, retriever)
        question = fiche["variantes"][0] if fiche.get("variantes") else fiche["probleme"]
        base_prompt = (f"DOCUMENTATION :\n{ctx}\n\n"
                       f"PROBLÈME DE L'USAGER : {question}\n\n")

        officielle = reponse_officielle(fiche)
        if officielle:
            fr = officielle                       # texte publié : verbatim
            origine = ("verbatim assistant" if fiche["id"].startswith("AST.")
                       else "verbatim FAQ")
        else:
            fr = llm.generate(PRECOMPUTE_SYSTEM,
                              base_prompt + "Rédige la réponse officielle en FRANÇAIS.")
            # le modèle ajoute préambules et salutations malgré la consigne :
            # on les retire mécaniquement plutôt que d'espérer qu'il obéisse
            fr = nettoyer(fr)
            origine = "rédaction LLM"

        # traduction VÉRIFIÉE : sans contrôle, le modèle recopie le français
        # (constaté sur 25 fiches sur 44)
        ar = traduire(llm, fr)
        if ar is None:
            ar, origine = fr, origine + " — traduction arabe en échec"
        else:
            ar = preserver_liens(fr, ar)

        done[fiche["id"]] = {"fr": fr.strip(), "ar": ar.strip(),
                             "probleme": fiche["probleme"]}
        # sauvegarde après CHAQUE fiche → interruptible sans perte
        with open(PRECOMPUTED_JSON, "w", encoding="utf-8") as f:
            json.dump(done, f, ensure_ascii=False, indent=2)

        print(f"[{i}/{len(a_faire)}] fiche {fiche['id']} : {origine} "
              f"({time.time() - start:.0f}s)")

    print(f"\nTerminé — {len(done)} réponses officielles pré-rédigées dans :\n  {PRECOMPUTED_JSON}")
    print("L'assistant les servira désormais en moins d'une seconde. ⚡")


if __name__ == "__main__":
    main()
