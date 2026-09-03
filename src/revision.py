"""
Relecture humaine des réponses pré-validées.

Enjeu : une réponse pré-rédigée est servie à l'usager SANS vérification à
l'exécution. Tant qu'un agent TGR ne l'a pas relue, c'est le texte d'un modèle
3B qui engage l'administration. Ce module fournit la couche de validation :

  - liste des réponses avec leur note source en regard (pour comparer)
  - alerte sur les fiches à risque : note source laconique → le modèle a dû
    combler, c'est là que naissent les procédures inventées
  - édition + validation, tracées dans reponses_precalculees.json

Champs ajoutés à chaque entrée : "validee", "validee_le", "modifiee".
"""
import json
import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (
    ASSISTANT_FICHES_JSON, FAQ_FICHES_JSON, PRECOMPUTED_JSON, QA_FICHES_JSON,
)
from retraduire_ar import proportion_arabe

# En deçà, la note source ne contient pas de procédure : le modèle a forcément
# comblé le vide. Ces fiches sont les plus exposées à l'invention.
SEUIL_NOTE_LACONIQUE = 80

# Longueur de coupe appliquée par prepare_faq.LONGUEUR_REPONSE_MAX. Une note
# qui l'atteint a perdu du texte : la coupe tombe en fin de phrase, la réponse
# se lit donc normalement, mais la suite du guide a été abandonnée — ce n'est
# plus une réponse officielle complète.
# Valeur recopiée, pas importée : prepare_faq tire langchain, que l'étape
# rapide de la CI n'installe pas. test_donnees.py vérifie qu'elles coïncident.
LONGUEUR_COUPE_FAQ = 2000


def charger_fiches() -> dict:
    fiches = {}
    for chemin in (QA_FICHES_JSON, FAQ_FICHES_JSON, ASSISTANT_FICHES_JSON):
        if os.path.exists(chemin):
            with open(chemin, encoding="utf-8") as f:
                for fi in json.load(f):
                    fiches[fi["id"]] = fi
    return fiches


def charger_reponses() -> dict:
    if os.path.exists(PRECOMPUTED_JSON):
        with open(PRECOMPUTED_JSON, encoding="utf-8") as f:
            return json.load(f)
    return {}


def enregistrer_reponses(d: dict):
    with open(PRECOMPUTED_JSON, "w", encoding="utf-8") as f:
        json.dump(d, f, ensure_ascii=False, indent=2)


def liste() -> dict:
    """Toutes les réponses à relire, avec leur note source et leur niveau de risque."""
    fiches, reponses = charger_fiches(), charger_reponses()
    items = []
    for fid, rep in reponses.items():
        fiche = fiches.get(fid, {})
        note = (fiche.get("solution") or "").strip()
        # Une fiche tronquée n'est PAS un texte officiel complet : la classer
        # « risque : aucun » la ferait sauter à la relecture alors qu'elle
        # s'arrête au milieu d'une phrase.
        tronquee = fid.startswith("FAQ.") and len(note) >= LONGUEUR_COUPE_FAQ
        verbatim = (fid.startswith("AST.")
                    or (fid.startswith("FAQ.") and len(note) >= 150)) and not tronquee
        items.append({
            "id": fid,
            "categorie": fiche.get("categorie", "?"),
            "probleme": fiche.get("probleme", rep.get("probleme", "")),
            "note_source": note,
            "fr": rep.get("fr", ""),
            "ar": rep.get("ar", ""),
            "validee": rep.get("validee", False),
            "modifiee": rep.get("modifiee", False),
            "validee_le": rep.get("validee_le", ""),
            # verbatim = texte officiel publié, recopié tel quel : risque nul
            "risque": ("aucun" if verbatim
                       else "eleve" if tronquee or len(note) < SEUIL_NOTE_LACONIQUE
                       else "moyen"),
            "origine": (
                "assistant eServices (verbatim)" if verbatim and fid.startswith("AST.")
                else "FAQ officielle (verbatim)" if verbatim
                else "FAQ officielle TRONQUÉE — à reprendre" if tronquee
                else "rédigée par le modèle"
            ),
            # une traduction ratée laisse du français dans le champ arabe :
            # l'usager arabophone recevrait une réponse qu'il ne peut pas lire
            "ar_douteuse": proportion_arabe(rep.get("ar", "")) < 0.30,
        })
    ordre = {"eleve": 0, "moyen": 1, "aucun": 2}
    items.sort(key=lambda x: (x["validee"], ordre[x["risque"]], x["id"]))
    return {
        "total": len(items),
        "validees": sum(1 for x in items if x["validee"]),
        "risque_eleve": sum(1 for x in items if x["risque"] == "eleve" and not x["validee"]),
        "ar_douteuses": sum(1 for x in items if x["ar_douteuse"]),
        "fiches": items,
    }


def enregistrer(fiche_id: str, fr: str, ar: str, valider: bool,
                devalider: bool = False) -> bool:
    reponses = charger_reponses()
    if fiche_id not in reponses:
        return False
    rep = reponses[fiche_id]
    if fr.strip() and fr.strip() != rep.get("fr", ""):
        rep["fr"], rep["modifiee"] = fr.strip(), True
    if ar.strip() and ar.strip() != rep.get("ar", ""):
        rep["ar"], rep["modifiee"] = ar.strip(), True
        # « ar_defaillante » veut dire « le modèle a renoncé à traduire ». Un
        # relecteur qui écrit lui-même l'arabe lève ce constat : sans cela la
        # fiche resterait signalée comme non traduite, et la relecture
        # suivante retomberait dessus pour rien.
        if proportion_arabe(rep["ar"]) >= 0.30:
            rep.pop("ar_defaillante", None)
    if valider:
        rep["validee"] = True
        rep["validee_le"] = datetime.now().strftime("%Y-%m-%d %H:%M")
    elif devalider:
        rep["validee"], rep["validee_le"] = False, ""
    elif rep.get("modifiee") and rep.get("validee"):
        # Le vert certifie UN texte, pas une fiche : si le texte change, la
        # signature de la TGR ne le couvre plus et la fiche repart à relire.
        rep["validee"], rep["validee_le"] = False, ""
    enregistrer_reponses(reponses)
    return True


if __name__ == "__main__":
    etat = liste()
    print(f"{etat['validees']}/{etat['total']} réponses validées "
          f"— {etat['risque_eleve']} à risque élevé restantes")
    for f in etat["fiches"][:10]:
        print(f"  [{f['risque']:<6}] {f['id']:<8} {f['probleme'][:58]}")
