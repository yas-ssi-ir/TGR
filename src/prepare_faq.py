r"""
Extraction des questions/réponses de la FAQ officielle (FAQ_TGR.pdf) en FICHES.

Pourquoi : une fiche est l'unité de connaissance du système. Elle bénéficie du
vote de consensus ET d'une réponse officielle pré-rédigée → réponse en < 1 s.
Tant que la FAQ restait un simple PDF découpé en morceaux, ses questions
retombaient sur la voie lente (45-55 s mesurées).

La FAQ est structurée en « question ? » suivie de sa réponse. On délimite donc
chaque question sur le point d'interrogation, et la réponse court jusqu'à la
question suivante.

Sortie : data/processed/faq_fiches.json (même schéma que qa_fiches.json)
Lancement :  python -X utf8 src\prepare_faq.py
"""
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import DATA_PROCESSED_DIR, DATA_RAW_DIR

from langchain_community.document_loaders import PyPDFLoader

FAQ_PDF = os.path.join(DATA_RAW_DIR, "FAQ_TGR.pdf")
FAQ_FICHES_JSON = os.path.join(DATA_PROCESSED_DIR, "faq_fiches.json")

# Les six parties du guide → catégorie de la fiche
SECTIONS = [
    ("L’utilisateur des téléservices", "Téléservices & inscription"),
    ("Redevable/contribuable", "Taxes & contribuable"),
    ("incident de paiement", "Activité bancaire"),
    ("Ordonnateur/Titulaire d’une commande publique", "Commande publique"),
    ("Fonctionnaire Je souhaite", "Fonctionnaire"),
    ("Avant de réclamer", "Réclamations"),
]

BRUIT = re.compile(
    r"servicequalité@tgr\.gov\.ma|www\.tgr\.gov\.ma|^\d{1,2}$|^\s*$", re.MULTILINE
)


def texte_faq() -> str:
    brut = "\n".join(p.page_content for p in PyPDFLoader(FAQ_PDF).load())
    brut = brut.replace("", "-")                 # puces PDF
    brut = re.sub(r"\.{4,}\s*\d*", " ", brut)          # lignes de sommaire
    brut = BRUIT.sub(" ", brut)
    brut = re.sub(r"[ \t]+", " ", brut)
    return re.sub(r"\s*\n\s*", " ", brut).strip()      # tout à plat


def categorie_a(position: int, texte: str) -> str:
    """Section du guide dans laquelle tombe cette position.
    On cherche la DERNIÈRE occurrence du titre : la première est celle du
    sommaire, qui donnerait la même (mauvaise) section à toutes les fiches."""
    courante = "Portail TGR"
    depart = fin_du_sommaire(texte)
    for titre, cat in SECTIONS:
        i = texte.find(titre, depart)
        if 0 <= i <= position:
            courante = cat
    return courante


def fin_du_sommaire(texte: str) -> int:
    """Le sommaire cite les six titres : les y chercher donnerait la même
    section à toutes les fiches. On démarre donc après l'avant-propos."""
    for marqueur in ("lecture exhaustive", "utilisation aisée"):
        i = texte.find(marqueur)
        if i > 0:
            return i
    return 0


# Numéro de page et titre de section collés devant une question
PREFIXE_PARASITE = re.compile(
    r"^\s*\d{0,2}\s*(?:" + "|".join(re.escape(t) for t, _ in SECTIONS) +
    r"|Fonctionnaire|Avant de réclamer)?\s*", re.IGNORECASE)

# Une vraie question commence par une majuscule ou un mot interrogatif
DEBUT_VALIDE = re.compile(
    r"^(?:Comment|Quel|Quelle|Quels|Quelles|Que |Qui |Où|Pourquoi|Est-ce|"
    r"Je |J’|J'|Nous |En |Dans quel|A quel|À quel)", re.IGNORECASE)


def premiere_phrase(reponse: str) -> str:
    """Début de la réponse — sert de variante pour rendre repérable une
    question qui ne se suffit pas à elle-même (« Que veut dire cette
    mention ? » ne désigne aucun sujet ; sa réponse, si)."""
    texte = reponse.strip()
    phrase = re.split(r"(?<=[.!?]) ", texte, maxsplit=1)[0].strip()
    if 25 <= len(phrase) <= 280:
        return phrase
    return texte[:200].rsplit(" ", 1)[0] if len(texte) >= 40 else ""


def extraire() -> list[dict]:
    texte = texte_faq()
    # Une question se termine par « ? » ; elle commence après la ponctuation
    # forte précédente (fin de la réponse d'avant).
    spans = []
    for m in re.finditer(r"\?", texte):
        fin = m.end()
        debut = max(texte.rfind(c, 0, m.start()) for c in ".?!:")
        debut = debut + 1 if debut > 0 else max(0, m.start() - 200)
        question = texte[debut:fin].strip(" -–—\"«»,;")
        question = PREFIXE_PARASITE.sub("", question, count=1).strip()
        # écarter les fragments issus d'une coupure de ligne du PDF
        if not (15 <= len(question) <= 220) or not DEBUT_VALIDE.match(question):
            continue
        spans.append((debut, fin, question))

    fiches = []
    for i, (debut, fin, question) in enumerate(spans):
        fin_reponse = spans[i + 1][0] if i + 1 < len(spans) else len(texte)
        reponse = texte[fin:fin_reponse].strip()
        if len(reponse) < 60:            # question sans réponse exploitable
            continue
        if "satisfaction est notre priorité" in question or "Ce guide" in reponse[:60]:
            continue                     # page de garde / avant-propos
        fiches.append({
            "id": f"FAQ.{i + 1}",
            "categorie": categorie_a(debut, texte),
            "probleme": question,
            "solution": reponse[:1400],
            "status": "ok",
            # La question officielle est indexée comme variante (la fiche dispose
            # ainsi de 2 chunks, condition du vote de consensus), complétée par
            # le début de la réponse : indispensable pour les questions
            # déictiques du guide (« Que veut dire cette mention ? ») qui, seules,
            # ne désignent aucun sujet. Les formules passe-partout sont écartées
            # plus bas : partagées par plusieurs fiches, elles les rendraient
            # indiscernables.
            "variantes": [question, premiere_phrase(reponse)],
        })
    return ecarter_formules_communes(fiches)


def ecarter_formules_communes(fiches: list[dict]) -> list[dict]:
    """Le guide répète des formules identiques d'une réponse à l'autre
    (« Vous devez contacter votre perception de rattachement et fournir les
    pièces suivantes »). Gardée comme variante, une telle phrase rendrait
    plusieurs fiches indiscernables : on ne conserve que ce qui est unique."""
    from collections import Counter
    empreintes = Counter(
        fi["variantes"][1][:70].lower() for fi in fiches
        if len(fi["variantes"]) > 1 and fi["variantes"][1]
    )
    for fi in fiches:
        gardees = [fi["variantes"][0]]
        extra = fi["variantes"][1] if len(fi["variantes"]) > 1 else ""
        if extra and empreintes[extra[:70].lower()] == 1:
            gardees.append(extra)
        fi["variantes"] = gardees
    return fiches


def main():
    fiches = extraire()
    os.makedirs(DATA_PROCESSED_DIR, exist_ok=True)
    with open(FAQ_FICHES_JSON, "w", encoding="utf-8") as f:
        json.dump(fiches, f, ensure_ascii=False, indent=2)
    print(f"{len(fiches)} fiches extraites de la FAQ officielle → {FAQ_FICHES_JSON}\n")
    for fi in fiches:
        print(f"  [{fi['id']}] ({fi['categorie']}) {fi['probleme'][:88]}")
        print(f"        → {fi['solution'][:88]}...")


if __name__ == "__main__":
    main()
