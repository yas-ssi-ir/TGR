"""
Nettoyage des réponses pré-rédigées — enlève ce que le modèle ajoute malgré
les consignes : préambules, salutations, signatures.

Symptômes constatés sur 25 des 44 réponses :
    « Voici la réponse officiée en français : »   ← préambule + faute
    « Cher utilisateur, »                         ← salutation inutile
    « 1. Bonjour, je comprends que… »             ← salutation dans l'étape 1

Un petit modèle n'obéit pas parfaitement à une consigne de format ; plutôt que
de le relancer en espérant mieux, on corrige mécaniquement. C'est déterministe,
gratuit, et reproductible.

L'usager voit la réponse dans une bulle de chat ou dans une lettre qui porte
déjà sa propre formule d'appel : la réponse doit commencer par l'information.

Lancement :  python -X utf8 src\nettoyer_reponses.py          (aperçu)
             python -X utf8 src\nettoyer_reponses.py --appliquer
"""
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (
    ASSISTANT_FICHES_JSON, FAQ_FICHES_JSON, PRECOMPUTED_JSON, QA_FICHES_JSON,
)

# Ligne d'annonce que le modèle place avant sa réponse
PREAMBULE = re.compile(
    r"^\s*(?:voici|voil[àa]|ci-dessous)[^\n:]{0,90}:\s*"
    r"|^\s*r[ée]ponse[^\n:]{0,60}:\s*",
    re.IGNORECASE)

# Salutation seule sur sa ligne
SALUTATION_LIGNE = re.compile(
    r"^\s*(?:bonjour|bonsoir|salut|cher\s+\w+|ch[èe]re\s+\w+|"
    r"madame,?\s*monsieur|monsieur,?\s*madame|السلام عليكم|تحية طيبة)\s*[,!.]?\s*$",
    re.IGNORECASE)

# Salutation collée au début d'un texte ou d'une étape numérotée
SALUTATION_DEBUT = re.compile(
    r"^(\s*(?:\d+[.)]\s*)?)(?:bonjour|bonsoir|salut|السلام(?:\s*عليكم)?|تحية طيبة)\s*[,!:؟،]?\s*",
    re.IGNORECASE)

# Politesse d'ouverture sans contenu : « je comprends que… », « merci de votre message »
OUVERTURE_VIDE = re.compile(
    r"^(\s*(?:\d+[.)]\s*)?)(?:je\s+(?:vous\s+)?(?:comprends|remercie)[^.\n]*[.]\s*|"
    r"merci\s+(?:de|pour)[^.\n]*[.]\s*)",
    re.IGNORECASE)

# Renvoi au corpus : l'usager ne voit aucune « documentation fournie », cette
# formule ne fait que trahir le fonctionnement interne du système
META = re.compile(
    r"\b(?:selon|d'apr[èe]s|conform[ée]ment [àa])\s+"
    r"(?:la|les|le)?\s*(?:documentation|passages?|informations?|donn[ée]es)\s+"
    r"(?:fournie?s?|disponibles?|ci-dessus)?\s*,?\s*",
    re.IGNORECASE)

# Signature : la lettre en ajoute déjà une. On coupe à partir de la formule
# de politesse finale JUSQU'À LA FIN — sinon un « Cordialement, » suivi d'un
# « [Votre Nom] » échappe au filtre, et l'usager lit un crochet à remplir.
SIGNATURE = re.compile(
    r"\n\s*(?:cordialement|bien [àa] vous|sinc[èe]res salutations|"
    r"salutations distingu[ée]es|veuillez agr[ée]er|"
    r"le support eservices tgr|l'?[ée]quipe [^\n]*|"
    r"مع تحيات|مع خالص التحية|تحياتي)[\s\S]*$",
    re.IGNORECASE)

# Crochet resté à remplir : le modèle imite une lettre type et laisse le
# marqueur en place. Servi tel quel, c'est un aveu que personne n'a relu.
PLACEHOLDER = re.compile(
    r"\[\s*(?:votre|vos|nom|pr[ée]nom|ins[ée]rer|[àa] compl[ée]ter|xxx)[^\]\n]{0,40}\]",
    re.IGNORECASE)

# Marqueur d'élément de liste : « 1. », « 2) ». DEUX chiffres au maximum, et
# précédé d'un début de texte ou d'une espace — sans quoi un numéro de
# téléphone (« 0537273727. ») est pris pour une étape numérotée.
MARQUEUR_LISTE = re.compile(r"(?:(?<=^)|(?<=\s))(\d{1,2})[.)]\s")


def nettoyer(texte: str) -> str:
    """Retire préambule, salutations et signature ; conserve tout le contenu utile."""
    t = (texte or "").strip()
    if not t:
        return t

    # 1-3. En-tête : préambule, salutations, politesse creuse. On boucle jusqu'à
    # stabilité car ces éléments s'empilent (« Cher utilisateur, » puis « Voici
    # la réponse… : ») et retirer le premier fait apparaître le suivant.
    avant = None
    while avant != t:
        avant = t
        t = PREAMBULE.sub("", t, count=1).lstrip()
        lignes = t.split("\n")
        if lignes and SALUTATION_LIGNE.match(lignes[0]):
            t = "\n".join(lignes[1:]).lstrip()
        t = SALUTATION_DEBUT.sub(r"\1", t, count=1)
        t = OUVERTURE_VIDE.sub(r"\1", t, count=1)

    # 3bis. renvois au corpus interne (« selon la documentation fournie »)
    t = META.sub("", t)

    # 4. signature finale, puis crochets à remplir laissés par le modèle
    t = SIGNATURE.sub("", t).strip()
    t = PLACEHOLDER.sub("", t)
    t = re.sub(r"[ \t]{2,}", " ", t).strip(" ,;\n")

    # 5. une étape vidée de son texte ne doit pas rester (« 1. » orpheline)
    t = re.sub(r"^\s*\d+[.)]\s*$", "", t, flags=re.MULTILINE).strip()
    t = re.sub(r"^\s*\d+[.)]\s+(?=\d+[.)]\s)", "", t)      # « 1. 2. Texte » → « 2. Texte »

    # 6. renuméroter : après suppression d'une étape, la liste doit repartir à 1
    #
    # ⚠️ Le marqueur est limité à DEUX chiffres. Sans cette borne, « \d+ »
    # avalait « 0537273727. » — le numéro du centre d'appel de la DGI — et le
    # remplaçait par « 1. ». L'usager se serait vu communiquer un numéro de
    # téléphone inexistant, dans un texte officiel recopié tel quel.
    numeros = re.findall(MARQUEUR_LISTE, t)
    if numeros and numeros[0] != "1":
        compteur = [0]

        def renumeroter(m):
            compteur[0] += 1
            return f"{m.group(1)}{compteur[0]}. "

        t = re.sub(r"(^|\s)\d{1,2}[.)]\s", renumeroter, t)

    # majuscule initiale si la coupe a laissé une minuscule
    m = re.match(r"^(\s*(?:\d+[.)]\s*)?)([a-zà-ÿ])", t)
    if m:
        i = m.end() - 1
        t = t[:i] + t[i].upper() + t[i + 1:]
    return t


def retire(avant: str, apres: str) -> str:
    """Décrit CE QUI DISPARAÎT. Afficher la première ligne avant/après ne
    montrait rien quand la coupe portait sur la fin du texte (une signature,
    par exemple) : l'aperçu semblait alors ne rien faire."""
    debut = avant[:len(avant) - len(apres)] if avant.endswith(apres) else ""
    fin = avant[len(debut) + len(apres):] if avant.startswith(debut + apres) else ""
    morceaux = [m.strip().replace("\n", " ⏎ ") for m in (debut, fin) if m.strip()]
    if not morceaux:                      # coupes internes (« selon la doc… »)
        return f"{len(avant) - len(apres)} caractère(s) retiré(s) dans le texte"
    return " … ".join(f"« {m[:70]} »" for m in morceaux)


def textes_officiels() -> set[str]:
    """Réponses recopiées MOT POUR MOT depuis une source officielle.

    Le nettoyage retire des artefacts de RÉDACTION par le modèle (préambules,
    salutations, « [Votre Nom] »). Un texte officiel n'en contient aucun : y
    toucher ne peut que l'abîmer. Mesuré : le nettoyage remplaçait le numéro du
    centre d'appel de la DGI par « 1. » dans une réponse verbatim de la FAQ.

    Le test est exact plutôt que fondé sur le préfixe de l'identifiant : une
    réponse est officielle si elle est IDENTIQUE à la solution de sa fiche.
    """
    officiels = set()
    for chemin in (QA_FICHES_JSON, FAQ_FICHES_JSON, ASSISTANT_FICHES_JSON):
        if os.path.exists(chemin):
            with open(chemin, encoding="utf-8") as f:
                for fiche in json.load(f):
                    if (fiche.get("solution") or "").strip():
                        officiels.add((fiche["id"], fiche["solution"].strip()))
    return officiels


def main():
    appliquer = "--appliquer" in sys.argv
    with open(PRECOMPUTED_JSON, encoding="utf-8") as f:
        data = json.load(f)
    officiels = textes_officiels()

    modifiees = 0
    intouchables = 0
    for fid, rep in data.items():
        if (fid, (rep.get("fr") or "").strip()) in officiels:
            intouchables += 1
            continue
        for langue in ("fr", "ar"):
            avant = rep.get(langue, "")
            apres = nettoyer(avant)
            if apres != avant:
                modifiees += 1
                if not appliquer and modifiees <= 12:
                    print(f"--- {fid} [{langue}] : {retire(avant, apres)}")
                rep[langue] = apres

    if intouchables:
        print(f"\n{intouchables} réponse(s) officielle(s) laissée(s) intacte(s) "
              f"— texte recopié mot pour mot depuis sa source, rien à nettoyer.")
    if appliquer:
        with open(PRECOMPUTED_JSON, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"{modifiees} texte(s) nettoyé(s) et enregistré(s).")
    else:
        print(f"{modifiees} texte(s) seraient nettoyés. "
              f"Relancez avec --appliquer pour écrire.")


if __name__ == "__main__":
    main()
