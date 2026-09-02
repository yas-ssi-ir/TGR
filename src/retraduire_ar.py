"""
Retraduction des réponses arabes défaillantes.

Constat : sur 44 fiches, 25 champs « ar » contenaient en réalité du FRANÇAIS.
Le modèle 3B, à qui l'on passait le texte à traduire sans consigne dans le
message utilisateur, se contentait de le recopier. Un usager arabophone
recevait donc une réponse en français.

Ce script :
  1. détecte les champs « ar » qui ne sont pas en écriture arabe ;
  2. les retraduit depuis le texte français NETTOYÉ ;
  3. VÉRIFIE que le résultat est bien en arabe — et réessaie une fois sinon ;
  4. ne remplace jamais un texte correct par un texte douteux.

Interruptible : sauvegarde après chaque fiche.

Lancement :  python -X utf8 src\retraduire_ar.py
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
from typing import TYPE_CHECKING

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import PRECOMPUTED_JSON
from nettoyer_reponses import nettoyer

if TYPE_CHECKING:                    # uniquement pour les annotations de type
    from llm import OllamaLLM

# « decouper », « budget_tokens » et « proportion_arabe » sont des fonctions de
# TEXTE PUR : elles n'appellent aucun modèle. Importer le client HTTP au
# chargement du module les rendait pourtant intestables sans lui — l'étage
# rapide de l'intégration continue, qui n'installe volontairement que ruff et
# pytest, échouait sur un « ModuleNotFoundError: requests ». Le client n'est
# donc chargé qu'au moment de s'en servir.

ARABE = re.compile("[؀-ۿ]")
URL = re.compile(r"https?://\S+")
PROPORTION_MIN = 0.30      # en deçà, ce n'est pas de l'arabe

TRADUCTION_SYSTEM = """أنت مترجم محترف. تترجم من الفرنسية إلى العربية الفصحى.
Tu traduis en ARABE STANDARD. Tu ne réponds JAMAIS en français.
Garde en caractères latins uniquement les sigles et noms propres : TGR, CNIE, MFA, OTP, PPR, P1007, Google Authenticator.
Conserve la numérotation des étapes. N'ajoute ni salutation, ni commentaire, ni explication."""


def proportion_arabe(texte: str) -> float:
    lettres = [c for c in (texte or "") if c.isalpha()]
    if not lettres:
        return 0.0
    return sum(1 for c in lettres if ARABE.match(c)) / len(lettres)


TAILLE_MORCEAU = 450       # caractères — reste largement à la portée d'un 3B


def budget_tokens(texte_fr: str) -> int:
    """Une traduction coupée en plein milieu est un échec silencieux : elle
    reste en arabe, passe la vérification, et l'usager reçoit la moitié de la
    réponse. Le budget suit donc la longueur du texte."""
    return max(400, min(1600, 120 + len(texte_fr)))


def decouper(texte_fr: str) -> list[str]:
    """Blocs d'environ TAILLE_MORCEAU caractères, coupés en fin de phrase."""
    phrases = re.split(r"(?<=[.!?:])\s+", texte_fr.strip())
    blocs, courant = [], ""
    for phrase in phrases:
        if courant and len(courant) + len(phrase) > TAILLE_MORCEAU:
            blocs.append(courant.strip())
            courant = phrase
        else:
            courant = f"{courant} {phrase}".strip()
    if courant.strip():
        blocs.append(courant.strip())
    return blocs


def traduire_par_morceaux(llm: OllamaLLM, texte_fr: str) -> str | None:
    """Dernier recours. Sur un texte long et dense — juridique, administratif —
    un modèle 3B décroche et recopie le français au lieu de traduire (constaté
    sur FAQ.35, 1 657 caractères sur les commandes publiques). Découpé en
    paragraphes, chaque morceau redevient à sa portée.

    Chaque bloc est vérifié séparément, et un seul échec annule tout : un texte
    à moitié français serait plus déroutant pour l'usager que du français franc,
    qui est au moins signalé comme défaillant à la relecture.
    """
    blocs = decouper(texte_fr)
    if len(blocs) < 2:
        return None                    # rien à gagner à redécouper
    traduits = []
    for bloc in blocs:
        sortie = nettoyer(llm.generate(
            TRADUCTION_SYSTEM, f"ترجم النص التالي إلى العربية الفصحى:\n\n{bloc}",
            temperature=0.1, max_tokens=budget_tokens(bloc)))
        if proportion_arabe(sortie) < PROPORTION_MIN:
            return None
        traduits.append(sortie)
    return "\n\n".join(traduits)


def traduire(llm: OllamaLLM, texte_fr: str) -> str | None:
    """Traduit et vérifie. Retourne None si le modèle n'a pas produit d'arabe."""
    demandes = [
        f"ترجم النص التالي إلى العربية الفصحى:\n\n{texte_fr}",
        f"Traduis INTÉGRALEMENT le texte suivant en arabe standard. "
        f"Ta réponse doit être écrite en caractères arabes.\n\n{texte_fr}",
    ]
    for demande in demandes:
        sortie = llm.generate(TRADUCTION_SYSTEM, demande, temperature=0.1,
                              max_tokens=budget_tokens(texte_fr))
        sortie = nettoyer(sortie)
        if proportion_arabe(sortie) >= PROPORTION_MIN:
            return sortie
    return traduire_par_morceaux(llm, texte_fr)


def preserver_liens(fr: str, ar: str) -> str:
    """Un lien officiel altéré par la traduction envoie l'usager nulle part.
    Si une URL du texte français a disparu de la traduction arabe, on
    réattache le bloc de liens d'origine, inchangé.

    Indispensable depuis que les réponses de l'assistant en production sont
    servies verbatim : leur contenu utile EST souvent un lien (activation du
    compte, guide CNIE, téléchargement des quittances)."""
    manquants = [u for u in URL.findall(fr) if u not in ar]
    if not manquants:
        return ar
    return ar.rstrip() + "\n\n" + "\n".join(f"Lien : {u}" for u in manquants)


def main():
    from llm import OllamaLLM

    with open(PRECOMPUTED_JSON, encoding="utf-8") as f:
        data = json.load(f)

    llm = OllamaLLM()
    if not llm.is_available():
        print("Ollama indisponible — lancez Ollama d'abord.")
        sys.exit(1)

    a_refaire = [fid for fid, rep in data.items()
                 if proportion_arabe(rep.get("ar", "")) < PROPORTION_MIN]
    print(f"{len(a_refaire)} traduction(s) arabe(s) à refaire "
          f"(sur {len(data)} fiches).\n")

    reussies = echecs = 0
    for i, fid in enumerate(a_refaire, 1):
        debut = time.time()
        traduction = traduire(llm, data[fid]["fr"])
        if traduction:
            data[fid]["ar"] = preserver_liens(data[fid]["fr"], traduction)
            data[fid].pop("ar_defaillante", None)   # la fiche n'est plus en défaut
            reussies += 1
            etat = "OK"
        else:
            data[fid]["ar_defaillante"] = True   # signalé à la relecture
            echecs += 1
            etat = "ECHEC (le modèle ne traduit pas) — français conservé"
        with open(PRECOMPUTED_JSON, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"[{i}/{len(a_refaire)}] {fid:<8} {etat} ({time.time() - debut:.0f}s)")

    print(f"\n{reussies} traduites, {echecs} en échec.")
    if echecs:
        print("Les échecs gardent le texte français : à traduire à la main "
              "dans l'espace agent (/revision).")


if __name__ == "__main__":
    main()
