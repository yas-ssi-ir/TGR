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
import json
import os
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import PRECOMPUTED_JSON
from llm import OllamaLLM
from nettoyer_reponses import nettoyer

ARABE = re.compile("[؀-ۿ]")
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


def traduire(llm: OllamaLLM, texte_fr: str) -> str | None:
    """Traduit et vérifie. Retourne None si le modèle n'a pas produit d'arabe."""
    demandes = [
        f"ترجم النص التالي إلى العربية الفصحى:\n\n{texte_fr}",
        f"Traduis INTÉGRALEMENT le texte suivant en arabe standard. "
        f"Ta réponse doit être écrite en caractères arabes.\n\n{texte_fr}",
    ]
    for demande in demandes:
        sortie = llm.generate(TRADUCTION_SYSTEM, demande, temperature=0.1, max_tokens=700)
        sortie = nettoyer(sortie)
        if proportion_arabe(sortie) >= PROPORTION_MIN:
            return sortie
    return None


def main():
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
            data[fid]["ar"] = traduction
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
