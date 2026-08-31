"""
Module de traitement des réclamations — calqué sur le vrai portail eServices
(page /my/reclamation : onglets « Dépôt » et « Suivi »).

Fonctionnement (option B — validation humaine) :
  Dépôt d'une réclamation
     │
  [1. CLASSIFIER]  → nature (catégorie) détectée automatiquement
     │
  [2. CHERCHER]    → problème déjà connu dans la base ChromaDB ?
     │
     ├─ solution connue  → REPONSE_PROPOSEE : brouillon généré par le LLM,
     │                     montré à l'usager comme « réponse automatique,
     │                     sous réserve de validation par un agent »
     ├─ bug connu        → ESCALADE_TECHNIQUE : transmis à l'équipe technique
     └─ problème inconnu → ESCALADE_AGENT : transmis à un agent humain
     │
  [3. ENREGISTRER] → référence de suivi REC-AAAA-NNNNN
                     + file d'attente data/processed/reclamations.json

  Suivi : recherche par référence → statut + réponse éventuelle.
"""
import json
import os
import re
import sys
import time
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import DATA_PROCESSED_DIR, DIST_AUTO_OUT, PRECOMPUTED_JSON
from llm import OllamaLLM
from retriever import TGRRetriever
from agent_rag import dedupe_sources, fiche_consensus, langue_de, marge_ecriture
from lexique import construire_lexique

RECLAMATIONS_JSON = os.path.join(DATA_PROCESSED_DIR, "reclamations.json")

# Catégorie de la fiche trouvée → nature de réclamation (évite l'appel LLM
# de classification quand le problème est déjà connu dans la base)
CATEGORIE_TO_NATURE = {
    "Mot de passe & connexion": "Compte & connexion (mot de passe, email)",
    "MFA / OTP / Google Authenticator": "MFA / OTP / Google Authenticator",
    "Vérification CNIE / NFC (Mon Identité Numérique)": "Vérification CNIE / Identité numérique",
    "Inscription & adhésion aux services": "Inscription & adhésion aux services",
    "Gestion du compte": "Compte & connexion (mot de passe, email)",
}

# Natures de réclamation (menu déroulant « Nature réclamation » du portail)
NATURES = [
    "Compte & connexion (mot de passe, email)",
    "MFA / OTP / Google Authenticator",
    "Vérification CNIE / Identité numérique",
    "Inscription & adhésion aux services",
    "Taxes territoriales (déclaration, paiement)",
    "Attestations (imposition, salaire)",
    "Quittances & paiements",
    "Fonctionnaire (salaire, situation administrative)",
    "Autre",
]

CLASSIFY_SYSTEM = """Tu es le classificateur des réclamations du portail eServices TGR.
Choisis LA nature qui correspond le mieux à la réclamation parmi cette liste numérotée :
1. Compte & connexion (mot de passe, email)
2. MFA / OTP / Google Authenticator
3. Vérification CNIE / Identité numérique
4. Inscription & adhésion aux services
5. Taxes territoriales (déclaration, paiement)
6. Attestations (imposition, salaire)
7. Quittances & paiements
8. Fonctionnaire (salaire, situation administrative)
9. Autre
Réponds UNIQUEMENT par le numéro (1 à 9)."""

DRAFT_SYSTEM = """Tu es un agent du support du portail eServices TGR. Rédige une réponse officielle, polie et concise à la réclamation de l'usager, UNIQUEMENT à partir des passages de documentation fournis.
Règles : ne jamais inventer de lien/délai/procédure ; structurer en étapes si nécessaire ; répondre dans la langue de la réclamation ; ne JAMAIS mettre de crochets à remplir comme [Votre nom] ; signer exactement : « Le Support eServices TGR »."""

GRADE_SYSTEM = """Tu évalues si le passage documentaire correspond au problème décrit dans la réclamation.
Réponds UNIQUEMENT par OUI ou NON."""


class ReclamationHandler:
    def __init__(self, retriever: TGRRetriever = None, llm: OllamaLLM = None):
        self.retriever = retriever or TGRRetriever()
        self.llm = llm or OllamaLLM()
        # Vocabulaire du domaine — sert à ajuster les seuils pour une
        # réclamation rédigée en arabe (voir DIST_OFFSET_TRANSLANGUE)
        self.lexique = construire_lexique(self.retriever.vectorstore)
        # Réponses officielles pré-rédigées hors ligne → lettre instantanée
        self.reponses_precalculees: dict = {}
        if os.path.exists(PRECOMPUTED_JSON):
            with open(PRECOMPUTED_JSON, encoding="utf-8") as f:
                self.reponses_precalculees = json.load(f)

    # ── Persistance ──────────────────────────────────────────────────
    def _load_all(self) -> list[dict]:
        if os.path.exists(RECLAMATIONS_JSON):
            with open(RECLAMATIONS_JSON, encoding="utf-8") as f:
                return json.load(f)
        return []

    def _save_all(self, items: list[dict]):
        os.makedirs(DATA_PROCESSED_DIR, exist_ok=True)
        with open(RECLAMATIONS_JSON, "w", encoding="utf-8") as f:
            json.dump(items, f, ensure_ascii=False, indent=2)

    def _new_reference(self, existing: list[dict]) -> str:
        annee = datetime.now().year
        return f"REC-{annee}-{len(existing) + 1:05d}"

    # ── Étapes du pipeline ───────────────────────────────────────────
    def classify_nature(self, texte: str) -> str:
        verdict = self.llm.generate(CLASSIFY_SYSTEM, f"Réclamation : {texte}",
                                    temperature=0.0, max_tokens=4)
        m = re.search(r"[1-9]", verdict)
        idx = int(m.group()) - 1 if m else len(NATURES) - 1
        return NATURES[min(idx, len(NATURES) - 1)]

    def find_known_problem(self, texte: str) -> tuple[list[dict], bool]:
        """Cherche le problème dans la base. Retourne (passages pertinents, bug_connu).
        Décision par CONSENSUS (vote des voisins) : plusieurs chunks d'une même
        fiche = correspondance certaine, sans appel LLM. Sinon, le LLM tranche
        uniquement si le meilleur passage est en zone grise."""
        from config import CONSENSUS_K
        passages = self.retriever.search(texte, k=CONSENSUS_K)
        if not passages:
            return [], False

        consensus = fiche_consensus(passages, marge_ecriture(texte, self.lexique))
        if consensus:
            relevant = consensus["passages"]
        else:
            best = passages[0]
            if best["distance"] >= DIST_AUTO_OUT:
                return [], False          # trop loin → problème inconnu
            verdict = self.llm.decide(    # zone grise → 1 appel court
                GRADE_SYSTEM,
                f"RÉCLAMATION : {texte}\n\nPASSAGE :\n{best['text'][:500]}"
            )
            if "OUI" not in verdict:
                return [], False
            # borné : la fenêtre de recherche est large, mais une lettre de
            # réponse ne se rédige pas à partir de vingt extraits
            relevant = [p for p in passages
                        if abs(p["distance"] - best["distance"]) < 0.08][:4]

        bug_connu = all(p.get("status") == "no_answer" for p in relevant)
        return relevant, bug_connu

    def draft_response(self, texte: str, passages: list[dict]) -> str:
        # VOIE RAPIDE ⚡ : lettre assemblée avec la réponse officielle
        # pré-rédigée de la fiche correspondante — aucun appel LLM (< 1 s)
        lang = langue_de(texte)
        for p in passages:
            pre = self.reponses_precalculees.get(p["fiche_id"]) if p["fiche_id"] else None
            if pre:
                corps = pre.get(lang) or pre["fr"]
                if lang == "ar":
                    return ("تحية طيبة،\n\n" + corps + "\n\n"
                            "إذا استمر المشكل بعد هذه الخطوات، يرجى الرد على هذه الشكاية "
                            "وسيتكفل بها أحد أعواننا.\n\n"
                            "دعم بوابة الخدمات الإلكترونية — الخزينة العامة للمملكة")
                return ("Bonjour,\n\n" + corps + "\n\n"
                        "Si le problème persiste après ces étapes, répondez à cette "
                        "réclamation en précisant votre identifiant : un agent prendra "
                        "le relais.\n\nLe Support eServices TGR")

        # Voie lente : rédaction LLM (problème sans réponse pré-rédigée)
        ctx = "\n\n".join(f"--- Passage (fiche {p['fiche_id'] or p['fichier']}) ---\n{p['text']}"
                          for p in passages)
        return self.llm.generate(
            DRAFT_SYSTEM,
            f"DOCUMENTATION :\n{ctx}\n\nRÉCLAMATION DE L'USAGER : {texte}\n\nRédige la réponse officielle."
        )

    # ── API principale : Dépôt ───────────────────────────────────────
    def depot(self, objet: str, description: str, nature: str = "") -> dict:
        """Traite une nouvelle réclamation. Retourne référence + décision + réponse éventuelle."""
        start = time.time()
        texte = f"{objet}. {description}".strip(". ")
        etapes = []

        # 1. Recherche du problème connu (d'abord : la fiche trouvée donne
        #    aussi la nature, sans appel LLM)
        passages, bug_connu = self.find_known_problem(texte)

        # 2. Nature : choix de l'usager > catégorie de la fiche > LLM en dernier recours
        if not nature or nature == "Autre":
            categorie = passages[0]["categorie"] if passages else ""
            nature = CATEGORIE_TO_NATURE.get(categorie) or self.classify_nature(texte)
        etapes.append(f"Nature détectée : {nature}")

        # 3. Décision
        if passages and not bug_connu:
            reponse = self.draft_response(texte, passages)
            statut = "REPONSE_PROPOSEE"     # option B : sous réserve de validation
            etapes.append(f"Problème connu ({len(passages)} fiche(s)) → réponse automatique proposée")
        elif passages and bug_connu:
            reponse = ""
            statut = "ESCALADE_TECHNIQUE"
            etapes.append("Bug connu des équipes → escalade technique")
        else:
            reponse = ""
            statut = "ESCALADE_AGENT"
            etapes.append("Problème inconnu → transmis à un agent")

        # 4. Enregistrement + référence
        items = self._load_all()
        reference = self._new_reference(items)
        record = {
            "reference": reference,
            "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "nature": nature,
            "objet": objet,
            "description": description,
            "statut": statut,
            "reponse_proposee": reponse,
            "validee_par_agent": False,
            "sources": [{"id": p["fiche_id"] or p["fichier"], "categorie": p["categorie"]}
                        for p in dedupe_sources(passages)] if passages else [],
        }
        items.append(record)
        self._save_all(items)

        return {
            "reference": reference,
            "nature": nature,
            "statut": statut,
            "reponse_proposee": reponse,
            "sources": record["sources"],
            "etapes": etapes,
            "latence_s": round(time.time() - start, 1),
            "message_usager": self._message_usager(statut, reference),
        }

    @staticmethod
    def _message_usager(statut: str, reference: str) -> str:
        if statut == "REPONSE_PROPOSEE":
            return (f"Votre réclamation {reference} a été enregistrée. Une réponse automatique "
                    f"vous est proposée ci-dessous — elle sera confirmée par un agent TGR.")
        if statut == "ESCALADE_TECHNIQUE":
            return (f"Votre réclamation {reference} a été enregistrée. Ce problème est connu de "
                    f"nos équipes techniques et votre dossier leur a été transmis en priorité.")
        return (f"Votre réclamation {reference} a été enregistrée et transmise à un agent TGR. "
                f"Vous pouvez suivre son traitement avec votre référence.")

    # ── API : Suivi ──────────────────────────────────────────────────
    def suivi(self, reference: str) -> dict:
        for r in self._load_all():
            if r["reference"].upper() == reference.strip().upper():
                if r["statut"] == "REPONSE_PROPOSEE" and not r["validee_par_agent"]:
                    msg = "Réponse automatique disponible — en attente de validation par un agent."
                elif r["validee_par_agent"]:
                    msg = "Réclamation traitée et validée par un agent."
                else:
                    msg = "Pas encore de réponse, votre réclamation est en cours de traitement !"
                return {"trouvee": True, **r, "message_statut": msg}
        return {"trouvee": False,
                "message_statut": "Aucune réclamation trouvée avec cette référence."}

    # ── API : File d'attente agent (validation) ──────────────────────
    def file_attente(self) -> list[dict]:
        return [r for r in self._load_all() if not r["validee_par_agent"]]

    def valider(self, reference: str, reponse_finale: str = "") -> bool:
        items = self._load_all()
        for r in items:
            if r["reference"] == reference:
                r["validee_par_agent"] = True
                if reponse_finale:
                    r["reponse_proposee"] = reponse_finale
                r["statut"] = "TRAITEE"
                self._save_all(items)
                return True
        return False


if __name__ == "__main__":
    handler = ReclamationHandler()
    if not handler.llm.is_available():
        print("Ollama indisponible.")
        sys.exit(1)
    # Démo : dépôt d'une réclamation connue
    r = handler.depot(
        objet="Problème de connexion MFA",
        description="J'ai changé de téléphone et je n'ai plus accès à mes codes Google Authenticator.",
    )
    print(json.dumps(r, ensure_ascii=False, indent=2))
    # Démo : suivi
    print(json.dumps(handler.suivi(r["reference"]), ensure_ascii=False, indent=2))
