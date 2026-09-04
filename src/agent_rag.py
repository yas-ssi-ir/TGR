"""
Phase 4 — AGENTIC RAG (cœur du projet).
Pattern CRAG (Corrective RAG) adapté aux contraintes CPU :

  question
     │
  [1. GUARDRAIL]  hors périmètre TGR ? → refus poli
     │
  [2. RETRIEVE]   top-4 ChromaDB
     │
  [3. GRADE]      le LLM note la pertinence des passages (1 seul appel)
     │
     ├─ ≥1 pertinent → [5. GENERATE] réponse ancrée + citations
     │                      │
     │                 [6. VERIFY] réponse appuyée par les passages ? sinon fallback
     │
     └─ 0 pertinent → [4. REWRITE] reformulation (max 1 retry) → RETRIEVE bis
                          │
                     toujours rien → FALLBACK honnête (orientation support)

Chaque étape est tracée → l'interface montre l'agent "réfléchir".
"""
import json
import os
import re
import sys
import time
import uuid
from datetime import datetime

ARABIC_RE = re.compile("[؀-ۿ]")  # écriture arabe


def langue_de(text: str) -> str:
    """'ar' si le texte contient de l'écriture arabe, 'fr' sinon."""
    return "ar" if ARABIC_RE.search(text or "") else "fr"


def fiche_consensus(passages: list[dict], marge: float = 0.0,
                    question: str = "") -> dict | None:
    """VOTE DES VOISINS — le discriminant fiable (voir config.py).

    Une fiche est RECEVABLE si plusieurs de ses chunks (la fiche et ses variantes
    de question) remontent ensemble — signature d'une vraie correspondance — ou
    si un seul remonte mais très proche. Parmi les recevables, la plus PROCHE
    l'emporte : classer au nombre de voix favoriserait mécaniquement les fiches
    les mieux dotées en variantes.

    `question` active le garde-fou anti-antonymes : une fiche qui décrit
    exclusivement l'action inverse de celle demandée est écartée, même si
    le vecteur la place très près (voir lexique.conflit_action).

    Retourne {"fiche_id", "votes", "best_distance", "passages"} ou None.
    """
    votes: dict[str, list[dict]] = {}
    for p in passages:
        if p["distance"] > DIST_CANDIDATE_MAX + marge or not p["fiche_id"]:
            continue          # trop loin, ou chunk de PDF (pas de fiche à voter)
        votes.setdefault(p["fiche_id"], []).append(p)
    if not votes:
        return None

    # Deux temps, et l'ordre compte :
    #  1. RECEVABILITÉ — une fiche n'est candidate que si elle apporte une vraie
    #     preuve : plusieurs de ses chunks concordent, ou un seul mais très proche.
    #  2. CHOIX — parmi les recevables, la PLUS PROCHE l'emporte.
    # Trancher au nombre de voix biaiserait vers les fiches les mieux dotées en
    # variantes, au détriment d'une fiche plus proche mais moins fournie.
    recevables = {
        fid: chunks for fid, chunks in votes.items()
        if (len(chunks) >= CONSENSUS_MIN_CHUNKS
            # Question translangue (marge > 0) : la distance plus grande est déjà
            # le signe attendu (barrière linguistique), pas un indice de faux
            # positif — DIST_CONSENSUS_MULTI_MAX ne s'applique qu'en français.
            and (marge > 0 or min(c["distance"] for c in chunks) <= DIST_CONSENSUS_MULTI_MAX))
        or min(c["distance"] for c in chunks) <= DIST_SOLO_ACCEPT + marge
    }
    if not recevables:
        return None

    # Garde-fou anti-antonymes : « créer un compte » ne doit jamais tomber sur
    # la fiche « supprimer un compte », si proche soit-elle vectoriellement.
    # Si TOUTES les candidates décrivent l'action inverse, il n'y a pas de
    # correspondance : on rend None plutôt que de servir le contraire.
    if question:
        recevables = {
            fid: chunks for fid, chunks in recevables.items()
            if not conflit_action(question, " ".join(c["text"] for c in chunks))
        }
        if not recevables:
            return None

    fiche_id, chunks = min(recevables.items(),
                           key=lambda kv: min(c["distance"] for c in kv[1]))
    best = min(c["distance"] for c in chunks)

    # Départage : si la gagnante est une fiche « bug connu » (sans solution) et
    # qu'une fiche porteuse de solution la talonne, on préfère la solution.
    if all(c.get("status") == "no_answer" for c in chunks):
        for autre_id, autres in recevables.items():
            if autre_id == fiche_id or any(c.get("status") == "no_answer" for c in autres):
                continue
            best_autre = min(c["distance"] for c in autres)
            if best_autre - best <= DIST_DEPARTAGE:
                fiche_id, chunks, best = autre_id, autres, best_autre
                break

    return {"fiche_id": fiche_id, "votes": len(chunks), "best_distance": best,
            "passages": sorted(chunks, key=lambda c: c["distance"])}


def dedupe_sources(passages: list[dict]) -> list[dict]:
    """Une seule entrée par fiche (les variantes pointent vers la même fiche) —
    on garde la meilleure distance."""
    best = {}
    for p in passages:
        key = p["fiche_id"] or p["fichier"]
        if key not in best or p["distance"] < best[key]["distance"]:
            best[key] = p
    return list(best.values())

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (
    CONSENSUS_K, CONSENSUS_MIN_CHUNKS, DIST_AUTO_IN, DIST_AUTO_OUT,
    DIST_CANDIDATE_MAX, DIST_CONSENSUS_MULTI_MAX, DIST_DEPARTAGE, DIST_HORS_SUJET,
    DIST_SOLO_ACCEPT, FALLBACK_ANSWER, GEN_TOP_K,
    KNOWN_BUG_ANSWER, MAX_REWRITE_RETRIES, OUT_OF_SCOPE_ANSWER,
    PRECOMPUTED_JSON, QUESTIONS_ATTENTE_JSON, TOP_K,
)
# marge_ecriture vit dans lexique (c'est une règle d'écriture, pas de
# recherche) ; réexportée ici, où tous les appelants la cherchent déjà.
from lexique import (
    conflit_action, construire_lexique, contient_terme_noyau, lexique_applicable,
    marge_ecriture, recouvrement,
)
from llm import LLMIndisponible, OllamaLLM
from rag_classic import SYSTEM_PROMPT, build_context
from retriever import TGRRetriever

# ── Prompts des nœuds de décision (réponses ultra-courtes = rapides sur CPU) ──

# Prompt COURT volontairement : sur CPU, chaque mot du prompt système est relu
# à chaque appel. Passer de 450 à ~90 mots divise la latence du guardrail par 3.
GUARDRAIL_SYSTEM = """Filtre du portail eServices TGR (Trésorerie Générale du Royaume, Maroc).

OUI si la question touche : compte, connexion, mot de passe, email, MFA/OTP/Google Authenticator, téléphone perdu, CNIE/NFC, inscription, adhésion, taxes territoriales, télé-paiement, situation fiscale, attestation, quittance, salaire/paie de fonctionnaire, banque-net, réclamation.
NON si c'est un autre sujet (culture générale, cuisine, actualité, autre administration).

Exemples : "mon mot de passe ne marche plus" → OUI. "j'ai changé de téléphone" → OUI. "كيف أحذف حسابي؟" → OUI. "capitale de la France" → NON. "recette de tajine" → NON.

En cas de doute : OUI. Réponds UNIQUEMENT OUI ou NON."""

GRADE_SYSTEM = """Tu es un évaluateur de pertinence documentaire. Pour chaque passage numéroté, indique s'il permet de répondre à la question de l'usager.
Réponds UNIQUEMENT au format : 1:OUI 2:NON 3:OUI 4:NON (un verdict par passage, rien d'autre)."""

REWRITE_SYSTEM = """Tu reformules des questions d'usagers pour une recherche documentaire sur le portail eServices TGR. Réécris la question avec le vocabulaire administratif officiel (mot de passe, MFA, OTP, CNIE, adhésion, PPR...).
Réponds UNIQUEMENT avec la question reformulée, rien d'autre."""

VERIFY_SYSTEM = """Tu es un vérificateur anti-hallucination. Compare la RÉPONSE aux PASSAGES de documentation.

Réponds NON UNIQUEMENT si la réponse INVENTE des faits absents des passages : un lien inventé, un montant inventé, un délai inventé, une procédure inventée.

Réponds OUI si la réponse reformule, résume ou paraphrase les passages — même avec d'autres mots. Les formules de politesse et l'orientation vers le support ne comptent pas comme inventions. En cas de doute, réponds OUI.

Réponds UNIQUEMENT par OUI ou NON."""


class AgenticRAG:
    def __init__(self, retriever: TGRRetriever = None, llm: OllamaLLM = None):
        self.retriever = retriever or TGRRetriever()
        self.llm = llm or OllamaLLM()
        # Réponses officielles pré-rédigées hors ligne (src/precompute_answers.py)
        # → servies instantanément quand la preuve vectorielle est forte
        self.reponses_precalculees: dict = {}
        if os.path.exists(PRECOMPUTED_JSON):
            with open(PRECOMPUTED_JSON, encoding="utf-8") as f:
                self.reponses_precalculees = json.load(f)
            print(f"[Agent] {len(self.reponses_precalculees)} réponses pré-validées chargées ⚡")
        else:
            print("[Agent] Pas de réponses pré-calculées — lancez "
                  "« python -X utf8 src\\precompute_answers.py » pour des réponses < 2 s")
        # Vocabulaire du domaine extrait du corpus → filtre hors-sujet instantané
        self.lexique = construire_lexique(self.retriever.vectorstore)
        print(f"[Agent] Lexique du domaine : {len(self.lexique)} mots")

    # ── Nœuds du graphe ──────────────────────────────────────────────

    def node_guardrail(self, question: str, consensus: dict = None,
                       best_distance: float = None) -> tuple[bool, str]:
        """La question relève-t-elle du portail ? Cascade de signaux
        déterministes (0 ms chacun), du plus sûr au plus faible ; le LLM
        n'est sollicité que pour ce qui échappe à tous.
        Retourne (in_scope, méthode utilisée) — la méthode est affichée à
        l'usager dans la trace de l'agent."""
        # a) Aucun mot commun avec TOUTE la documentation → hors sujet certain.
        #    Ce test passe AVANT le consensus : deux chunks peuvent se ressembler
        #    par hasard (bruit), mais un vocabulaire totalement étranger, jamais.
        if (lexique_applicable(question, self.lexique)
                and recouvrement(question, self.lexique) == 0):
            return False, "aucun mot commun avec la documentation (filtre lexical)"
        # b) Plusieurs chunks d'une même fiche concordent → question du portail.
        if consensus:
            return True, (f"consensus vectoriel (fiche {consensus['fiche_id']}, "
                          f"{consensus['votes']} chunks concordants)")
        # c) Terme métier non ambigu employé → question du portail.
        if contient_terme_noyau(question):
            return True, "terme métier du portail reconnu (filtre lexical)"
        # d) Ni terme métier, ni source proche → hors sujet.
        seuil = DIST_HORS_SUJET + marge_ecriture(question, self.lexique)
        if best_distance is not None and best_distance >= seuil:
            return False, (f"ni terme métier, ni source proche "
                           f"(dist={best_distance:.2f} ≥ {seuil:.2f})")
        # e) Cas réellement ambigu : le LLM arbitre (seul appel possible ici).
        verdict = self.llm.decide(GUARDRAIL_SYSTEM, f"Question : {question}")
        return "OUI" in verdict, "décision LLM (dernier recours)"

    def _journaliser_attente(self, question: str, statut: str, fiche_id: str = None) -> None:
        """Question dans le périmètre TGR mais pas correctement traitée : conservée
        pour relecture humaine sur /revision plutôt que perdue silencieusement.
        Best-effort — une erreur d'écriture ne doit jamais faire échouer la réponse
        déjà décidée à l'usager."""
        try:
            attente = []
            if os.path.exists(QUESTIONS_ATTENTE_JSON):
                with open(QUESTIONS_ATTENTE_JSON, encoding="utf-8") as f:
                    attente = json.load(f)
            attente.append({
                "id": uuid.uuid4().hex[:8],
                "question": question,
                "langue": langue_de(question),
                "statut": statut,
                "fiche_id": fiche_id,
                "date": datetime.now().isoformat(timespec="seconds"),
            })
            with open(QUESTIONS_ATTENTE_JSON, "w", encoding="utf-8") as f:
                json.dump(attente, f, ensure_ascii=False, indent=2)
        except OSError:
            pass

    def node_direct(self, question: str, consensus: dict) -> dict | None:
        """VOIE RAPIDE ⚡ — réponse en < 1 s, zéro appel LLM.
        Si le consensus désigne une fiche dont la réponse officielle a été
        rédigée et validée hors ligne, on la sert telle quelle : ancrée par
        construction, dans la langue de l'usager (FR/AR)."""
        if not consensus:
            return None
        tete = consensus["passages"][0]
        if tete.get("status") == "no_answer":
            self._journaliser_attente(question, "known_bug", tete.get("fiche_id"))
            return {"reponse": KNOWN_BUG_ANSWER, "statut": "KNOWN_BUG", "fiche": tete}
        pre = self.reponses_precalculees.get(consensus["fiche_id"])
        if not pre:
            return None
        lang = langue_de(question)
        return {"reponse": pre.get(lang) or pre["fr"], "statut": "SUCCESS", "fiche": tete}

    def node_grade(self, question: str, passages: list[dict]) -> list[dict]:
        """Tri des passages pertinents. Preuve vectorielle d'abord :
        ≤ DIST_AUTO_IN → gardé sans LLM ; ≥ DIST_AUTO_OUT → écarté sans LLM.
        Le LLM n'est appelé QUE s'il reste des passages en zone grise
        (un seul appel pour tous)."""
        auto = [p for p in passages if p["distance"] <= DIST_AUTO_IN]
        gris = [p for p in passages if DIST_AUTO_IN < p["distance"] < DIST_AUTO_OUT]
        if not gris:
            return auto

        numbered = "\n\n".join(
            f"Passage {i} :\n{p['text'][:400]}" for i, p in enumerate(gris, 1)
        )
        verdict = self.llm.generate(
            GRADE_SYSTEM,
            f"QUESTION : {question}\n\nPASSAGES :\n{numbered}",
            temperature=0.0, max_tokens=32,
        )
        # parse "1:OUI 2:NON ..."
        votes = dict(re.findall(r"(\d)\s*:\s*(OUI|NON)", verdict.upper()))
        keep = auto + [p for i, p in enumerate(gris, 1) if votes.get(str(i)) == "OUI"]
        # si le parsing échoue totalement, on garde le meilleur passage (sécurité)
        if not keep and not votes:
            keep = passages[:1]
        return keep

    def node_rewrite(self, question: str) -> str:
        rewritten = self.llm.generate(
            REWRITE_SYSTEM, f"Question usager : {question}",
            temperature=0.3, max_tokens=64,
        )
        return rewritten.strip().strip('"')

    def node_generate(self, question: str, passages: list[dict]) -> str:
        # GEN_TOP_K passages seulement : un prompt court divise le temps de
        # lecture (prefill) du LLM sur CPU
        user_prompt = (
            f"PASSAGES DE DOCUMENTATION :\n\n{build_context(passages[:GEN_TOP_K])}\n\n"
            f"QUESTION DE L'USAGER : {question}\n\nRéponds selon les règles."
        )
        return self.llm.generate(SYSTEM_PROMPT, user_prompt)

    def node_verify(self, reponse: str, passages: list[dict]) -> bool:
        """Groundedness check : la réponse est-elle ancrée dans les passages ?
        (texte INTÉGRAL des passages — une troncature ferait rejeter à tort)"""
        ctx = "\n".join(p["text"] for p in passages)
        verdict = self.llm.decide(
            VERIFY_SYSTEM, f"PASSAGES :\n{ctx}\n\nRÉPONSE :\n{reponse}"
        )
        return "OUI" in verdict

    # ── Orchestration ────────────────────────────────────────────────

    def answer(self, question: str) -> dict:
        """Exécute le graphe complet et retourne réponse + sources + trace."""
        start = time.time()
        etapes = []

        def result(reponse, sources, statut):
            return {
                "question": question,
                "reponse": reponse,
                "sources": sources,
                "statut": statut,
                "etapes": etapes,
                "pipeline": "agentic",
                "latence_s": round(time.time() - start, 1),
            }

        # 1. Retrieve (d'abord : le vote des voisins sert de preuve au guardrail)
        query = question
        passages = self.retriever.search(query, k=CONSENSUS_K)
        best_dist = passages[0]["distance"] if passages else None
        consensus = fiche_consensus(passages, marge_ecriture(question, self.lexique),
                                    question)
        etapes.append({"noeud": "retrieve", "detail": f"{len(passages)} passages récupérés"})

        # 2. Guardrail périmètre (consensus vectoriel, LLM seulement en zone grise)
        in_scope, methode = self.node_guardrail(question, consensus, best_dist)
        etapes.append({"noeud": "guardrail",
                       "detail": ("dans le périmètre TGR — " if in_scope
                                  else "HORS périmètre — ") + methode})
        if not in_scope:
            return result(OUT_OF_SCOPE_ANSWER, [], "OUT_OF_SCOPE")

        # 2bis. VOIE RAPIDE ⚡ — réponse officielle pré-validée, zéro LLM
        direct = self.node_direct(question, consensus)
        if direct:
            etapes.append({"noeud": "direct",
                           "detail": f"⚡ réponse officielle pré-validée (fiche {consensus['fiche_id']}, "
                                     f"{consensus['votes']} chunks concordants, "
                                     f"dist={consensus['best_distance']:.2f}) — aucun appel LLM"})
            sources = [{"id": x["fiche_id"] or x["fichier"], "categorie": x["categorie"],
                        "distance": x["distance"]}
                       for x in dedupe_sources(consensus["passages"])]
            return result(direct["reponse"], sources, direct["statut"])

        # La voie rapide n'a pas suffi : à partir d'ici le LLM devient
        # nécessaire. On ne le vérifie qu'ici (pas en tête de answer()) —
        # sinon CHAQUE question, y compris celles servies par la voie rapide
        # sans aucun appel LLM, paierait l'aller-retour réseau vers Ollama.
        if not self.llm.is_available():
            raise LLMIndisponible(
                f"Ollama indisponible — lancez Ollama et « ollama pull {self.llm.model} ».")

        # 3. Grade (+ 4. Rewrite si nécessaire, borné) — sur le top-K seulement
        passages = passages[:TOP_K]
        relevant = self.node_grade(question, passages) if passages else []
        etapes.append({"noeud": "grade",
                       "detail": f"{len(relevant)}/{len(passages)} passages jugés pertinents"})

        retries = 0
        while not relevant and retries < MAX_REWRITE_RETRIES:
            retries += 1
            query = self.node_rewrite(question)
            etapes.append({"noeud": "rewrite", "detail": f"question reformulée : « {query} »"})
            passages = self.retriever.search(query, k=TOP_K)
            relevant = self.node_grade(question, passages) if passages else []
            etapes.append({"noeud": "grade",
                           "detail": f"retry {retries} : {len(relevant)}/{len(passages)} pertinents"})

        # Fallback honnête si rien de pertinent
        if not relevant:
            etapes.append({"noeud": "fallback", "detail": "aucune source fiable → réponse honnête"})
            return result(FALLBACK_ANSWER, [], "INSUFFICIENT_KNOWLEDGE")

        # Cas spécial : bug connu sans solution documentée
        if all(p.get("status") == "no_answer" for p in relevant):
            etapes.append({"noeud": "known_bug", "detail": "problème connu sans solution en ligne"})
            sources = [{"id": p["fiche_id"], "categorie": p["categorie"],
                        "distance": p["distance"]} for p in dedupe_sources(relevant)]
            self._journaliser_attente(question, "known_bug", relevant[0].get("fiche_id"))
            return result(KNOWN_BUG_ANSWER, sources, "KNOWN_BUG")

        # 5. Generate
        reponse = self.node_generate(question, relevant)
        etapes.append({"noeud": "generate", "detail": f"réponse rédigée ({len(reponse)} caractères)"})

        # 6. Verify (groundedness) — la preuve vectorielle prime : si les
        # sources sont très proches de la question, l'ancrage est garanti et
        # l'appel LLM de vérification est ÉVITÉ (gain 10-25 s sur CPU)
        tres_proche = relevant[0]["distance"] <= DIST_AUTO_IN
        if tres_proche:
            etapes.append({"noeud": "verify",
                           "detail": "ancrage garanti par la preuve vectorielle "
                                     f"(dist={relevant[0]['distance']:.2f}) — appel LLM évité"})
        elif self.node_verify(reponse, relevant):
            etapes.append({"noeud": "verify", "detail": "réponse ancrée dans les sources ✓"})
        else:
            etapes.append({"noeud": "verify", "detail": "réponse NON ancrée → remplacée par fallback"})
            self._journaliser_attente(question, "verification_echouee")
            return result(FALLBACK_ANSWER, [], "NOT_GROUNDED")

        sources = [{"id": p["fiche_id"] or p["fichier"], "categorie": p["categorie"],
                    "distance": p["distance"]} for p in dedupe_sources(relevant)]
        return result(reponse, sources, "SUCCESS")


if __name__ == "__main__":
    question = sys.argv[1] if len(sys.argv) > 1 else "J'ai changé de téléphone et je n'ai plus mes codes"
    agent = AgenticRAG()
    if not agent.llm.is_available():
        print("⚠ Ollama indisponible. Installez-le puis : ollama pull qwen2.5:3b-instruct")
        sys.exit(1)
    r = agent.answer(question)
    print(f"\nQ : {r['question']}\nStatut : {r['statut']}\n")
    print("Trace de l'agent :")
    for e in r["etapes"]:
        print(f"  → [{e['noeud']}] {e['detail']}")
    print(f"\nR : {r['reponse']}\n\nSources : {r['sources']}\nLatence : {r['latence_s']}s")
