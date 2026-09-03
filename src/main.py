"""
Phase 5 — API Web FastAPI de l'Assistant Agentic RAG TGR.

Endpoints :
  GET  /                  → interface chat (static/index.html)
  POST /api/chat          → réponse complète JSON (agentic par défaut, ?pipeline=classic pour la baseline)
  POST /api/chat/stream   → réponse en streaming SSE : étapes de l'agent en direct + tokens
  POST /api/feedback      → enregistre un feedback 👍/👎
  GET  /api/stats         → statistiques d'usage
  GET  /health            → état (ChromaDB, Ollama)

Lancement :  venv\\Scripts\\python.exe -m uvicorn src.main:app --port 8000
"""
import json
import os
import sys
import time
from collections import Counter

# La console Windows démarre en cp1252 : sans cela, un simple caractère
# accentué ou une icône dans un print fait planter le serveur au démarrage.
for flux in (sys.stdout, sys.stderr):
    if hasattr(flux, "reconfigure"):
        flux.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import FEEDBACK_JSON, GEN_TOP_K, LLM_MODEL, STATIC_DIR
from agent_rag import AgenticRAG
from rag_classic import ClassicRAG, SYSTEM_PROMPT, build_context
from reclamation_handler import NATURES, ReclamationHandler
import revision
from semantic_cache import SemanticCache, empreinte_corpus

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

app = FastAPI(
    title="Assistant IA TGR — Agentic RAG",
    description="Assistant du portail eServices TGR (ChromaDB + multilingual-e5 + Ollama, 100% local)",
    version="1.0.0",
)

# Les polices sont servies depuis le projet, jamais depuis un CDN : l'assistant
# doit fonctionner sans réseau, et une police chargée chez un tiers signalerait
# chaque visite d'usager. Voir eval/installer_polices.py.
if os.path.isdir(STATIC_DIR):
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# Instanciation unique au démarrage (le chargement du modèle E5 prend ~10 s)
print("Démarrage de l'assistant TGR...")
agent = AgenticRAG()
classic = ClassicRAG(retriever=agent.retriever, llm=agent.llm)  # partage les ressources
reclamations = ReclamationHandler(retriever=agent.retriever, llm=agent.llm)
cache = SemanticCache(agent.retriever.embeddings,
                      version=empreinte_corpus(agent.retriever.vectorstore))

# Statistiques en mémoire
STATS = {"total": 0, "statuts": Counter(), "latences": []}


class ChatRequest(BaseModel):
    message: str
    pipeline: str = "agentic"   # "agentic" | "classic"


class FeedbackRequest(BaseModel):
    question: str
    reponse: str
    utile: bool
    commentaire: str = ""


class ReclamationDepot(BaseModel):
    objet: str
    description: str
    nature: str = ""


class RevisionEnregistrement(BaseModel):
    id: str
    fr: str = ""
    ar: str = ""
    valider: bool = False
    # Un relecteur se trompe de bouton, et la fiche passe pour certifiée par la
    # TGR. Sans marche arrière, la seule issue était de retoucher le fichier à
    # la main. « valider: false » ne peut pas servir : il veut déjà dire
    # « j'enregistre sans me prononcer », ce qui ne doit RIEN retirer.
    devalider: bool = False
    # « j'ai comparé cette réponse à sa note, il n'y a rien à reprendre » —
    # sans cet état, le seul bouton qui faisait avancer le compteur était
    # « Valider », et un rédacteur finissait par signer à la place de la TGR.
    relue: bool = False
    # Une validation anonyme ne prouve rien à une administration.
    relecteur: str = ""


class ReclamationValidation(BaseModel):
    reference: str
    reponse_finale: str = ""


# ── Endpoints ────────────────────────────────────────────────────────

@app.get("/")
def serve_index():
    index_path = os.path.join(STATIC_DIR, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    raise HTTPException(404, "index.html introuvable")


@app.get("/revision")
def serve_revision():
    """Page de relecture humaine des réponses pré-validées."""
    page = os.path.join(STATIC_DIR, "revision.html")
    if os.path.exists(page):
        return FileResponse(page)
    raise HTTPException(404, "revision.html introuvable")


@app.get("/api/revision/fiches")
def revision_fiches():
    return revision.liste()


@app.post("/api/revision/enregistrer")
def revision_enregistrer(req: RevisionEnregistrement):
    if not revision.enregistrer(req.id, req.fr, req.ar, req.valider,
                                req.devalider, req.relue, req.relecteur):
        raise HTTPException(404, f"Fiche {req.id} introuvable.")
    # l'agent sert les réponses depuis la mémoire : on la rafraîchit
    agent.reponses_precalculees = revision.charger_reponses()
    reclamations.reponses_precalculees = agent.reponses_precalculees
    cache.version = empreinte_corpus(agent.retriever.vectorstore)
    cache.entries = []          # une correction invalide les réponses mémorisées
    return {"ok": True}


@app.get("/health")
def health():
    return {
        "status": "ok",
        "ollama": agent.llm.is_available(),
        "modele": LLM_MODEL,
        "chromadb": agent.retriever.vectorstore._collection.count(),
    }


@app.post("/api/chat")
def chat(req: ChatRequest):
    if not req.message.strip():
        raise HTTPException(400, "Le message ne peut pas être vide.")
    if not agent.llm.is_available():
        raise HTTPException(503, f"Ollama indisponible — lancez Ollama et « ollama pull {LLM_MODEL} »")

    question = req.message.strip()

    # Cache sémantique (pipeline agentic seulement — la baseline reste brute)
    if req.pipeline != "classic":
        hit = cache.get(question)
        if hit:
            result = {"question": question, "reponse": hit["reponse"],
                      "sources": hit["sources"], "statut": "SUCCESS",
                      "mode": "cache", "pipeline": "agentic", "latence_s": 0.3}
            STATS["total"] += 1
            STATS["statuts"]["SUCCESS"] += 1
            STATS["latences"].append(result["latence_s"])
            return result

    pipeline = classic if req.pipeline == "classic" else agent
    result = pipeline.answer(question)

    # Mémorise les réponses rédigées par le LLM (pas les pré-validées, déjà rapides)
    if (req.pipeline != "classic" and result.get("statut") == "SUCCESS"
            and not any(e.get("noeud") == "direct" for e in result.get("etapes", []))):
        cache.put(question, result["reponse"], result["sources"])

    STATS["total"] += 1
    STATS["statuts"][result.get("statut", "SUCCESS")] += 1
    STATS["latences"].append(result["latence_s"])
    return result


@app.post("/api/chat/stream")
def chat_stream(req: ChatRequest):
    """SSE : émet les étapes de l'agent en direct, puis la réponse token par token.
    Événements : {type: "etape"|"token"|"final"|"error", ...}"""
    if not req.message.strip():
        raise HTTPException(400, "Le message ne peut pas être vide.")
    question = req.message.strip()

    def sse(payload: dict) -> str:
        return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"

    def event_generator():
        start = time.time()
        try:
            # 0. Cache sémantique — question déjà traitée par le LLM ? (~0,3 s)
            hit = cache.get(question)
            if hit:
                yield sse({"type": "etape", "noeud": "cache",
                           "detail": f"⚡ Réponse déjà validée en cache (similarité {hit['similarite']:.2f})"})
                latence = round(time.time() - start, 1)
                STATS["total"] += 1
                STATS["statuts"]["SUCCESS"] += 1
                STATS["latences"].append(latence)
                yield sse({"type": "final", "reponse": hit["reponse"], "sources": hit["sources"],
                           "statut": "SUCCESS", "mode": "cache", "latence_s": latence})
                return

            # 1. Retrieve (d'abord : la distance sert de preuve au guardrail)
            yield sse({"type": "etape", "noeud": "retrieve", "detail": "🔍 Recherche documentaire…"})
            from agent_rag import fiche_consensus, marge_ecriture
            from config import CONSENSUS_K
            passages = agent.retriever.search(question, k=CONSENSUS_K)
            best_dist = passages[0]["distance"] if passages else None
            consensus = fiche_consensus(passages, marge_ecriture(question, agent.lexique),
                                        question)
            yield sse({"type": "etape", "noeud": "retrieve",
                       "detail": f"{len(passages)} passages récupérés"})

            # 2. Guardrail (consensus vectoriel, LLM en zone grise seulement)
            yield sse({"type": "etape", "noeud": "guardrail", "detail": "🛡️ Vérification du périmètre TGR…"})
            in_scope, methode = agent.node_guardrail(question, consensus, best_dist)
            if not in_scope:
                yield sse({"type": "etape", "noeud": "guardrail", "detail": f"Hors périmètre ✗ ({methode})"})
                from config import OUT_OF_SCOPE_ANSWER
                yield sse({"type": "final", "reponse": OUT_OF_SCOPE_ANSWER, "sources": [],
                           "statut": "OUT_OF_SCOPE", "latence_s": round(time.time() - start, 1)})
                return
            yield sse({"type": "etape", "noeud": "guardrail", "detail": f"Dans le périmètre ✓ ({methode})"})

            # 2bis. VOIE RAPIDE ⚡ — réponse officielle pré-validée, zéro LLM
            direct = agent.node_direct(question, consensus)
            if direct:
                from agent_rag import dedupe_sources
                yield sse({"type": "etape", "noeud": "direct",
                           "detail": f"⚡ Réponse officielle pré-validée (fiche {consensus['fiche_id']}, "
                                     f"{consensus['votes']} chunks concordants) — aucune génération nécessaire"})
                sources = [{"id": x["fiche_id"] or x["fichier"], "categorie": x["categorie"],
                            "distance": x["distance"]}
                           for x in dedupe_sources(consensus["passages"])]
                latence = round(time.time() - start, 1)
                STATS["total"] += 1
                STATS["statuts"][direct["statut"]] += 1
                STATS["latences"].append(latence)
                # L'usager a le droit de savoir si un agent TGR a signé ce texte
                # ou si personne ne l'a encore relu. C'est toute la différence
                # entre une réponse officielle et une réponse vraisemblable.
                memo = agent.reponses_precalculees.get(consensus["fiche_id"], {})
                yield sse({"type": "final", "reponse": direct["reponse"], "sources": sources,
                           "statut": direct["statut"], "mode": "direct", "latence_s": latence,
                           "fiche": consensus["fiche_id"],
                           "validee": bool(memo.get("validee")),
                           "validee_par": memo.get("validee_par", "")})
                return

            # Voie lente (question inédite) → le LLM devient nécessaire
            if not agent.llm.is_available():
                yield sse({"type": "error",
                           "detail": f"Ollama indisponible — lancez Ollama puis « ollama pull {LLM_MODEL} »"})
                return

            # 3. Grade
            yield sse({"type": "etape", "noeud": "grade", "detail": "⚖️ Évaluation de la pertinence…"})
            from config import TOP_K
            passages = passages[:TOP_K]
            relevant = agent.node_grade(question, passages) if passages else []
            yield sse({"type": "etape", "noeud": "grade",
                       "detail": f"{len(relevant)}/{len(passages)} passages jugés pertinents"})

            # 4. Rewrite (1 retry max)
            if not relevant:
                yield sse({"type": "etape", "noeud": "rewrite", "detail": "✏️ Reformulation de la question…"})
                query2 = agent.node_rewrite(question)
                yield sse({"type": "etape", "noeud": "rewrite", "detail": f"« {query2} »"})
                passages = agent.retriever.search(query2)
                relevant = agent.node_grade(question, passages) if passages else []
                yield sse({"type": "etape", "noeud": "grade",
                           "detail": f"retry : {len(relevant)}/{len(passages)} pertinents"})

            from config import FALLBACK_ANSWER, KNOWN_BUG_ANSWER
            if not relevant:
                yield sse({"type": "etape", "noeud": "fallback",
                           "detail": "Aucune source fiable → réponse honnête"})
                yield sse({"type": "final", "reponse": FALLBACK_ANSWER, "sources": [],
                           "statut": "INSUFFICIENT_KNOWLEDGE",
                           "latence_s": round(time.time() - start, 1)})
                return

            from agent_rag import dedupe_sources
            sources = [{"id": p["fiche_id"] or p["fichier"], "categorie": p["categorie"],
                        "distance": p["distance"]} for p in dedupe_sources(relevant)]

            if all(p.get("status") == "no_answer" for p in relevant):
                yield sse({"type": "etape", "noeud": "known_bug",
                           "detail": "Problème connu des équipes techniques"})
                yield sse({"type": "final", "reponse": KNOWN_BUG_ANSWER, "sources": sources,
                           "statut": "KNOWN_BUG", "latence_s": round(time.time() - start, 1)})
                return

            # 5. Generate en streaming (prompt court : GEN_TOP_K passages max)
            yield sse({"type": "etape", "noeud": "generate", "detail": "✍️ Rédaction de la réponse…"})
            user_prompt = (
                f"PASSAGES DE DOCUMENTATION :\n\n{build_context(relevant[:GEN_TOP_K])}\n\n"
                f"QUESTION DE L'USAGER : {question}\n\nRéponds selon les règles."
            )
            full_response = []
            for token in agent.llm.generate_stream(SYSTEM_PROMPT, user_prompt):
                full_response.append(token)
                yield sse({"type": "token", "token": token})
            reponse = "".join(full_response)

            # 6. Verify — la preuve vectorielle garantit l'ancrage sans appel LLM
            yield sse({"type": "etape", "noeud": "verify", "detail": "🔎 Vérification de l'ancrage…"})
            from config import DIST_AUTO_IN
            tres_proche = relevant[0]["distance"] <= DIST_AUTO_IN
            if tres_proche:
                yield sse({"type": "etape", "noeud": "verify",
                           "detail": "Ancrage garanti par la preuve vectorielle ✓ (appel LLM évité)"})
            elif agent.node_verify(reponse, relevant):
                yield sse({"type": "etape", "noeud": "verify", "detail": "Réponse ancrée dans les sources ✓"})
            else:
                yield sse({"type": "etape", "noeud": "verify", "detail": "Réponse non ancrée → fallback"})
                yield sse({"type": "final", "reponse": FALLBACK_ANSWER, "sources": [],
                           "statut": "NOT_GROUNDED", "latence_s": round(time.time() - start, 1)})
                return

            # Mémorisation : la prochaine question similaire sera servie en ~0,3 s
            cache.put(question, reponse, sources)

            latence = round(time.time() - start, 1)
            STATS["total"] += 1
            STATS["statuts"]["SUCCESS"] += 1
            STATS["latences"].append(latence)
            yield sse({"type": "final", "reponse": reponse, "sources": sources,
                       "statut": "SUCCESS", "mode": "llm", "latence_s": latence})

        except Exception as e:
            yield sse({"type": "error", "detail": str(e)})

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.post("/api/feedback")
def feedback(req: FeedbackRequest):
    entry = {"question": req.question, "reponse": req.reponse[:500],
             "utile": req.utile, "commentaire": req.commentaire,
             "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")}
    feedbacks = []
    if os.path.exists(FEEDBACK_JSON):
        with open(FEEDBACK_JSON, encoding="utf-8") as f:
            feedbacks = json.load(f)
    feedbacks.append(entry)
    with open(FEEDBACK_JSON, "w", encoding="utf-8") as f:
        json.dump(feedbacks, f, ensure_ascii=False, indent=2)
    return {"ok": True}


# ── Module Réclamations (Dépôt / Suivi, comme le vrai portail) ──────

@app.get("/api/reclamation/natures")
def reclamation_natures():
    return {"natures": NATURES}


@app.post("/api/reclamation")
def reclamation_depot(req: ReclamationDepot):
    if not req.description.strip():
        raise HTTPException(400, "La description ne peut pas être vide.")
    if not agent.llm.is_available():
        raise HTTPException(503, f"Ollama indisponible — lancez Ollama et « ollama pull {LLM_MODEL} »")
    return reclamations.depot(req.objet.strip(), req.description.strip(), req.nature)


@app.get("/api/reclamation/suivi/{reference}")
def reclamation_suivi(reference: str):
    return reclamations.suivi(reference)


@app.get("/api/reclamation/file-attente")
def reclamation_file_attente():
    """File des réclamations en attente de validation par un agent TGR."""
    return {"reclamations": reclamations.file_attente()}


@app.post("/api/reclamation/valider")
def reclamation_valider(req: ReclamationValidation):
    ok = reclamations.valider(req.reference, req.reponse_finale)
    if not ok:
        raise HTTPException(404, "Référence introuvable.")
    return {"ok": True}


@app.get("/api/stats")
def stats():
    lat = STATS["latences"]
    return {
        "questions_totales": STATS["total"],
        "statuts": dict(STATS["statuts"]),
        "latence_moyenne_s": round(sum(lat) / len(lat), 1) if lat else 0,
        "latence_max_s": max(lat) if lat else 0,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
