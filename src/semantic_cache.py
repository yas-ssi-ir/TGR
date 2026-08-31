"""
Cache sémantique persistant.

Quand le LLM a déjà rédigé une réponse (voie lente, 30-60 s sur CPU), on
mémorise (embedding de la question, réponse, sources). Toute question
future quasi identique (similarité cosinus ≥ CACHE_SIM_MIN) est servie
depuis le cache en ~0,3 s, sans aucun appel LLM.

Les embeddings e5 sont normalisés → produit scalaire = similarité cosinus.
Persistance : data/processed/cache_reponses.json
"""
import json
import os

from config import CACHE_JSON, CACHE_SIM_MIN, E5_QUERY_PREFIX


class SemanticCache:
    def __init__(self, embeddings, version: str = ""):
        """version : empreinte du corpus. Si elle a changé depuis l'écriture du
        cache, celui-ci est vidé : une réponse mémorisée avant une correction de
        la documentation ne doit plus jamais être servie."""
        self.embeddings = embeddings          # même modèle e5 que le retriever
        self.version = version
        self.entries: list[dict] = []
        if os.path.exists(CACHE_JSON):
            try:
                with open(CACHE_JSON, encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, dict):
                    if data.get("version") == version:
                        self.entries = data.get("entries", [])
                    else:
                        print("[Cache] documentation modifiée → cache vidé")
                # ancien format (liste nue) : on repart à zéro
            except Exception:
                self.entries = []

    @staticmethod
    def _similarite(a: list[float], b: list[float]) -> float:
        """Produit scalaire = similarité cosinus (les vecteurs e5 sont normalisés).
        Une entrée de dimension différente provient d'un autre modèle
        d'embeddings : on l'écarte au lieu de comparer ce qui n'est pas
        comparable — sans ce garde-fou, zip() tronquerait en silence."""
        if len(a) != len(b):
            return -1.0
        return sum(x * y for x, y in zip(a, b, strict=True))

    def _embed(self, question: str) -> list[float]:
        return self.embeddings.embed_query(E5_QUERY_PREFIX + question.strip())

    def get(self, question: str) -> dict | None:
        """Retourne l'entrée la plus proche si similarité ≥ CACHE_SIM_MIN."""
        if not self.entries or not question.strip():
            return None
        v = self._embed(question)
        best, best_sim = None, -1.0
        for e in self.entries:
            sim = self._similarite(v, e["embedding"])
            if sim > best_sim:
                best, best_sim = e, sim
        if best is not None and best_sim >= CACHE_SIM_MIN:
            return {"reponse": best["reponse"], "sources": best["sources"],
                    "similarite": round(best_sim, 3)}
        return None

    def put(self, question: str, reponse: str, sources: list[dict]):
        """Mémorise une réponse générée par le LLM (évite les doublons)."""
        if not question.strip() or not reponse.strip():
            return
        v = self._embed(question)
        for e in self.entries:
            if self._similarite(v, e["embedding"]) >= CACHE_SIM_MIN:
                return  # une entrée quasi identique existe déjà
        self.entries.append({"question": question.strip(), "embedding": v,
                             "reponse": reponse, "sources": sources})
        os.makedirs(os.path.dirname(CACHE_JSON), exist_ok=True)
        with open(CACHE_JSON, "w", encoding="utf-8") as f:
            json.dump({"version": self.version, "entries": self.entries},
                      f, ensure_ascii=False)


def empreinte_corpus(vectorstore) -> str:
    """Empreinte de l'état de la connaissance : nombre de chunks indexés +
    date de dernière modification des fichiers de fiches et de réponses.
    Toute ré-ingestion ou relecture change l'empreinte → le cache se vide."""
    import hashlib
    from config import FAQ_FICHES_JSON, PRECOMPUTED_JSON, QA_FICHES_JSON
    morceaux = []
    try:
        morceaux.append(str(vectorstore._collection.count()))
    except Exception:
        morceaux.append("?")
    for chemin in (QA_FICHES_JSON, FAQ_FICHES_JSON, PRECOMPUTED_JSON):
        morceaux.append(str(int(os.path.getmtime(chemin))) if os.path.exists(chemin) else "0")
    return hashlib.md5("|".join(morceaux).encode()).hexdigest()[:12]
