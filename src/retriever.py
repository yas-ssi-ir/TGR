"""
Retriever — recherche sémantique dans ChromaDB.
Ajoute automatiquement le préfixe "query: " requis par multilingual-e5.
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (
    CHROMA_DB_DIR, COLLECTION_NAME, E5_QUERY_PREFIX, EMBEDDING_MODEL, TOP_K,
)

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings


class TGRRetriever:
    def __init__(self):
        print(f"[Retriever] Chargement embeddings {EMBEDDING_MODEL} (CPU)...")
        start = time.time()
        self.embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
        self.vectorstore = Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=self.embeddings,
            persist_directory=CHROMA_DB_DIR,
        )
        print(f"[Retriever] Prêt en {time.time() - start:.1f}s")

    def search(self, query: str, k: int = TOP_K) -> list[dict]:
        """Retourne les k passages les plus proches, avec score et métadonnées.
        Note ChromaDB : score = distance cosinus → PLUS PETIT = PLUS PERTINENT.
        """
        if not query or not query.strip():
            return []
        results = self.vectorstore.similarity_search_with_score(
            E5_QUERY_PREFIX + query.strip(), k=k
        )
        passages = []
        for doc, distance in results:
            # retirer le préfixe "passage: " pour l'affichage / le prompt
            text = doc.page_content
            if text.startswith("passage: "):
                text = text[len("passage: "):]
            passages.append({
                "text": text,
                "distance": round(float(distance), 4),
                "source": doc.metadata.get("source", "?"),
                "fiche_id": doc.metadata.get("fiche_id", ""),
                "categorie": doc.metadata.get("categorie", ""),
                "status": doc.metadata.get("status", "ok"),
                "fichier": doc.metadata.get("fichier", ""),
            })
        return passages


if __name__ == "__main__":
    retriever = TGRRetriever()
    query = sys.argv[1] if len(sys.argv) > 1 else "j'ai perdu mon téléphone comment avoir le code"
    print(f"\nRequête : {query}\n")
    for i, p in enumerate(retriever.search(query), 1):
        print(f"{i}. [{p['fiche_id'] or p['fichier']}] dist={p['distance']} ({p['categorie']})")
        print(f"   {p['text'][:150]}...\n")
