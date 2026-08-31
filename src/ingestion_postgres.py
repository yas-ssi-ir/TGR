"""
(Préparé pour la suite) Ingestion depuis PostgreSQL → ChromaDB.
À activer quand la DSI fournira l'accès à la base des réclamations.

Prérequis : pip install psycopg2-binary
Configuration : variables d'environnement PG_HOST, PG_PORT, PG_DB, PG_USER, PG_PASSWORD
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import E5_PASSAGE_PREFIX

from langchain_core.documents import Document

# Requête à adapter avec la DSI (colonnes réelles de la table des réclamations)
SQL_QUERY = """
SELECT id, sujet, description, reponse, service, date_creation
FROM reclamations
WHERE statut = 'RESOLUE'
  AND reponse IS NOT NULL
ORDER BY date_creation DESC
"""


def fetch_reclamations() -> list[Document]:
    """Extrait les réclamations résolues et les transforme en Documents LangChain."""
    import psycopg2  # import local : dépendance optionnelle

    conn = psycopg2.connect(
        host=os.environ.get("PG_HOST", "localhost"),
        port=os.environ.get("PG_PORT", "5432"),
        dbname=os.environ.get("PG_DB", "tgr"),
        user=os.environ.get("PG_USER", "readonly"),
        password=os.environ.get("PG_PASSWORD", ""),
    )
    docs = []
    try:
        with conn.cursor() as cur:
            cur.execute(SQL_QUERY)
            for rid, sujet, description, reponse, service, date_creation in cur.fetchall():
                contenu = (f"[{service}] Réclamation : {sujet}\n"
                           f"Description : {description}\nRéponse apportée : {reponse}")
                docs.append(Document(
                    page_content=E5_PASSAGE_PREFIX + contenu,
                    metadata={"source": "postgres_reclamations", "reclamation_id": str(rid),
                              "service": service or "", "date": str(date_creation)},
                ))
    finally:
        conn.close()
    print(f"PostgreSQL : {len(docs)} réclamations extraites")
    return docs


if __name__ == "__main__":
    # Une fois l'accès obtenu :
    #   from ingestion import init_embeddings, store_in_chromadb
    #   docs = fetch_reclamations()
    #   store_in_chromadb(docs, init_embeddings())
    print("Squelette prêt — à brancher quand l'accès PostgreSQL sera fourni par la DSI.")
