"""
Phase 2 — Ingestion multi-sources → ChromaDB.

Sources ingérées :
  1. Fiches Q/R issues des réclamations (qa_fiches.json)  → 1 fiche = 1 chunk
     + chaque variante de question usager = 1 chunk supplémentaire pointant
       vers la même fiche (améliore fortement le rappel)
  2. Documents PDF/DOCX du dossier data/raw/  → chunking 500/50 (séparateurs FR/AR)

Modèle d'embeddings : intfloat/multilingual-e5-base (préfixe "passage: " OBLIGATOIRE).
"""
import json
import os
import re
import shutil
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (
    ASSISTANT_FICHES_JSON, CHROMA_DB_DIR, CHUNK_SOLUTION_MAX, COLLECTION_NAME,
    DATA_RAW_DIR,
    DOC_CHUNK_OVERLAP, DOC_CHUNK_SIZE, DOCS_DEJA_STRUCTURES, E5_PASSAGE_PREFIX,
    EMBEDDING_MODEL, FAQ_FICHES_JSON, QA_FICHES_JSON,
)

from langchain_chroma import Chroma
from langchain_community.document_loaders import Docx2txtLoader, PyPDFLoader
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

BATCH_SIZE = 500


# ── Embeddings ───────────────────────────────────────────────────────
def init_embeddings() -> HuggingFaceEmbeddings:
    print(f"Chargement du modèle d'embeddings : {EMBEDDING_MODEL} (CPU)...")
    start = time.time()
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    print(f"Modèle chargé en {time.time() - start:.1f}s")
    return embeddings


# ── Nettoyage texte (FR + AR) ────────────────────────────────────────
def clean_text(text: str) -> str:
    if not text:
        return ""
    text = text.replace("\x00", "").replace("﻿", "")
    text = re.sub(r"[\x01-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"\n\s+\n", "\n\n", text)
    return text.strip()


# ── Source 1 : fiches Q/R des réclamations ──────────────────────────
SOURCES_FICHES = (
    (QA_FICHES_JSON, "reclamations_xlsx"),
    (FAQ_FICHES_JSON, "faq_pdf"),
    (ASSISTANT_FICHES_JSON, "assistant_docx"),
)


def build_fiche_documents() -> list[Document]:
    """1 fiche = 1 chunk + 1 chunk par variante de question usager.
    Trois sources de fiches : les réclamations (xlsx), la FAQ officielle (PDF)
    et le relevé de l'assistant en production (docx). Toutes passent par le
    même moule → même traitement, même voie rapide."""
    fiches = []
    for chemin, origine in SOURCES_FICHES:
        if os.path.exists(chemin):
            with open(chemin, encoding="utf-8") as f:
                lot = json.load(f)
            # Les nœuds de menu ne portent aucune réponse : les indexer
            # leur ferait voler les questions aux fiches qui, elles, répondent.
            lot = [fi for fi in lot if fi.get("status") != "menu"]
            for fi in lot:
                fi["_origine"] = origine
            fiches += lot
            print(f"  {os.path.basename(chemin)} : {len(lot)} fiches")

    docs = []
    for fiche in fiches:
        meta = {
            "source": fiche["_origine"],
            "fiche_id": fiche["id"],
            "categorie": fiche["categorie"],
            "status": fiche["status"],
        }
        # Chunk principal : problème + solution
        contenu = f"[{fiche['categorie']}] Problème : {fiche['probleme']}"
        if fiche["status"] == "ok":
            contenu += f"\nSolution : {fiche['solution']}"
        else:
            contenu += "\n(Problème connu des équipes techniques — pas de solution en ligne : orienter vers le support.)"
        docs.append(Document(page_content=E5_PASSAGE_PREFIX + clean_text(contenu), metadata=meta))

        # Chunks variantes : la question usager reformulée, sans la solution
        # (voir CHUNK_SOLUTION_MAX). Y recopier la solution transformait les
        # variantes d'une fiche en quasi-clones du même texte, et noyait les
        # variantes courtes non francophones sous le français environnant —
        # au point qu'une question darija ne retrouvait plus sa propre fiche.
        for variante in fiche.get("variantes", []):
            v_contenu = f"[{fiche['categorie']}] Question usager : {variante}\nProblème officiel : {fiche['probleme']}"
            if fiche["status"] == "ok" and CHUNK_SOLUTION_MAX:
                v_contenu += f"\nSolution : {fiche['solution'][:CHUNK_SOLUTION_MAX]}"
            docs.append(Document(page_content=E5_PASSAGE_PREFIX + clean_text(v_contenu),
                                 metadata={**meta, "is_variante": True}))

    print(f"Fiches Q/R : {len(fiches)} fiches → {len(docs)} chunks (avec variantes)")
    return docs


# ── Source 2 : documents PDF/DOCX ────────────────────────────────────
def build_doc_documents() -> list[Document]:
    """Charge et découpe les PDF/DOCX déposés dans data/raw/."""
    separators = ["\n\n", "\n", ".", "。", "؟", "!", "،", ",", " ", ""]
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=DOC_CHUNK_SIZE, chunk_overlap=DOC_CHUNK_OVERLAP,
        separators=separators, length_function=len,
    )

    raw_docs = []
    for filename in sorted(os.listdir(DATA_RAW_DIR)):
        if filename in DOCS_DEJA_STRUCTURES:
            print(f"  (ignoré : {filename} — déjà découpé en fiches)")
            continue
        path = os.path.join(DATA_RAW_DIR, filename)
        ext = os.path.splitext(filename)[1].lower()
        try:
            if ext == ".pdf":
                pages = PyPDFLoader(path).load()
                raw_docs.extend(pages)
                print(f"  PDF  : {filename} → {len(pages)} page(s)")
            elif ext == ".docx":
                sections = Docx2txtLoader(path).load()
                raw_docs.extend(sections)
                print(f"  DOCX : {filename} → {len(sections)} section(s)")
        except Exception as e:
            print(f"  ERREUR sur {filename}: {e}")

    if not raw_docs:
        print("Documents PDF/DOCX : aucun (ok, les fiches Q/R suffisent pour démarrer)")
        return []

    chunks = splitter.split_documents(raw_docs)
    prepared = []
    for chunk in chunks:
        cleaned = clean_text(chunk.page_content)
        if len(cleaned) < 20:
            continue
        prepared.append(Document(
            page_content=E5_PASSAGE_PREFIX + cleaned,
            metadata={
                "source": "doc_officiel",
                "fichier": os.path.basename(chunk.metadata.get("source", "?")),
                "page": chunk.metadata.get("page", -1),
            },
        ))
    print(f"Documents PDF/DOCX : {len(prepared)} chunks")
    return prepared


# ── Stockage ChromaDB ────────────────────────────────────────────────
def reset_database():
    if not os.path.exists(CHROMA_DB_DIR):
        return
    try:
        shutil.rmtree(CHROMA_DB_DIR)
        print(f"Ancienne base supprimée : {CHROMA_DB_DIR}")
    except PermissionError:
        # Sous Windows, un fichier ouvert ne peut pas être supprimé. L'assistant
        # garde la base ChromaDB ouverte tant qu'il tourne : ré-ingérer pendant
        # ce temps échoue. La trace brute de shutil n'aide personne — on dit
        # quoi faire.
        print("\nERREUR : la base vectorielle est verrouillée par un autre processus.")
        print("\n  L'assistant est probablement encore en cours d'exécution : il garde")
        print("  la base ouverte. Arrêtez-le (Ctrl+C dans sa fenêtre), puis relancez")
        print("  cette commande.")
        print(f"\n  Dossier concerné : {CHROMA_DB_DIR}")
        sys.exit(1)


def store_in_chromadb(docs: list[Document], embeddings) -> Chroma:
    """Crée le vectorstore UNE fois puis ajoute par lots (corrige le bug
    du script initial qui rappelait Chroma.from_documents à chaque batch)."""
    print(f"\nEnregistrement de {len(docs)} chunks dans ChromaDB...")
    vectorstore = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=CHROMA_DB_DIR,
    )
    for i in range(0, len(docs), BATCH_SIZE):
        batch = docs[i:i + BATCH_SIZE]
        vectorstore.add_documents(batch)
        print(f"  Batch {i // BATCH_SIZE + 1} : {len(batch)} chunks OK")
    print("Enregistrement terminé.")
    return vectorstore


# ── Vérification ─────────────────────────────────────────────────────
def verify(vectorstore: Chroma):
    tests = [
        "query: j'ai perdu mon téléphone comment me connecter avec le code",
        "query: mon mot de passe ne marche plus",
        "query: comment supprimer mon compte",
        "query: ما هي مشاكل رمز التحقق",
    ]
    print("\n=== Vérification de la base ===")
    for q in tests:
        results = vectorstore.similarity_search_with_score(q, k=1)
        if results:
            doc, score = results[0]
            fid = doc.metadata.get("fiche_id", doc.metadata.get("fichier", "?"))
            print(f"  '{q[7:50]}...' → [{fid}] (dist={score:.3f})")
        else:
            print(f"  '{q}' → AUCUN RÉSULTAT")


def main():
    print("=" * 60)
    print("  RAG TGR — Pipeline d'ingestion → ChromaDB")
    print("=" * 60)
    total_start = time.time()

    embeddings = init_embeddings()
    reset_database()

    docs = build_fiche_documents() + build_doc_documents()
    if not docs:
        print("Aucun document à ingérer !")
        return

    vectorstore = store_in_chromadb(docs, embeddings)
    verify(vectorstore)

    print(f"\nIngestion terminée en {time.time() - total_start:.1f}s "
          f"— {len(docs)} chunks dans '{COLLECTION_NAME}'")


if __name__ == "__main__":
    main()
