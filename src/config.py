"""
Configuration centrale de l'Assistant Agentic RAG TGR.
Toutes les constantes du projet sont regroupées ici.
"""
import os

# ── Chemins ──────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
DATA_PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
CHROMA_DB_DIR = os.path.join(BASE_DIR, "data", "chroma_db")
STATIC_DIR = os.path.join(BASE_DIR, "static")
EVAL_DIR = os.path.join(BASE_DIR, "eval")

XLSX_RECLAMATIONS = os.path.join(DATA_RAW_DIR, "request_response.xlsx")
DOCX_ASSISTANT = os.path.join(DATA_RAW_DIR, "Assistant_IA_eServices_TGR_questions_reponses.docx")
QA_FICHES_JSON = os.path.join(DATA_PROCESSED_DIR, "qa_fiches.json")
FAQ_FICHES_JSON = os.path.join(DATA_PROCESSED_DIR, "faq_fiches.json")
ASSISTANT_FICHES_JSON = os.path.join(DATA_PROCESSED_DIR, "assistant_fiches.json")
FEEDBACK_JSON = os.path.join(DATA_PROCESSED_DIR, "feedbacks.json")

# Documents bruts déjà découpés finement en fiches par un script dédié :
# les ré-ingérer en chunks de 500 caractères ajouterait des centaines de
# passages sans fiche_id, qui ne peuvent pas voter au consensus et ne font
# que du bruit autour des vraies fiches.
DOCS_DEJA_STRUCTURES = {os.path.basename(DOCX_ASSISTANT)}

# Longueur de la solution recopiée dans un chunk de VARIANTE. Zéro = aucune.
#
# Un chunk de variante n'a qu'un seul rôle : représenter UNE façon de poser la
# question. Y recopier la solution le fait échouer de deux manières, mesurées
# l'une après l'autre :
#
#   1. Sur une fiche à procédure longue (AST.28, « Authentification par CNIE »,
#      1 653 caractères), les 7 chunks devenaient des quasi-clones du même texte
#      générique — écart de distance entre variantes : 0,006. Elles ne
#      discriminaient plus rien et votaient toujours ensemble : « donne-moi une
#      recette de tajine » remontait 6 variantes sur 6 sous le seuil.
#
#   2. Sur une variante COURTE et non francophone, le français environnant
#      l'écrase. Mesuré sur « bdelt telephone w mabqitch n9der ndkhol b code » :
#      46 caractères de darija dans un chunk de 268, soit 17 %. Distance de la
#      variante à SON PROPRE chunk :
#
#          la variante seule ......................... 0,136
#          variante + problème, sans solution ........ 0,279   ← retenu
#          + 60 caractères de solution ............... 0,330
#          + 120 caractères (ancien réglage) ......... 0,348   au-delà du seuil
#
#      À 0,348 la question ne retrouvait plus sa propre fiche : elle partait sur
#      la voie lente et mettait 124 secondes à répondre, depuis un morceau de PDF.
#      À 0,279 elle passe sous DIST_SOLO_ACCEPT et revient en voie rapide.
#
# La solution complète reste dans le chunk PRINCIPAL de la fiche, et surtout dans
# la réponse pré-rédigée — c'est elle qui est servie à l'usager, jamais le chunk.
CHUNK_SOLUTION_MAX = 0
PRECOMPUTED_JSON = os.path.join(DATA_PROCESSED_DIR, "reponses_precalculees.json")
CACHE_JSON = os.path.join(DATA_PROCESSED_DIR, "cache_reponses.json")

# ── Embeddings (multilingual-e5 : préfixes obligatoires !) ───────────
EMBEDDING_MODEL = "intfloat/multilingual-e5-base"
E5_PASSAGE_PREFIX = "passage: "
E5_QUERY_PREFIX = "query: "

# ── ChromaDB ─────────────────────────────────────────────────────────
COLLECTION_NAME = "tgr_knowledge_base"

# ── Chunking (documents PDF/DOCX uniquement — les fiches Q/R ne sont
#    pas découpées) ──────────────────────────────────────────────────
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
# Les PDF type FAQ ont des Q/R longues : chunks plus grands pour ne pas
# séparer la question de sa réponse
DOC_CHUNK_SIZE = 900
DOC_CHUNK_OVERLAP = 150

# ── Retrieval ────────────────────────────────────────────────────────
TOP_K = 4                     # nb de passages récupérés
SCORE_THRESHOLD = 0.30        # distance max acceptée (cosine distance, plus petit = plus proche)

# Seuils de décision par distance vectorielle (preuve mathématique avant LLM) :
#   distance <= DIST_AUTO_IN  → passage/question considérés pertinents SANS demander au LLM
#   distance >= DIST_AUTO_OUT → question considérée hors périmètre SANS demander au LLM
#   entre les deux            → zone grise, on demande au LLM (petit modèle 3B)
DIST_AUTO_IN = 0.33
DIST_AUTO_OUT = 0.45

# ── Décision par CONSENSUS (le vrai discriminant) ────────────────────
# Constat mesuré : multilingual-e5 écrase toutes les distances entre 0.22 et
# 0.45, si bien qu'un seuil absolu ne sépare PAS le pertinent du hors-sujet
# (« كيف أحذف حسابي؟ » légitime = 0.344, « recette de tajine » = 0.356).
# Le signal fiable est le VOTE DES VOISINS : pour une vraie question, plusieurs
# chunks du top-k pointent vers la MÊME fiche (la fiche + ses variantes) ;
# pour une question hors sujet, les résultats sont dispersés.
# Profondeur de recherche pour le vote (coût ~50 ms, sans appel au modèle).
#
# Le VRAI filtre est la distance (DIST_CANDIDATE_MAX) : un chunk au-delà ne
# vote pas, où qu'il soit classé. La fenêtre ne doit donc servir qu'à ne rien
# tronquer arbitrairement — une fenêtre étroite revient à écarter des chunks
# pourtant assez proches, au seul motif que d'autres fiches ont pris la place.
#
# Mesuré : à 6, une fiche portant la bonne réponse n'apparaissait qu'une fois
# (son second chunk était au rang 22) et ne pouvait donc pas former de
# consensus ; la question partait vers une fiche « bug » générique.
# Avec 51 fiches portant chacune jusqu'à 5 variantes, il faut voir large.
CONSENSUS_K = 25
CONSENSUS_MIN_CHUNKS = 2  # nb de chunks d'une même fiche pour emporter la décision
DIST_CANDIDATE_MAX = 0.37 # au-delà, un chunk ne vote pas (mesuré : les vrais
                          # consensus sont ≤ 0.35, le bruit commence vers 0.39)
DIST_SOLO_ACCEPT = 0.30   # un chunk unique n'est accepté que s'il est très proche
# Décalage TRANSLANGUE : une question posée dans une écriture absente de la
# documentation (arabe, alors que le corpus TGR est en français) est
# systématiquement plus « loin » — non parce qu'elle est hors sujet, mais parce
# que l'embedding franchit la barrière linguistique. Mesuré : « ما هي مشاكل رمز
# التحقق؟ » place 5 de ses 6 voisins sur la MÊME fiche, mais à 0.40-0.41.
# Les seuils sont donc relevés d'autant pour ces questions.
DIST_OFFSET_TRANSLANGUE = 0.07
# Darija en alphabet latin (« bghit ndir chikaya ») : même barrière linguistique,
# mais franchie de moins haut — les caractères latins et les emprunts au français
# rapprochent déjà la question du corpus. Les 14 questions darija du corpus sont
# toutes à ≤ 0.351, contre 0.40-0.41 pour l'arabe : une marge de 0.07 serait
# du gaspillage de tolérance.
#
# Calibré sur la décision du garde-fou (question légitime acceptée / hors-sujet
# refusé), et non sur la seule distance :
#
#   marge   darija légitimes    refusées à tort    hors-sujet admis
#   0.00        12/14                  1                 0/6
#   0.03        13/14                  0                 2/6      ← retenu
#   0.05        13/14                  0                 2/6
#   0.07        14/14                  0                 3/6
#
# 0.03 fait aussi bien que 0.05 en exposant moins, et supprime le refus à tort —
# le pire des défauts : un usager marocain éconduit par sa propre administration.
DIST_OFFSET_DARIJA = 0.03
# Hors sujet certain : aucun consensus + aucun terme métier + rien de proche
DIST_HORS_SUJET = 0.34
# Départage : une fiche « bug connu » (sans solution) ne doit pas l'emporter
# de justesse sur une fiche qui, elle, porte une solution documentée. Répondre
# « problème connu, contactez le support » quand un correctif existe est la
# pire des deux erreurs pour l'usager.
# Valeur calibrée : 0.04 minimise les erreurs (mesuré sur les 166 questions de
# src/audit_couverture.py). À 0.08, de vrais bugs connus se font voler leur
# question par une fiche voisine — le remède devient pire que le mal.
DIST_DEPARTAGE = 0.04

# ── Vitesse (objectif : ≤ 2 s sur les questions connues) ────────────
# Le LLM sur CPU coûte 40-100 s par réponse. La parade :
#   1. Réponses officielles PRÉ-RÉDIGÉES hors ligne (une fois) → servies
#      instantanément quand la preuve vectorielle est forte (dist ≤ DIST_AUTO_IN)
#   2. Cache sémantique : une question déjà traitée par le LLM n'est
#      jamais re-générée (similarité cosinus ≥ CACHE_SIM_MIN)
#   3. Le LLM ne reçoit que GEN_TOP_K passages (prompt court = prefill court)
CACHE_SIM_MIN = 0.96
GEN_TOP_K = 2

# ── LLM local (Ollama) ───────────────────────────────────────────────
# Adresse lue dans l'environnement : en local Ollama tourne sur la machine,
# en conteneur il s'agit d'un autre service (OLLAMA_URL=http://ollama:11434).
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
LLM_MODEL = os.getenv("LLM_MODEL", "qwen2.5:3b-instruct")   # bon FR/AR, ~2 Go, CPU
LLM_TEMPERATURE = 0.2
LLM_MAX_TOKENS = 512

# ── Agent (bornes strictes pour la latence CPU) ──────────────────────
MAX_REWRITE_RETRIES = 1       # 1 seule reformulation max

# ── Catégories des réclamations (déduites de la numérotation xlsx) ───
CATEGORIES = {
    "1": "Mot de passe & connexion",
    "2": "MFA / OTP / Google Authenticator",
    "3": "Vérification CNIE / NFC (Mon Identité Numérique)",
    "4": "Inscription & adhésion aux services",
    "5": "Gestion du compte",
}

# ── Réponses institutionnelles ───────────────────────────────────────
FALLBACK_ANSWER = (
    "Désolé, je ne trouve pas cette information dans la documentation officielle "
    "du portail eServices TGR. Pour votre sécurité, nous vous invitons à contacter "
    "le support du portail ou votre perception, ou à consulter directement "
    "https://eservices.tgr.gov.ma."
)

KNOWN_BUG_ANSWER = (
    "Ce problème est connu de nos équipes techniques et fait l'objet d'une correction. "
    "En attendant, nous vous invitons à contacter le support du portail eServices TGR "
    "afin qu'un agent puisse débloquer votre situation."
)

OUT_OF_SCOPE_ANSWER = (
    "Je suis l'assistant du portail eServices de la Trésorerie Générale du Royaume (TGR). "
    "Je ne peux répondre qu'aux questions concernant le portail et ses services : "
    "compte et connexion, taxes et paiements, quittances, services aux fonctionnaires, "
    "activité bancaire, commande publique et réclamations. "
    "N'hésitez pas à me poser une question sur ces sujets !"
)
