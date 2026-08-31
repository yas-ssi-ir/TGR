"""
Lexique du domaine — filtre hors-sujet DÉTERMINISTE (0 ms, zéro appel LLM).

Idée : le vocabulaire du portail eServices est fermé et connu. On le construit
AUTOMATIQUEMENT à partir du corpus déjà ingéré dans ChromaDB (aucune liste
écrite à la main → il reste juste quand la base évolue).

Une question dont AUCUN mot significatif n'appartient à ce vocabulaire ne peut
pas concerner le portail : « recette de tajine », « capitale de la France ».
À l'inverse, deux mots ou plus du domaine = question du portail.
Entre les deux (un seul mot commun), on laisse le LLM trancher.
"""
import re
import unicodedata

# Mots-outils : présents partout, ils ne portent aucune information de domaine
STOPWORDS = {
    "avec", "dans", "pour", "sans", "sous", "vous", "nous", "elle", "cette",
    "cela", "leur", "mais", "donc", "quel", "quelle", "quels", "quelles",
    "comment", "pourquoi", "quand", "est-ce", "plus", "moins", "tout", "tous",
    "toute", "toutes", "faire", "fait", "peux", "peut", "puis", "veux", "veut",
    "suis", "sont", "etre", "avoir", "mon", "ma", "mes", "son", "sa", "ses",
    "une", "des", "les", "que", "qui", "quoi", "quest", "jai", "jaimerais",
    "bonjour", "merci", "svp", "sil", "plait", "aide", "aidez", "besoin",
    "probleme", "question", "demande", "please", "hello",
}

MOT_RE = re.compile(r"[^\W\d_]+", re.UNICODE)


def normaliser(mot: str) -> str:
    """minuscules + suppression des accents (é→e) pour comparer sereinement."""
    mot = mot.lower()
    return "".join(c for c in unicodedata.normalize("NFD", mot)
                   if unicodedata.category(c) != "Mn")


def mots_significatifs(texte: str) -> set[str]:
    """Mots de 4 lettres ou plus, normalisés, hors mots-outils."""
    return {m for m in (normaliser(x) for x in MOT_RE.findall(texte or ""))
            if len(m) >= 4 and m not in STOPWORDS}


def construire_lexique(vectorstore) -> set[str]:
    """Vocabulaire du domaine, extrait du corpus ChromaDB lui-même."""
    try:
        docs = vectorstore._collection.get(include=["documents"])["documents"]
    except Exception:
        return set()
    lexique = set()
    for d in docs:
        lexique |= mots_significatifs(d)
    return lexique


def recouvrement(question: str, lexique: set[str]) -> int:
    """Nombre de mots de la question appartenant au vocabulaire du domaine."""
    return len(mots_significatifs(question) & lexique)


# ── Termes NOYAU du portail ──────────────────────────────────────────
# Le lexique auto-construit sert de signal NÉGATIF (aucun mot commun = hors
# sujet certain). Il ne peut pas servir de signal POSITIF : il contient tout
# le vocabulaire des PDF, y compris des mots ambigus (« recette » désigne la
# recette de perception à la TGR… et aussi une recette de cuisine).
# D'où cette liste courte de termes non ambigus, propres au portail.
TERMES_NOYAU = {
    # compte & authentification
    "compte", "comptes", "connexion", "connecter", "deconnecter", "identifiant",
    "motdepasse", "passe", "password", "mail", "email", "adresse", "inscription",
    "inscrire", "adhesion", "adherer", "authentification", "authenticator",
    "otp", "code", "codes", "verification", "verifier", "cnie", "nfc",
    "telephone", "portable", "sms", "bloque", "bloquee", "reinitialiser",
    "reinitialisation", "supprimer", "suppression", "profil", "utilisateur",
    # services TGR
    "portail", "eservices", "tgr", "tresorerie", "perception", "taxe", "taxes",
    "impot", "impots", "fiscale", "fiscal", "declaration", "declarer",
    "paiement", "payer", "quittance", "quittances", "attestation", "attestations",
    "imposition", "salaire", "paie", "virement", "pension", "fonctionnaire",
    "ppr", "situation", "reclamation", "reclamations", "banque",
}


def contient_terme_noyau(question: str) -> bool:
    """Vrai si la question emploie au moins un terme non ambigu du portail."""
    mots = mots_significatifs(question)
    mots |= {m.replace("mot de passe", "motdepasse") for m in mots}
    if "passe" in mots and "mot" in normaliser(question):
        mots.add("motdepasse")
    return bool(mots & TERMES_NOYAU)


def lexique_applicable(question: str, lexique: set[str]) -> bool:
    """Le filtre « zéro mot commun » n'a de sens que si le lexique couvre
    l'écriture de la question. La documentation TGR est en français : une
    question en arabe n'a évidemment aucun mot en commun avec elle, ce qui
    la ferait rejeter à tort. Dans ce cas, seuls le consensus vectoriel et
    la distance décident (l'embedding e5, lui, est multilingue)."""
    if not lexique:
        return False
    arabe = re.compile("[؀-ۿ]")
    if arabe.search(question or ""):
        return sum(1 for m in lexique if arabe.search(m)) >= 30
    return True
