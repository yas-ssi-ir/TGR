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
import os
import re
import sys
import unicodedata

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import DIST_OFFSET_DARIJA, DIST_OFFSET_TRANSLANGUE

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
    # Mots TGR authentiques mais à double sens courant — déjà écartés du signal
    # POSITIF (TERMES_NOYAU, voir plus bas) pour la même raison ; il fallait
    # aussi les écarter du signal NÉGATIF ci-dessous, sans quoi « une recette
    # de tajine » contient « recette » (recette de perception), le test
    # « aucun mot commun » ne se déclenche pas, et la question retombe sur le
    # vote des voisins — sans aucun garde-fou de sens, lui. Mesuré en
    # production : elle y a gagné le consensus de la fiche « mot de passe
    # oublié », servie tel quelle en 0,1 s par la voie rapide.
    "recette", "recettes", "gestion",
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
    "creer", "creation", "activer", "activation", "desinscrire", "desinscription",
    # services TGR
    "portail", "eservices", "tgr", "tresorerie", "perception", "taxe", "taxes",
    "impot", "impots", "fiscale", "fiscal", "declaration", "declarer",
    "paiement", "payer", "quittance", "quittances", "attestation", "attestations",
    "imposition", "salaire", "paie", "virement", "pension", "fonctionnaire",
    "ppr", "situation", "reclamation", "reclamations", "banque",
    # périmètre ouvert par le relevé de l'assistant en production
    # (taxes territoriales, activité bancaire, commande publique) : sans ces
    # termes, le garde-fou refuserait des questions désormais couvertes.
    "amende", "amendes", "contravention", "contraventions", "radar",
    "cheque", "cheques", "chequier", "releve", "releves", "solde", "agence",
    "succession", "successions", "opposition", "provision",
    "boissons", "portuaires", "sejour", "terrains", "urbains",
    "bulletin", "rappels", "prelevement", "prelevements", "familiale",
    "soumission", "soumissionnaire", "adjudication", "fournisseur",
    "fournisseurs", "facture", "factures", "ordonnateur", "banquenet",
    "cessation", "restitution", "degrevement",
    # NB : « recette » (de perception) et « gestion » (des actes) sont bien du
    # vocabulaire TGR, mais restent ABSENTS d'ici à dessein — trop ambigus
    # pour servir de signal positif : « une recette de tajine » passerait.
}


def contient_terme_noyau(question: str) -> bool:
    """Vrai si la question emploie au moins un terme non ambigu du portail."""
    mots = mots_significatifs(question)
    mots |= {m.replace("mot de passe", "motdepasse") for m in mots}
    if "passe" in mots and "mot" in normaliser(question):
        mots.add("motdepasse")
    return bool(mots & TERMES_NOYAU)


# ── Garde-fou anti-antonymes ─────────────────────────────────────────
# Les embeddings sont aveugles à l'opposition de sens : pour multilingual-e5,
# « créer un compte » et « supprimer un compte » sont deux actions sur un
# compte, donc très proches. Mesuré en production : « comment créer mon
# compte » remontait 3 chunks de la fiche « Demandes de suppression de compte »
# et le consensus la déclarait certaine — on expliquait à un usager voulant
# ouvrir un compte comment le détruire.
#
# Aucun réglage de seuil ne corrige cela : la distance est réellement faible.
# Il faut un signal lexical explicite, indépendant du vecteur.
ACTIONS_OPPOSEES = [
    ({"creer", "creation", "ouvrir", "ouverture", "inscrire", "inscription",
      "sinscrire", "adherer", "adhesion", "nouveau", "nouvelle"},
     {"supprimer", "suppression", "supprime", "desinscrire", "desinscription",
      "fermer", "fermeture", "resilier", "resiliation", "cloturer", "cloture",
      "effacer", "desactiver", "desactivation"}),
    ({"activer", "activation", "active"},
     {"desactiver", "desactivation", "desactive", "bloquer", "blocage"}),
    ({"ajouter", "ajout"},
     {"retirer", "retrait", "enlever", "supprimer"}),
]


def _poles(texte: str) -> list[tuple[bool, bool]]:
    """Pour chaque couple d'actions opposées : (le texte emploie le pôle A,
    le texte emploie le pôle B)."""
    mots = mots_significatifs(texte)
    return [(bool(mots & a), bool(mots & b)) for a, b in ACTIONS_OPPOSEES]


def conflit_action(question: str, texte_fiche: str) -> bool:
    """Vrai si la question demande une action et que la fiche décrit
    EXCLUSIVEMENT l'action inverse.

    L'exclusivité est indispensable : une fiche qui mentionne les deux
    (« ouvrez un nouveau compte et demandez la fermeture de l'ancien »)
    répond légitimement aux deux questions — on ne la rejette pas.
    """
    for (q_a, q_b), (f_a, f_b) in zip(_poles(question), _poles(texte_fiche), strict=True):
        if q_a and not q_b and f_b and not f_a:
            return True
        if q_b and not q_a and f_a and not f_b:
            return True
    return False


# Part de mots arabes en deçà de laquelle le corpus reste « une documentation
# française contenant quelques traductions ». Un comptage absolu ne convient
# pas : ajouter des variantes de questions en arabe (47 mots, soit 2,8 % du
# lexique) suffisait à franchir l'ancien seuil de 30 mots, et le filtre lexical
# se remettait à rejeter les questions arabes — la régression même qu'il devait
# empêcher. Une documentation réellement arabophone dépasse largement 20 %.
PART_MIN_ECRITURE = 0.20

ARABE = re.compile("[؀-ۿ]")

# ── Darija en alphabet latin ─────────────────────────────────────────
# « bghit ndir chikaya », « kifach nsajjel », « site dyal TGR wa9ef ma
# khedamch » : c'est du marocain, écrit en lettres latines. Aussi éloigné
# d'une documentation française que l'arabe — mais invisible pour un test
# d'écriture, qui ne voit que des caractères latins. Résultat mesuré : 5 des
# 6 questions du corpus laissées sans réponse étaient de la darija latine,
# renvoyées sur la voie lente (40 à 100 s) faute de marge.
#
# Deux signaux, volontairement stricts pour ne jamais attraper du français.
DARIJA_MOTS = {
    "bghit", "bghina", "bgha", "bghat", "kifach", "kifash", "kifah",
    "dyal", "dyali", "dyalna", "dyalek", "wach", "chno", "chnou", "chhal",
    "makayn", "mabqitch", "khedamch", "walakin", "hadchi", "bezzaf",
    "daba", "3lach", "3andi", "3and", "ndir", "nsajjel", "ndkhol", "nmse7",
    "chikaya", "mochkil", "wa9ef", "khass", "3awtani", "safi", "jdid", "jdida",
}
# Négation darija « ma…ch » : makaynch, mabqitch, mabghitch. Le français n'a
# pas ce motif (« match » n'a qu'une lettre entre « ma » et « ch »).
DARIJA_NEGATION = re.compile(r"\bma[a-z]{2,}ch\b", re.IGNORECASE)
# Chiffre EMPLOYÉ COMME LETTRE au milieu d'un mot : « n9der », « ma3reftch »,
# « 7it ». Une date ou un montant français ne produit jamais ce motif, les
# chiffres y étant isolés (« 2025 », « article 35 »).
DARIJA_CHIFFRE_LETTRE = re.compile(r"[a-zà-ÿ][2379][a-zà-ÿ]", re.IGNORECASE)


def ecriture_darija(question: str) -> bool:
    """Vrai si la question est du marocain transcrit en alphabet latin."""
    texte = question or ""
    if DARIJA_CHIFFRE_LETTRE.search(texte) or DARIJA_NEGATION.search(texte):
        return True
    return bool({normaliser(m) for m in MOT_RE.findall(texte)} & DARIJA_MOTS)


def lexique_applicable(question: str, lexique: set[str]) -> bool:
    """Le filtre « zéro mot commun » n'a de sens que si le lexique couvre
    l'écriture de la question. La documentation TGR est en français : une
    question en arabe ou en darija n'a évidemment aucun mot en commun avec
    elle, ce qui la ferait rejeter à tort. Dans ce cas, seuls le consensus
    vectoriel et la distance décident (l'embedding e5, lui, est multilingue)."""
    if not lexique:
        return False
    if ARABE.search(question or ""):
        part = sum(1 for m in lexique if ARABE.search(m)) / len(lexique)
        return part >= PART_MIN_ECRITURE
    return not ecriture_darija(question)


def marge_ecriture(question: str, lexique: set[str]) -> float:
    """Marge à accorder aux seuils de distance pour une question écrite dans une
    langue absente de la documentation. Deux barrières d'ampleur différente :
    l'arabe (voir DIST_OFFSET_TRANSLANGUE) et la darija en alphabet latin, plus
    proche du corpus et donc moins coûteuse (voir DIST_OFFSET_DARIJA).

    Sa place est ici, et non dans le moteur de décision : c'est une question
    d'ÉCRITURE, pas de recherche. L'y laisser obligeait à charger toute la pile
    ML (torch, chromadb, client HTTP) pour tester une règle de trois lignes.
    """
    if lexique_applicable(question, lexique):
        return 0.0
    return DIST_OFFSET_DARIJA if ecriture_darija(question) else DIST_OFFSET_TRANSLANGUE
