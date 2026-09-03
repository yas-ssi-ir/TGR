"""
Phase 1c — Extraction du guide de référence de l'assistant TGR (.docx).

Source : data/raw/Assistant_IA_eServices_TGR_questions_reponses.docx
         relevé exhaustif des réponses affichées par l'assistant en production
         sur https://eservices.tgr.gov.ma (portail version 1.1.471).

C'est la source la PLUS fiable du projet : ce ne sont ni des notes de tickets,
ni du texte reconstitué par un modèle, mais les réponses officielles réellement
servies aux usagers, avec leurs liens. Elles sont donc reprises VERBATIM.

Le document est structuré par styles Word, ce qui permet un découpage exact
(contrairement au PDF de la FAQ, découpé à l'heuristique) :

    Heading1      → espace (Utilisateur, Contribuable, Assistance, Fonctionnaire)
    Heading2      → rubrique
    TGRQuestion   → « Question / action : <libellé> »
    TGRAnswer     → une ligne de la réponse (ou « Lien : <url> »)
    TGRNote       → « Observation : <constat du relevé> »

Aucune dépendance nouvelle : un .docx est une archive zip contenant du XML.

Trois natures d'entrées sont distinguées :

    ok        → une vraie réponse, servie verbatim
    no_answer → l'assistant en production ne sait pas répondre (anomalie
                relevée) → escalade, jamais de réponse inventée
    menu      → simple nœud de navigation (« le chatbot demande de préciser… ») :
                conservé pour la traçabilité mais JAMAIS indexé, sinon il
                volerait les questions aux fiches filles qui portent la réponse
"""
import json
import os
import re
import sys
import unicodedata
import zipfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import ASSISTANT_FICHES_JSON, DOCX_ASSISTANT

# ── Styles Word du document source ───────────────────────────────────
S_QUESTION = "TGRQuestion"
S_ANSWER = "TGRAnswer"
S_NOTE = "TGRNote"
S_H1 = "Heading1"
S_H2 = "Heading2"

PREFIXE_QUESTION = re.compile(r"^Question\s*/\s*action\s*:\s*", re.I)
PREFIXE_NOTE = re.compile(r"^Observation\s*:\s*", re.I)
LIGNE_LIEN = re.compile(r"^Lien\s*:\s*(\S+)$", re.I)

# Phrases de politesse ajoutées par le chatbot en fin de réponse : elles
# n'apportent rien à la recherche et polluent l'embedding de la fiche.
CLOTURES = re.compile(
    r"^(n'hésitez pas|je reste (à votre|disponible)|si vous avez la moindre question"
    r"|je suis à votre disposition|pour toute (autre )?question)",
    re.I,
)

# Une entrée dont la « réponse » ne fait que DÉCRIRE le comportement du
# chatbot n'est pas une réponse : c'est un nœud de menu.
MARQUEURS_MENU = (
    "le chatbot demande",
    "le chatbot ouvre",
    "le chatbot affiche",
    "le chatbot propose",
    "ce profil renvoie",
    "le menu fonctionnaire principal reprend",
    "choisissez l'option correspondant à votre besoin",
)

# Une entrée où l'assistant en production échoue : on l'enregistre comme
# problème connu (escalade), on n'invente pas de réponse à sa place.
MARQUEURS_SANS_REPONSE = (
    "aucune réponse spécifique n'est fournie",
    "pouvez-vous dire cela autrement",
    "vouliez-vous dire",
    "le chatbot ne fournit pas de réponse utile",
)


# ── Lecture du .docx ─────────────────────────────────────────────────
def lire_paragraphes(chemin: str) -> list[tuple[str, str]]:
    """Retourne [(style, texte), ...] dans l'ordre du document."""
    with zipfile.ZipFile(chemin) as z:
        xml = z.read("word/document.xml").decode("utf-8")

    paragraphes = []
    for bloc in re.findall(r"<w:p[ >].*?</w:p>", xml, re.S):
        texte = "".join(re.findall(r"<w:t[^>]*>(.*?)</w:t>", bloc, re.S))
        texte = (texte.replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")
                      .replace("&quot;", '"').replace("&apos;", "'"))
        style = re.search(r'w:pStyle w:val="([^"]+)"', bloc)
        paragraphes.append((style.group(1) if style else "", nettoyer(texte)))
    return paragraphes


def nettoyer(texte: str) -> str:
    """Retire les caractères de contrôle et normalise les espaces."""
    texte = "".join(c for c in texte if unicodedata.category(c) != "Cc")
    texte = texte.replace("​", "").replace("﻿", "")
    return re.sub(r"[ \t]+", " ", texte).strip()


# ── Découpage en fiches ──────────────────────────────────────────────
def decouper(paragraphes: list[tuple[str, str]]) -> list[dict]:
    """Une fiche par paragraphe de style TGRQuestion, jusqu'au suivant."""
    fiches, courante = [], None
    espace = rubrique = ""

    for style, texte in paragraphes:
        if not texte:
            continue

        if style == S_H1:
            espace = re.sub(r"^\d+\.\s*", "", texte)
            continue
        if style == S_H2:
            rubrique = texte
            continue

        if style == S_QUESTION:
            if courante:
                fiches.append(courante)
            courante = {
                "libelle": PREFIXE_QUESTION.sub("", texte).strip(),
                "espace": espace,
                "rubrique": rubrique,
                "lignes": [],
                "liens": [],
                "observation": "",
            }
            continue

        if courante is None:
            continue

        if style == S_ANSWER:
            lien = LIGNE_LIEN.match(texte)
            if lien:
                if lien.group(1) not in courante["liens"]:
                    courante["liens"].append(lien.group(1))
            elif not CLOTURES.match(texte):
                courante["lignes"].append(texte)
        elif style == S_NOTE:
            courante["observation"] = PREFIXE_NOTE.sub("", texte).strip()

    if courante:
        fiches.append(courante)
    return fiches


def classer(fiche: dict) -> str:
    """ok | no_answer | menu — voir l'en-tête du module."""
    corps = " ".join(fiche["lignes"]).lower()
    if not corps:
        return "menu"
    if any(m in corps for m in MARQUEURS_SANS_REPONSE):
        return "no_answer"
    # un nœud de menu ne décrit que le comportement du chatbot ET ne porte
    # aucun lien exploitable
    if any(corps.startswith(m) or m in corps for m in MARQUEURS_MENU) and not fiche["liens"]:
        return "menu"
    return "ok"


# Dans le document, « ce lien » est un lien hypertexte. Mis à plat, il ne
# reste qu'un « Veuillez cliquer sur ce lien . » qui ne pointe vers rien,
# juste au-dessus du « Lien : … » qui porte réellement l'adresse : l'usager
# cherche un lien dans cette phrase et n'en trouve aucun.
# On ne retire que la PHRASE, jamais la ligne : dans AST.4 le renvoi précède,
# sur la même ligne, les instructions qui font toute la réponse (« Renseignez
# votre adresse email, cochez la case… »). Et seulement si elle commence la
# ligne ou suit un point : « Activez votre compte en cliquant sur ce lien »
# porte le verbe de la phrase, l'amputer laisserait une phrase sans fin.
RENVOI_ORPHELIN = re.compile(
    r"(?:^|(?<=[.!?]))[ \t]*(?:[Vv]euillez\s+)?[Cc]lique[rz]\s+sur\s+ce\s+lien\s*\.?",
    re.MULTILINE)

# Espace parasite devant la ponctuation, héritée de la mise en page du .docx.
ESPACE_AVANT_PONCTUATION = re.compile(r"\s+([.,])")


def composer_solution(fiche: dict) -> str:
    """Réponse verbatim + ses liens, dans l'ordre où le portail les affiche."""
    corps = "\n".join(fiche["lignes"]).strip()
    corps = ESPACE_AVANT_PONCTUATION.sub(r"\1", corps)
    if fiche["liens"]:
        # Le renvoi ne se retire que si l'adresse est bien restituée en
        # dessous : sans lien à afficher, cette phrase reste la seule
        # indication que l'usager doit en suivre un.
        corps = RENVOI_ORPHELIN.sub("", corps)
        corps = "\n".join(ligne.strip() for ligne in corps.split("\n"))
        corps = re.sub(r"\n{3,}", "\n\n", corps).strip()
        liens = "\n".join(f"Lien : {u}" for u in fiche["liens"])
        corps = f"{corps}\n\n{liens}" if corps else liens
    return corps


# ── Variantes : les formulations réelles des usagers ─────────────────
# Le libellé du document est une étiquette d'action (« Création de compte »),
# pas une question. Sans ces variantes, une question posée avec les mots de
# l'usager ne retrouve pas la fiche — c'est exactement le défaut qui faisait
# répondre « comment supprimer mon compte » à « comment créer mon compte ».
VARIANTES = {
    # ── Compte et accès ───────────────────────────────────────────────
    "Création de compte": [
        "Comment créer un compte sur eServices ?",
        "Je veux créer mon compte eServices TGR",
        "Comment ouvrir un compte sur le portail de la TGR ?",
        "Comment faire pour avoir un compte ?",
        "bghit ndir compte jdid f eservices",
        "kifach nsawb compte f site dyal TGR",
        "كيف أنشئ حسابا في بوابة الخدمات الإلكترونية؟",
        "أريد إنشاء حساب جديد",
    ],
    "Inscription aux eServices": [
        "Comment m'inscrire aux eServices ?",
        "Quelles sont les étapes de l'inscription en ligne ?",
        "Je n'arrive pas à m'inscrire, quelle est la procédure complète ?",
        "Expliquez-moi l'inscription étape par étape",
        "kifach nsajjel f eservices TGR",
        "ما هي خطوات التسجيل في الخدمات الإلكترونية؟",
    ],
    "Activation du compte": [
        "Comment activer mon compte ?",
        "Mon compte n'est pas encore activé",
        "Où se fait l'activation du compte eServices ?",
        "bghit n'activi compte dyali",
        "كيف أفعل حسابي؟",
    ],
    "Connexion au compte": [
        "Comment me connecter à mon compte ?",
        "Où se trouve le bouton de connexion ?",
        "Comment accéder à mon espace personnel ?",
        "kifach ndkhol l compte dyali",
        "كيف أدخل إلى حسابي؟",
    ],
    "Code d'activation non reçu": [
        "Je n'ai pas reçu le code d'activation",
        "Code activation non reçu",
        "Le mail avec le code d'activation n'arrive pas",
        "Comment renvoyer le code d'activation ?",
        "ma weslnich code d'activation",
        "لم أتوصل برمز التفعيل",
    ],
    "Login ou mot de passe oublié": [
        "J'ai oublié mon mot de passe",
        "J'ai oublié mon nom d'utilisateur",
        "Comment réinitialiser mon mot de passe ?",
        "Mot de passe oublié, que faire ?",
        "nsit password dyali",
        "نسيت كلمة المرور",
    ],
    "Changement de mot de passe": [
        "Comment changer mon mot de passe ?",
        "Je veux modifier mon mot de passe",
        "Où changer le mot de passe dans mon profil ?",
        "bghit nbdel password dyali",
        "كيف أغير كلمة المرور؟",
    ],
    "Changement d'adresse e-mail": [
        "Comment changer mon adresse email ?",
        "Je veux modifier l'email de mon compte",
        "Changement d'adresse mail eServices",
        "bghit nbdel email dyali",
        "كيف أغير عنوان بريدي الإلكتروني؟",
    ],
    "Activation de la double authentification (2FA)": [
        "Comment activer la double authentification ?",
        "Comment activer le 2FA / MFA sur mon compte ?",
        "Où trouver le guide de l'authentification à deux facteurs ?",
        "كيف أفعل المصادقة الثنائية؟",
    ],
    "Authentification par CNIE": [
        "Comment vérifier mon compte avec ma CNIE ?",
        "Comment m'authentifier avec ma carte nationale ?",
        "Quelles sont les étapes de la vérification CNIE ?",
        "Comment utiliser l'application E-ID pour m'authentifier ?",
        "kifach n'verifi compte dyali b CNIE",
        "كيف أوثق حسابي بالبطاقة الوطنية؟",
    ],
    "Validation du compte par CNIE": [
        "Le compte n'est pas vérifié par la CNIE",
        "Message « compte non vérifié par la CNIE » sur les services fonctionnaire",
        "Comment corriger l'erreur de vérification CNIE ?",
        "حسابي غير موثق بالبطاقة الوطنية",
    ],
    "Absence d'informations relatives aux fonctionnaires": [
        "Mes informations de fonctionnaire ne s'affichent pas",
        "Je ne vois pas mes données de fonctionnaire",
        "L'espace fonctionnaire est vide",
    ],
    "Accès au compte à désinscrire": [
        "Comment supprimer mon compte eServices ?",
        "Je veux me désinscrire du portail",
        "Comment désinscrire mon compte ?",
        "Comment fermer définitivement mon compte ?",
        "bghit nmse7 compte dyali",
        "كيف أحذف حسابي؟",
    ],
    # Variantes volontairement centrées sur la DÉSINSCRIPTION : formulées
    # autour de « je n'ai plus accès », elles captaient « je n'arrive pas à me
    # connecter » et proposaient de supprimer le compte à un usager qui voulait
    # simplement y entrer.
    "Aucun accès au compte à désinscrire": [
        "Désinscription d'un ancien compte eServices devenu inaccessible",
        "Comment supprimer définitivement un compte que je ne peux plus ouvrir ?",
    ],
    "Adresse e-mail d'inscription toujours accessible": [
        "Je veux réinitialiser mon mot de passe et j'ai toujours mon email",
        "Récupérer mon compte avec mon adresse email d'inscription",
    ],
    "Adresse e-mail d'inscription perdue ou inaccessible": [
        "J'ai perdu l'accès à l'email utilisé lors de l'inscription",
        "Mon adresse email d'inscription n'existe plus, comment récupérer mon compte ?",
    ],
    "Paramétrage du compte comme fonctionnaire et/ou contribuable": [
        "Comment m'inscrire au service des fonctionnaires ?",
        "Où saisir mon PPR et mon dernier salaire ?",
        "Comment paramétrer mon profil comme fonctionnaire ?",
        "Comment activer les services d'inscription dans mon profil ?",
    ],
    "Blocage du portail": [
        "Le portail est bloqué",
        "Le site eServices ne répond plus",
        "La page reste bloquée, que faire ?",
        "site dyal TGR wa9ef ma khedamch",
    ],

    # ── Réclamations ──────────────────────────────────────────────────
    "Dépôt d'une réclamation": [
        "Comment déposer une réclamation ?",
        "Où déposer une réclamation en ligne ?",
        "Je veux faire une réclamation",
        "bghit ndir chikaya",
        "كيف أضع شكاية؟",
    ],
    "Suivi d'une réclamation": [
        "Comment suivre ma réclamation ?",
        "Où voir l'avancement de ma réclamation ?",
        "Suivi réclamation en ligne",
        "كيف أتتبع شكايتي؟",
    ],
    "Suivre ou répondre à une demande de complément dans une réclamation": [
        "On me demande un complément d'information sur ma réclamation",
        "Comment répondre à une demande de complément ?",
        "Où se trouve le bouton recours de ma réclamation ?",
    ],

    # ── Taxes territoriales ───────────────────────────────────────────
    # « Guide de la TNB » a été retiré : un sigle nu, sans aucun terme métier
    # reconnaissable, ne ressemble à aucune question d'usager. Mesuré : le
    # garde-fou ne trouvait ni terme du portail ni source proche, réveillait le
    # modèle pour arbitrer — 32 secondes — et finissait par REFUSER une question
    # du portail. Une variante doit être une phrase que quelqu'un taperait.
    "Taxe sur les terrains urbains non bâtis": [
        "Comment déclarer la taxe sur les terrains urbains non bâtis ?",
        "Comment payer la taxe TNB sur mon terrain ?",
        "Taxe terrain non bâti, comment faire ?",
    ],
    "Taxe sur les débits de boissons": [
        "Comment déclarer la taxe sur les débits de boissons ?",
        "Guide taxe débits de boissons",
    ],
    "Taxe de séjour": [
        "Comment déclarer la taxe de séjour ?",
        "Guide de la taxe de séjour",
    ],
    "Taxe sur les eaux minérales et de table": [
        "Comment déclarer la taxe sur les eaux minérales ?",
        "Guide taxe eaux minérales et de table",
    ],
    "Taxe sur les véhicules soumis au contrôle technique": [
        "Comment déclarer la taxe sur les véhicules soumis au contrôle technique ?",
        "Guide taxe véhicules contrôle technique",
    ],
    "Taxe sur les services portuaires": [
        "Comment déclarer la taxe sur les services portuaires ?",
        "Guide taxe services portuaires",
    ],

    # ── Quittances et paiements ───────────────────────────────────────
    "Télécharger les quittances": [
        "Comment télécharger ma quittance ?",
        "Où trouver mes quittances de paiement ?",
        "Je veux récupérer une quittance",
        "bghit nteli9 quittance dyali",
        "كيف أحمل الوصل؟",
    ],
    "Bien déjà déclaré pour 2025 et les années antérieures": [
        "Comment payer ma taxe d'habitation pour 2025 ?",
        "Comment consulter ma situation fiscale TH/TSC ?",
        "Mon bien est déjà déclaré, où payer la TH et la TSC ?",
        "Je n'ai pas la référence de mon avis d'imposition",
    ],
    "Bien non déclaré ou paiement pour 2026 et les années postérieures": [
        "Comment payer ma taxe d'habitation pour 2026 ?",
        "Mon bien n'est pas déclaré, que dois-je faire ?",
        "Où déclarer un bien auprès de la DGI ?",
    ],
    "Consultation fiscale TH/TSC pour 2026 et les années suivantes": [
        "Comment consulter ma situation TH/TSC pour 2026 ?",
        "Paiement TH TSC 2026",
    ],
    "Paiement d'une contravention par radar fixe avec référence": [
        "Comment payer une amende de radar en ligne ?",
        "Comment régler une contravention routière ?",
        "Paiement amende radar fixe avec référence",
        "bghit nkhalles amende dyal radar",
        "كيف أؤدي غرامة الرادار؟",
    ],
    "Paiement d'une contravention par radar fixe sans référence sur le PV": [
        "Mon PV de radar n'a pas de référence, comment payer ?",
        "Amende radar sans référence, véhicule de société",
        "Comment déclarer le conducteur d'un véhicule de société ?",
    ],

    # ── Fonctionnaire ─────────────────────────────────────────────────
    "Situation administrative": [
        "Où consulter ma situation administrative ?",
        "Comment voir ma situation administrative de fonctionnaire ?",
        "أين أجد وضعيتي الإدارية؟",
    ],
    "Situation des ordres de recette": [
        "Où consulter ma situation des ordres de recette ?",
        "Comment voir mes ordres de recette ?",
    ],
    "Situation des actes de gestion": [
        "Où consulter ma situation des actes de gestion ?",
        "Comment voir mes actes de gestion ?",
    ],
    "Situation des rappels": [
        "Où consulter ma situation des rappels ?",
        "Comment voir mes rappels de salaire ?",
    ],
    "Situation des prélèvements": [
        "Où consulter ma situation des prélèvements ?",
        "Comment suivre mes prélèvements sur salaire ?",
    ],
    "Situation de paie": [
        "Où consulter ma situation de paie ?",
        "Comment voir ma paie sur le portail ?",
    ],
    "Attestation de salaire": [
        "Comment obtenir une attestation de salaire ?",
        "Où télécharger mon attestation de salaire ?",
        "bghit attestation de salaire",
        "كيف أحصل على شهادة الأجر؟",
    ],
    "Attestation de cessation de paiement": [
        "Comment obtenir une attestation de cessation de paiement ?",
        "Où télécharger l'attestation de cessation de paiement ?",
    ],
    "Situation familiale": [
        "Où consulter ma situation familiale ?",
        "Comment voir ma situation familiale sur le portail ?",
    ],
    "Bulletin de paie": [
        "Comment télécharger mon bulletin de paie ?",
        "Où trouver ma fiche de paie ?",
        "bghit bulletin de paie dyali",
        "كيف أحمل ورقة الأجر؟",
    ],

    # ── Activité bancaire ─────────────────────────────────────────────
    "Ouverture d'un compte bancaire": [
        "Comment ouvrir un compte bancaire à la TGR ?",
        "Quels documents pour ouvrir un compte bancaire TGR ?",
        "Ouverture de compte à l'agence bancaire de la TGR",
    ],
    "Demande de chéquier ou de carte bancaire": [
        "Comment demander un chéquier ?",
        "Comment obtenir une carte bancaire TGR ?",
        "Commander un chéquier en ligne",
    ],
    "Consultation du solde": [
        "Comment consulter le solde de mon compte bancaire ?",
        "Où voir mon solde TGR ?",
    ],
    "Code PIN oublié": [
        "J'ai oublié le code PIN de ma carte",
        "Comment récupérer mon code PIN ?",
        "nsit code PIN dyal carte",
    ],
    "Changement d'agence en conservant le même compte": [
        "Puis-je garder mon numéro de compte en changeant d'agence ?",
        "Comment transférer mon compte vers une autre agence ?",
    ],
    "Non-réception des relevés": [
        "Je ne reçois pas mes relevés bancaires",
        "Comment obtenir mes relevés de compte ?",
    ],
    "Erreur sur un relevé bancaire": [
        "Il y a une erreur sur mon relevé bancaire",
        "Comment contester une opération sur mon relevé ?",
    ],
    "Régularisation d'un chèque sans provision": [
        "Comment régulariser un chèque sans provision ?",
        "J'ai émis un chèque impayé, que faire ?",
    ],
    "Opposition sur un chèque ou une carte": [
        "Comment faire opposition sur un chèque ?",
        "Ma carte bancaire a été volée, comment faire opposition ?",
    ],
    "Liquidation d'une succession": [
        "Comment liquider une succession ?",
        "Quels documents pour une succession à la TGR ?",
        "Procédure de succession compte bancaire",
    ],

    # ── Commande publique ─────────────────────────────────────────────
    "Retard de paiement - établissement ou entreprise publique": [
        "Ma facture n'est pas payée par un établissement public",
        "Retard de paiement d'un marché public",
        "Comment suivre le paiement de mes factures fournisseur ?",
    ],
    "Retard de paiement - État ou collectivité territoriale": [
        "Retard de paiement par une collectivité territoriale",
        "L'État ne paie pas ma facture",
    ],
    "Lenteur d'accès au portail des marchés publics": [
        "Le portail des marchés publics est lent",
        "Problème de lenteur sur marchespublics.gov.ma",
    ],
    "Dépôt d'une nouvelle offre après un premier dépôt": [
        "Puis-je retirer mon pli et déposer une nouvelle offre ?",
        "Comment modifier mon offre après dépôt ?",
    ],
    "Présentation de sa société aux administrations": [
        "Comment présenter ma société aux administrations ?",
        "Qu'est-ce qu'un appel à manifestation d'intérêt ?",
        "Comment faire une offre spontanée ?",
    ],
    "Motifs de rejet après soumission": [
        "Pourquoi mon offre a-t-elle été rejetée ?",
        "Comment connaître les motifs de rejet de ma soumission ?",
    ],
}

# Rubrique du document → catégorie du projet (aligne les fiches de
# l'assistant sur le vocabulaire déjà utilisé par les fiches réclamations)
CATEGORIES = {
    "Compte et accès": "Compte & accès eServices",
    "Guides des taxes territoriales": "Taxes territoriales",
    "Quittances": "Quittances & paiements",
    "Paiement multicanal": "Quittances & paiements",
    "Réclamations": "Réclamations",
    "Réclamations fréquentes - Utilisateur eServices": "Compte & accès eServices",
    "Réclamations fréquentes - Fonctionnaire": "Fonctionnaire",
    "Réclamations fréquentes - Client de l'activité bancaire": "Activité bancaire",
    "Réclamations fréquentes - Titulaire d'une commande publique": "Commande publique",
    "Réclamations fréquentes - Redevable / contribuable": "Taxes territoriales",
    "Actions proposées": "Fonctionnaire",
}


def reponses_dupliquees(fiches: list[dict]) -> list[str]:
    """Réponses portées par plus d'une fiche. Deux fiches au contenu identique
    se partagent les voix du consensus au lieu de les additionner : la
    déduplication doit ramener cette liste à vide."""
    from collections import Counter
    compte = Counter(f["solution"] for f in fiches
                     if f["status"] == "ok" and len(f["solution"]) >= 40)
    return sorted(s[:70] for s, n in compte.items() if n > 1)


# ── Réponses obtenues APRÈS le relevé du document ────────────────────
# Le .docx date du 2026-09-01 et enregistre, pour certaines entrées, un échec
# de l'assistant en production (« Pouvez-vous dire cela autrement ? »). La TGR
# a corrigé une partie de ces cas depuis. Les textes ci-dessous ont été
# recueillis directement auprès de leur assistant le 2026-09-03 : même source
# et même statut que le reste du corpus AST.*, simplement plus récents.
#
# Ne rien mettre ici qui ne vienne pas d'une source TGR vérifiable, et noter
# la date : c'est ce qui distingue un complément d'une invention.
COMPLEMENTS = {
    "Consultation du solde": (
        "Pour consulter votre solde et suivre vos opérations, vous devez vous "
        "présenter à votre agence afin de demander l'adhésion au service TGR "
        "BanqueNet, qui vous permet un accès sécurisé à vos comptes 24h/24 et "
        "7j/7, ainsi que l'utilisation gratuite de plusieurs services à distance."
    ),
    # Deux retouches sur le texte reçu : la coquille « date d'chèvement » est
    # rétablie en « achèvement », et le « Oui, » d'ouverture est retiré — il
    # répondait à une question fermée que l'usager ne pose pas ici.
    "Motifs de rejet après soumission": (
        "En tant que soumissionnaire non retenu, vous avez le droit d'obtenir "
        "des précisions sur les motifs de rejet de votre offre. Cette obligation "
        "est prévue à l'article 47 du décret n° 2-22-431.\n"
        "\n"
        "1. Attendre la communication officielle : le maître d'ouvrage est tenu "
        "de vous informer des motifs de rejet par lettre recommandée avec accusé "
        "de réception, ou par tout autre moyen donnant date certaine. Cette "
        "notification doit intervenir dans un délai n'excédant pas le troisième "
        "jour suivant la date d'achèvement des travaux de la commission d'appel "
        "d'offres.\n"
        "2. Contester si nécessaire : si les motifs ne vous paraissent pas "
        "fondés, ou si vous estimez qu'il y a eu une erreur, vous pouvez former "
        "un recours gracieux dans les conditions de l'article 163 du même décret.\n"
        "3. Consulter l'extrait du procès-verbal : conformément à l'article 46, "
        "un extrait du procès-verbal de la séance d'examen des offres est publié "
        "sur le portail des marchés publics et affiché dans les locaux de "
        "l'organisme dont relève le maître d'ouvrage. Cet extrait mentionne les "
        "motifs d'élimination des concurrents évincés."
    ),
}


def build_fiches() -> list[dict]:
    brutes = decouper(lire_paragraphes(DOCX_ASSISTANT))

    fiches, vues = [], {}
    for brute in brutes:
        statut = classer(brute)
        solution = composer_solution(brute)
        libelle = brute["libelle"]

        # Un complément ne s'applique QUE là où le document n'a rien : il ne
        # doit jamais recouvrir une réponse déjà relevée sur le portail.
        if statut != "ok" and libelle in COMPLEMENTS:
            solution, statut = COMPLEMENTS[libelle], "ok"

        # Déduplication sur la RÉPONSE, pas sur le libellé. Le portail expose
        # la même réponse depuis plusieurs entrées de menu — sous le même
        # libellé (« Validation du compte par CNIE », sections 3 et 4) comme
        # sous des libellés différents (« Télécharger les quittances » et
        # « Téléchargement des quittances depuis le paiement multicanal »).
        # Deux fiches au contenu identique se partageraient les voix du
        # consensus ; une seule fiche, enrichie des libellés et variantes de
        # toutes ses entrées, est à la fois plus juste et plus facile à trouver.
        cle = solution if statut == "ok" and len(solution) >= 40 else (libelle.lower(), solution)
        if cle in vues:
            fiche = vues[cle]
            if brute["espace"] not in fiche["espaces"]:
                fiche["espaces"].append(brute["espace"])
            if libelle.lower() != fiche["probleme"].lower():
                fiche["aussi_appele"].append(libelle)
                for v in [libelle] + VARIANTES.get(libelle, []):
                    if v not in fiche["variantes"]:
                        fiche["variantes"].append(v)
            continue

        fiche = {
            "id": f"AST.{len(fiches) + 1}",
            "categorie": CATEGORIES.get(brute["rubrique"], brute["espace"] or "Portail eServices"),
            "probleme": libelle,
            "solution": solution,
            "status": "ok" if statut == "ok" else ("menu" if statut == "menu" else "no_answer"),
            "variantes": list(VARIANTES.get(libelle, [])),
            "liens": brute["liens"],
            "observation": brute["observation"],
            "aussi_appele": [],
            "espaces": [brute["espace"]],
            "rubrique": brute["rubrique"],
            "origine": "assistant eServices (relevé du 2026-09-01)",
        }
        vues[cle] = fiche
        fiches.append(fiche)

    # Renumérotation après déduplication pour garder des identifiants contigus
    for i, fiche in enumerate(fiches, 1):
        fiche["id"] = f"AST.{i}"
    return fiches


def main():
    print("=" * 62)
    print("  Extraction du guide de référence de l'assistant TGR (.docx)")
    print("=" * 62)

    if not os.path.exists(DOCX_ASSISTANT):
        print(f"Fichier introuvable : {DOCX_ASSISTANT}")
        sys.exit(1)

    fiches = build_fiches()
    ok = [f for f in fiches if f["status"] == "ok"]
    menus = [f for f in fiches if f["status"] == "menu"]
    sans = [f for f in fiches if f["status"] == "no_answer"]

    os.makedirs(os.path.dirname(ASSISTANT_FICHES_JSON), exist_ok=True)
    with open(ASSISTANT_FICHES_JSON, "w", encoding="utf-8") as f:
        json.dump(fiches, f, ensure_ascii=False, indent=2)

    print(f"\n{len(fiches)} entrées extraites :")
    print(f"  {len(ok):>3} avec réponse officielle  (indexées, servies verbatim)")
    print(f"  {len(sans):>3} sans réponse utilisable (escalade vers un agent)")
    print(f"  {len(menus):>3} nœuds de menu          (NON indexés)")
    print(f"  {sum(len(f['variantes']) for f in fiches):>3} variantes de questions usagers")
    print(f"  {sum(len(f['liens']) for f in fiches):>3} liens officiels conservés")

    sans_variantes = [f["id"] for f in ok if not f["variantes"]]
    if sans_variantes:
        print(f"\nFiches avec réponse mais SANS variante ({len(sans_variantes)}) — "
              f"elles ne seront trouvables que par leur libellé :")
        for fiche in fiches:
            if fiche["id"] in sans_variantes:
                print(f"    {fiche['id']:<8} {fiche['probleme'][:60]}")

    print(f"\nÉcrit dans : {ASSISTANT_FICHES_JSON}")


if __name__ == "__main__":
    main()
