"""
Phase 1 — Préparation des données.
Transforme request_response.xlsx (46 lignes brutes "problème => solution")
en fiches Q/R propres et structurées : data/processed/qa_fiches.json

Traitement :
  - regroupe les lignes de continuation (Ø ..., phrases isolées) avec leur fiche parente
  - sépare problème / solution (sur =>, è, ou :)
  - déduit la catégorie depuis la numérotation (1.x → Mot de passe, etc.)
  - marque les fiches sans solution exploitable (BUG, Incompréhensible) → status "no_answer"
  - ajoute des variantes de questions en langage usager (dictionnaire manuel)
"""
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import XLSX_RECLAMATIONS, QA_FICHES_JSON, DATA_PROCESSED_DIR, CATEGORIES

import openpyxl

# Motif d'une nouvelle fiche : commence par "1.1.1." ou "5.1." etc.
RE_NEW_FICHE = re.compile(r"^\s*(\d+(?:\.\d+)+)\.?\s*(.*)$", re.DOTALL)
# Séparateurs problème => solution, par ordre de priorité
RE_SEPARATORS = [r"\s*=>\s*", r"\s*è\s+", r"\s*:\s*"]

# Solutions considérées comme inexploitables → fiche "no_answer"
NO_ANSWER_MARKERS = ["bug", "incompréhensible", "amélioration", "confidentiel"]


def clean(text: str) -> str:
    """Nettoie un fragment de texte (espaces insécables, puces, espaces multiples)."""
    if not text:
        return ""
    text = text.replace("\xa0", " ").replace("﻿", "")
    text = re.sub(r"^[Øø>\-•\s]+", "", text)          # puces en début de ligne
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def read_xlsx_lines(path: str) -> list[str]:
    """Lit toutes les cellules non vides de la feuille 1 (colonnes A-B)."""
    wb = openpyxl.load_workbook(path)
    ws = wb.worksheets[0]
    lines = []
    for row in ws.iter_rows(values_only=True):
        for cell in row:
            if cell and str(cell).strip():
                lines.append(str(cell))
    return lines


def group_fiches(lines: list[str]) -> list[dict]:
    """Regroupe les lignes brutes en fiches : une fiche commence par 'N.N[.N].'"""
    fiches = []
    current = None
    for line in lines:
        m = RE_NEW_FICHE.match(line)
        if m:
            if current:
                fiches.append(current)
            current = {"numero": m.group(1), "raw": clean(m.group(2))}
        elif current:
            # ligne de continuation → rattachée à la fiche courante
            current["raw"] += "\n" + clean(line)
    if current:
        fiches.append(current)
    return fiches


def split_probleme_solution(raw: str) -> tuple[str, str]:
    """Sépare le texte brut en (problème, solution) sur le premier séparateur trouvé."""
    first_line, _, rest = raw.partition("\n")
    for sep in RE_SEPARATORS:
        parts = re.split(sep, first_line, maxsplit=1)
        if len(parts) == 2:
            probleme = clean(parts[0])
            solution = clean(parts[1])
            if rest:
                solution = (solution + "\n" + clean(rest)).strip()
            return probleme, solution
    # aucun séparateur : tout est le problème, la suite est la solution
    return clean(first_line), clean(rest)


def detect_status(solution: str) -> str:
    """Détermine si la solution est exploitable."""
    if not solution or len(solution) < 15:
        return "no_answer"
    low = solution.lower()
    if any(marker in low for marker in NO_ANSWER_MARKERS) and len(solution) < 80:
        return "no_answer"
    return "ok"


# ── Variantes en langage usager (levier qualité n°1 du retrieval) ────
# Clé = numéro de fiche ; valeurs = formulations réelles d'usagers (FR + darija latin)
# Chaque fiche porte des variantes en TROIS écritures : français, darija en
# alphabet latin, et arabe.
#
# L'arabe n'est pas un ornement. Mesuré : aucune des 23 fiches n'avait de
# variante en écriture arabe — leurs seules formulations « marocaines » étaient
# en lettres latines. Un usager arabophone ne pouvait donc les atteindre que par
# l'embedding seul, à 0,40 de distance : le chemin le plus fragile du système.
# Conséquence observée, « ما هي مشاكل رمز التحقق؟ » (problème de code de
# vérification) partait sur la fiche « Validation du compte par CNIE » — les deux
# parlent de « vérification », et rien côté MFA n'était écrit en arabe pour la
# retenir.
VARIANTES = {
    "1.1.1": ["Je n'arrive pas à me connecter, mon mot de passe ne marche plus",
               "Mon mot de passe est refusé alors qu'il est correct",
               "makayn mochkil f password dyali walakin makhdamch",
               "ma9dertch ndkhol l compte dyali",
               "password dyali ma bghach ykhdem",
               "كلمة المرور صحيحة لكن الموقع يرفضها",
               "لا أستطيع تسجيل الدخول رغم أن كلمة المرور صحيحة"],
    "1.1.2": ["Le lien de réinitialisation du mot de passe ne fonctionne pas",
               "J'ai cliqué sur le lien reçu par email mais il est expiré",
               "Impossible de réinitialiser mon mot de passe",
               "lien dyal reinitialisation dyal password ma khdamch",
               "رابط إعادة تعيين كلمة المرور لا يعمل",
               "توصلت برابط إعادة التعيين لكنه منتهي الصلاحية"],
    "1.2.1": ["Le système dit que mon compte existe déjà mais je ne le trouve pas",
               "Compte introuvable alors que je me suis déjà inscrit",
               "Mon compte a disparu, il existe pour le site mais pas pour moi",
               "compte dyali kayn walakin ma3ndich access",
               "النظام يقول إن حسابي موجود لكنني لا أجده",
               "سجلت من قبل لكنني لا أعثر على حسابي"],
    "1.2.2": ["Mon PPR est lié à un ancien compte que j'ai supprimé",
               "Je ne peux pas créer un compte car mon PPR est déjà utilisé",
               "رقم التأجير مرتبط بحساب قديم قمت بحذفه",
               "لا أستطيع إنشاء حساب لأن رقم التأجير مستعمل من قبل"],
    "1.3.1": ["Mon compte est bloqué après plusieurs tentatives de connexion",
               "Compte bloqué trop de tentatives",
               "حسابي محجوب بعد عدة محاولات للدخول",
               "تم حجب حسابي بسبب كثرة محاولات تسجيل الدخول"],
    "1.3.2": ["Mon compte a été désactivé pour inactivité",
               "Compte désactivé raison de sécurité",
               "تم تعطيل حسابي بسبب عدم الاستعمال",
               "حسابي معطل لأسباب أمنية"],
    "2.1.1": ["J'ai changé de téléphone et je n'ai plus accès à Google Authenticator",
               "J'ai perdu mon téléphone comment me connecter avec le MFA",
               "bdelt telephone w mabqitch n9der ndkhol b code",
               "telephone jdid, Google Authenticator makaynch",
               "Nouveau téléphone, comment récupérer mon authentification",
               "غيرت هاتفي ولم أعد أتوفر على تطبيق المصادقة",
               "ضاع هاتفي كيف أدخل إلى حسابي بالمصادقة الثنائية"],
    "2.1.2": ["J'ai désinstallé l'application d'authentification par erreur",
               "J'ai supprimé Google Authenticator comment récupérer mes codes",
               "حذفت تطبيق المصادقة عن طريق الخطأ",
               "أزلت تطبيق Google Authenticator كيف أسترجع رموزي"],
    "2.1.3": ["Mon téléphone est en panne je ne peux plus générer le code OTP",
               "Téléphone cassé impossible d'avoir le code de connexion",
               "هاتفي معطل ولا أستطيع توليد رمز الدخول",
               "الهاتف خارج الخدمة كيف أحصل على رمز الدخول"],
    "2.2.1": ["Je n'ai pas gardé les codes de secours au moment de l'activation",
               "Je n'ai pas noté mes codes de récupération MFA",
               "J'ai perdu la feuille des codes de secours",
               "لم أحتفظ برموز الاسترجاع عند التفعيل",
               "ضاعت مني ورقة رموز الاحتياط"],
    "2.2.2": ["J'ai utilisé tous mes codes de secours, il ne m'en reste aucun",
               "Mes codes de secours sont épuisés",
               "Plus aucun code de secours disponible comment en obtenir de nouveaux",
               "استعملت جميع رموز الاسترجاع ولم يبق لي أي رمز",
               "نفدت رموز الاحتياط كيف أحصل على رموز جديدة"],
    "2.2.3": ["Le bouton régénérer les codes de secours ne fonctionne pas",
               "Impossible de régénérer de nouveaux codes depuis mon compte",
               "La régénération échoue quand je demande de nouveaux codes",
               "زر إعادة توليد رموز الاسترجاع لا يعمل",
               "لا أستطيع توليد رموز احتياط جديدة من حسابي"],
    "2.3.1": ["Le code OTP est toujours invalide alors qu'il est bien saisi",
               "Mon code Google Authenticator est refusé à chaque tentative",
               "Le code à 6 chiffres est rejeté, message code invalide",
               "code OTP ghalat f kol mrra",
               # Variante volontairement précise : « مشكل في رمز التحقق »
               # (problème avec le code) désignait aussi bien un code REFUSÉ
               # que des codes ÉPUISÉS, et partait sur la fiche 2.2.2.
               "رمز التحقق مرفوض في كل مرة رغم إدخاله بشكل صحيح",
               "التطبيق يعطيني رمزا لكن الموقع يرفضه",
               "الرمز المكون من ستة أرقام غير صالح"],
    "2.3.2": ["Le QR code d'activation ne s'affiche pas ou ne fonctionne pas",
               "Je n'arrive pas à scanner le code QR pour activer le MFA",
               "Aucun QR code proposé lors de l'activation",
               "رمز الاستجابة السريعة لا يظهر عند التفعيل",
               "لا أستطيع مسح رمز QR لتفعيل المصادقة الثنائية"],
    "3.1.1": ["Mon téléphone n'a pas de NFC comment vérifier ma carte d'identité",
               "Téléphone sans NFC pour la vérification CNIE",
               "telephone dyali ma fihch NFC bach n verifier CNIE",
               "هاتفي لا يتوفر على تقنية NFC للتحقق من البطاقة الوطنية",
               "كيف أتحقق من بطاقتي الوطنية بدون خاصية NFC"],
    "3.1.2": ["L'application Mon e-ID n'est pas compatible avec mon téléphone",
               "Je ne peux pas installer l'application d'identité numérique",
               "تطبيق هويتي الرقمية غير متوافق مع هاتفي",
               "لا أستطيع تثبيت تطبيق الهوية الرقمية على هاتفي"],
    "3.1.3": ["Mon téléphone ne lit pas la carte d'identité",
               "La lecture NFC de ma CNIE ne marche pas",
               "telephoni maki9rach la carte nationale",
               "هاتفي لا يقرأ البطاقة الوطنية",
               "قراءة البطاقة الوطنية بتقنية NFC لا تعمل"],
    "3.2.1": ["Je ne trouve pas le bouton vérification via CNIE",
               "Le bouton de vérification CNIE n'apparaît pas",
               "لا أجد زر التحقق بواسطة البطاقة الوطنية",
               "زر التحقق عبر البطاقة الوطنية لا يظهر في حسابي"],
    "3.2.2": ["Les champs de saisie sont bloqués je ne peux rien modifier",
               "Impossible de modifier mes informations les champs sont grisés",
               "خانات الإدخال مقفلة ولا أستطيع تعديل أي شيء",
               "لا يمكنني تغيير معلوماتي الشخصية الخانات غير نشطة"],
    "3.3.1": ["J'ai l'ancienne carte CIN pas la nouvelle CNIE",
               "Est-ce que l'ancienne carte d'identité fonctionne pour la vérification",
               "أتوفر على البطاقة الوطنية القديمة وليست الإلكترونية",
               "هل تصلح البطاقة الوطنية القديمة لعملية التحقق"],
    "3.3.2": ["Ma carte CNIE neuve n'est pas lisible par le téléphone",
               "Carte d'identité neuve non reconnue par NFC",
               "بطاقتي الوطنية الجديدة غير مقروءة من طرف الهاتف",
               "البطاقة الإلكترونية الجديدة لا يتعرف عليها الهاتف"],
    "4.1.1": ["Je me suis inscrit mais je ne vois pas le menu du service",
               "Après inscription le service n'apparaît pas",
               "سجلت في الخدمة لكنها لا تظهر في القائمة",
               "بعد التسجيل لا تظهر لي الخدمة في حسابي"],
    # Les fiches 4.x décrivent des bugs de la page « adhésion aux services ».
    # Leurs variantes disaient « rien ne se passe », « ne répond pas », « erreur
    # technique » : des formules qui collent à N'IMPORTE QUELLE plainte et qui
    # captaient donc les questions des autres catégories (mesuré : une question
    # sur le lien de réinitialisation partait vers la fiche 4.1.2). Chaque
    # variante nomme désormais le contexte réel : adhésion, service, inscription.
    "4.1.2": ["Mon adhésion à un service est validée mais le service n'apparaît pas",
               "J'ai adhéré à un téléservice mais je n'y ai toujours pas accès",
               "Adhésion confirmée, service inaccessible dans mon espace",
               "تم قبول انخراطي في الخدمة لكنها لا تظهر لي",
               "انخرطت في خدمة إلكترونية ولا أستطيع الوصول إليها"],
    "4.2.1": ["Le bouton « Ajouter » de la page d'adhésion aux services reste inactif",
               "Je clique sur « Ajouter un service » mais aucune fenêtre ne s'ouvre",
               "Impossible d'ajouter un téléservice, le bouton est grisé",
               "زر إضافة خدمة لا يستجيب عند الضغط عليه",
               "لا أستطيع إضافة خدمة إلكترونية الزر غير نشط"],
    "4.2.2": ["Message erreur utilisateur déjà inscrit",
               "Le site dit que je suis déjà inscrit mais ce n'est pas le cas",
               "رسالة خطأ تقول إن المستخدم مسجل من قبل",
               "الموقع يقول إنني مسجل مسبقا وهذا غير صحيح"],
    "4.2.3": ["Mon dernier salaire est rejeté",
               "Rejet du salaire lors de l'inscription",
               "تم رفض آخر أجر عند التسجيل",
               "الأجر الصافي الأخير مرفوض في استمارة التسجيل"],
    "4.3.1": ["Mon tableau de bord est vide",
               "Mon tableau de bord n'affiche aucune de mes données fiscales",
               "Mon espace personnel eServices reste vide après connexion",
               "لوحة القيادة فارغة ولا تعرض أي معطيات",
               "الفضاء الشخصي يبقى فارغا بعد تسجيل الدخول"],
    "4.3.2": ["Une erreur technique s'affiche pendant mon adhésion à un service",
               "Message d'erreur technique lors de l'inscription à un téléservice",
               "Erreur technique au moment de valider mon adhésion",
               "تظهر لي رسالة خطأ تقني أثناء الانخراط في خدمة",
               "خطأ تقني عند تأكيد الانخراط في خدمة إلكترونية"],
    "5.1": ["Comment supprimer mon compte",
             "Je veux supprimer définitivement mon compte eservices",
             "bghit nmse7 compte dyali",
             "أريد حذف حسابي نهائيا من بوابة الخدمات الإلكترونية",
             "ما هي مسطرة حذف الحساب"],
    "5.2": ["Comment changer mon adresse email",
             "Je veux modifier l'email de mon compte",
             "bghit nbdel email dyali",
             "أريد تعديل البريد الإلكتروني الخاص بحسابي",
             "ما هي مسطرة تغيير عنوان البريد الإلكتروني"],
}


def build_fiches() -> list[dict]:
    lines = read_xlsx_lines(XLSX_RECLAMATIONS)
    print(f"Lignes non vides lues : {len(lines)}")

    grouped = group_fiches(lines)
    print(f"Fiches regroupées      : {len(grouped)}")

    fiches = []
    for g in grouped:
        numero = g["numero"]
        probleme, solution = split_probleme_solution(g["raw"])
        categorie = CATEGORIES.get(numero.split(".")[0], "Autre")
        status = detect_status(solution)
        fiches.append({
            "id": numero,
            "categorie": categorie,
            "probleme": probleme,
            "solution": solution,
            "status": status,
            "variantes": VARIANTES.get(numero, []),
        })
    return fiches


def main():
    os.makedirs(DATA_PROCESSED_DIR, exist_ok=True)
    fiches = build_fiches()

    ok = [f for f in fiches if f["status"] == "ok"]
    no_ans = [f for f in fiches if f["status"] == "no_answer"]

    with open(QA_FICHES_JSON, "w", encoding="utf-8") as f:
        json.dump(fiches, f, ensure_ascii=False, indent=2)

    print("\n=== Bilan ===")
    print(f"Fiches totales   : {len(fiches)}")
    print(f"  - exploitables : {len(ok)}")
    print(f"  - no_answer    : {len(no_ans)}  (BUG / incompréhensible / vide)")
    variantes_count = sum(len(f["variantes"]) for f in fiches)
    print(f"Variantes usager : {variantes_count}")
    print(f"\nFichier écrit : {QA_FICHES_JSON}")

    print("\n--- Aperçu (3 premières fiches) ---")
    for fiche in fiches[:3]:
        print(f"\n[{fiche['id']}] ({fiche['categorie']}) [{fiche['status']}]")
        print(f"  Problème : {fiche['probleme'][:100]}")
        print(f"  Solution : {fiche['solution'][:100]}")


if __name__ == "__main__":
    main()
