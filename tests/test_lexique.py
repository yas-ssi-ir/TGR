"""
Tests du filtre lexical — le premier rempart contre les questions hors sujet.

Ce filtre décide en 0 ms, sans appel au modèle. S'il se trompe, l'assistant
refuse une vraie question d'usager (grave) ou répond à n'importe quoi.
"""
import pytest

from lexique import (
    contient_terme_noyau,
    lexique_applicable,
    mots_significatifs,
    normaliser,
    recouvrement,
)


class TestNormalisation:
    def test_supprime_les_accents(self):
        assert normaliser("Réclamation") == "reclamation"
        assert normaliser("DÉCLARÉ") == "declare"

    def test_mots_significatifs_ecarte_les_mots_outils(self):
        # « plus » est un mot-outil, « mot » et « de » sont trop courts
        assert mots_significatifs("Mon mot de passe ne marche plus") == {"passe", "marche"}

    def test_mots_significatifs_texte_vide(self):
        assert mots_significatifs("") == set()
        assert mots_significatifs(None) == set()


class TestTermesMetier:
    """Le signal POSITIF : la question emploie-t-elle un mot du portail ?"""

    @pytest.mark.parametrize("question", [
        "Mon mot de passe ne marche plus",
        "Comment payer ma taxe d'habitation ?",
        "Je veux une attestation de salaire",
        "Où télécharger ma quittance ?",
        "Comment suivre ma réclamation ?",
        "Mon téléphone n'a pas le NFC",
    ])
    def test_reconnait_les_vraies_questions(self, question):
        assert contient_terme_noyau(question) is True

    @pytest.mark.parametrize("question", [
        "Donne-moi une recette de tajine",
        "Quelle est la capitale de la France ?",
        "Raconte-moi une blague",
    ])
    def test_ignore_les_questions_hors_sujet(self, question):
        assert contient_terme_noyau(question) is False


class TestRecouvrement:
    def test_compte_les_mots_communs(self):
        lexique = {"passe", "compte", "oublie"}
        assert recouvrement("mot de passe oublié", lexique) == 2

    def test_aucun_mot_commun(self):
        assert recouvrement("recette de tajine", {"passe", "compte"}) == 0


class TestApplicabilite:
    """RÉGRESSION — le filtre lexical avait rejeté toutes les questions arabes.

    Le lexique est construit sur une documentation française : une question en
    arabe n'a évidemment aucun mot en commun. L'appliquer telle quelle revenait
    à refuser tous les usagers arabophones.
    """

    def test_inapplicable_a_l_arabe_quand_le_corpus_est_francais(self):
        lexique_francais = {"compte", "passe", "attestation", "quittance"}
        assert lexique_applicable("كيف أحذف حسابي؟", lexique_francais) is False

    def test_applicable_au_francais(self):
        assert lexique_applicable("Mon mot de passe", {"compte", "passe"}) is True

    def test_inapplicable_si_lexique_vide(self):
        assert lexique_applicable("Mon mot de passe", set()) is False

    def test_applicable_a_l_arabe_si_le_corpus_contient_de_l_arabe(self):
        lexique_arabe = {f"كلمة{i}" for i in range(40)}
        assert lexique_applicable("كيف أحذف حسابي؟", lexique_arabe) is True
