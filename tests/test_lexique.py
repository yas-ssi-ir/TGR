"""
Tests du filtre lexical — le premier rempart contre les questions hors sujet.

Ce filtre décide en 0 ms, sans appel au modèle. S'il se trompe, l'assistant
refuse une vraie question d'usager (grave) ou répond à n'importe quoi.
"""
import pytest

from lexique import (
    conflit_action,
    contient_terme_noyau,
    ecriture_darija,
    lexique_applicable,
    marge_ecriture,
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

    def test_applicable_a_l_arabe_si_le_corpus_est_arabophone(self):
        lexique_arabe = {f"كلمة{i}" for i in range(40)}
        assert lexique_applicable("كيف أحذف حسابي؟", lexique_arabe) is True

    def test_quelques_traductions_ne_font_pas_un_corpus_arabophone(self):
        """RÉGRESSION — ajouter 47 variantes en arabe (2,8 % du lexique)
        franchissait l'ancien seuil de « 30 mots » : le filtre lexical se
        remettait alors à rejeter les questions arabes, exactement la panne
        qu'il devait empêcher. C'est la PART du lexique qui compte."""
        corpus_francais = {f"mot{i}" for i in range(1600)} | {f"كلمة{i}" for i in range(47)}
        assert lexique_applicable("كيف أحذف حسابي؟", corpus_francais) is False


class TestGardeFouAntonymes:
    """RÉGRESSION — « comment créer mon compte » servait la fiche « supprimer
    mon compte ».

    Mesuré en production : 3 chunks de la fiche de suppression concordaient,
    le consensus la déclarait certaine, et l'assistant expliquait à un usager
    voulant ouvrir un compte comment le détruire. Les embeddings ne voient pas
    l'opposition de sens — il faut un signal lexical explicite.
    """

    CREER = "Création de compte : procéder à la création de votre compte."
    SUPPRIMER = "Demandes de suppression de compte. Se connecter et supprimer le compte."

    def test_creer_ne_tombe_pas_sur_supprimer(self):
        assert conflit_action("comment créer mon compte", self.SUPPRIMER) is True

    def test_supprimer_ne_tombe_pas_sur_creer(self):
        assert conflit_action("comment supprimer mon compte", self.CREER) is True

    def test_la_bonne_fiche_passe(self):
        assert conflit_action("comment créer mon compte", self.CREER) is False
        assert conflit_action("comment supprimer mon compte", self.SUPPRIMER) is False

    def test_une_fiche_qui_traite_des_deux_actions_est_conservee(self):
        """« Ouvrez un nouveau compte et demandez la fermeture de l'ancien »
        répond légitimement aux deux questions : la rejeter serait une perte."""
        les_deux = ("vous pouvez ouvrir un nouveau compte dans la nouvelle agence "
                    "et demander la fermeture de l'ancien compte")
        assert conflit_action("je veux ouvrir un compte", les_deux) is False
        assert conflit_action("je veux fermer mon compte", les_deux) is False

    def test_une_question_sans_action_ne_declenche_jamais_le_veto(self):
        assert conflit_action("mon mot de passe ne marche plus", self.SUPPRIMER) is False

    def test_activer_ne_tombe_pas_sur_desactiver(self):
        assert conflit_action("comment activer mon compte",
                              "Compte désactivé pour inactivité ou raison de sécurité") is True

    def test_inscription_ne_tombe_pas_sur_desinscription(self):
        assert conflit_action("comment m'inscrire aux eservices",
                              "Je veux me désinscrire du portail") is True


class TestDarijaLatine:
    """La darija écrite en lettres latines — « bghit ndir chikaya » — est
    aussi éloignée d'une documentation française que l'arabe, mais invisible
    pour un test d'écriture. Mesuré : 5 des 6 questions du corpus laissées
    sans réponse étaient de la darija latine, et l'une d'elles était même
    REFUSÉE comme hors sujet (0,351 ≥ seuil 0,34).
    """

    DARIJA = [
        "bghit ndir chikaya",
        "kifach nsajjel f eservices TGR",
        "site dyal TGR wa9ef ma khedamch",
        "bdelt telephone w mabqitch n9der ndkhol b code",
        "ma3reftch ndkhol l compte dyali",
        "telephone jdid, Google Authenticator makaynch",
    ]
    # Français contenant des chiffres ou des sigles : aucun ne doit être pris
    # pour de la darija, sinon on relâcherait les seuils sur du français.
    FRANCAIS = [
        "Bien déjà déclaré pour 2025 et les années antérieures",
        "Je souhaite obtenir un certificat d'imposition ou de non-imposition P1007?",
        "selon l'article 35 du décret n° 2-22-431",
        "conformément à la loi n° 14-25 relative à l'assiette",
        "appeler le centre d'appel de la DGI : 0537273727",
        "Comment créer un compte sur eServices ?",
    ]

    def test_la_darija_est_reconnue(self):
        for q in self.DARIJA:
            assert ecriture_darija(q) is True, q

    def test_le_francais_n_est_jamais_pris_pour_de_la_darija(self):
        for q in self.FRANCAIS:
            assert ecriture_darija(q) is False, q

    def test_le_filtre_lexical_ne_s_applique_pas_a_la_darija(self):
        """Le lexique est construit sur une documentation française : exiger
        un mot commun reviendrait à éconduire les usagers marocains."""
        corpus_francais = {f"mot{i}" for i in range(1600)}
        assert lexique_applicable("bghit ndir chikaya", corpus_francais) is False
        assert lexique_applicable("Comment déposer une réclamation ?", corpus_francais) is True

    def test_la_darija_recoit_une_marge_plus_serree_que_l_arabe(self):
        """Les 14 questions darija du corpus sont à ≤ 0,351, contre 0,40-0,41
        pour l'arabe : une marge identique gaspillerait de la tolérance et
        laisserait passer davantage de hors-sujet."""
        corpus_francais = {f"mot{i}" for i in range(1600)}
        darija = marge_ecriture("bghit ndir chikaya", corpus_francais)
        arabe = marge_ecriture("كيف أحذف حسابي؟", corpus_francais)
        francais = marge_ecriture("Comment déposer une réclamation ?", corpus_francais)
        assert francais == 0.0
        assert 0 < darija < arabe
