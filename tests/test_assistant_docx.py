"""
Tests de l'extraction du guide de l'assistant TGR (.docx).

Cette source est la plus fiable du projet : ce sont les réponses réellement
servies aux usagers sur le portail, avec leurs liens officiels. Une erreur
d'extraction se traduit directement par un lien mort ou une procédure
tronquée envoyée à un usager.
"""
import pytest

from prepare_assistant import classer, composer_solution, reponses_dupliquees


def brute(lignes, liens=None, libelle="Test"):
    return {"libelle": libelle, "espace": "Test", "rubrique": "Test",
            "lignes": lignes, "liens": liens or [], "observation": ""}


class TestClassement:
    """ok / no_answer / menu — voir l'en-tête de prepare_assistant."""

    def test_une_vraie_reponse_est_exploitable(self):
        assert classer(brute(["Pour télécharger votre quittance, cliquez sur ce lien"],
                             ["https://eservices.tgr.gov.ma/my/mesquittances"])) == "ok"

    def test_un_noeud_de_navigation_n_est_pas_une_reponse(self):
        """Sans ce tri, le nœud « Désinscription du compte » (qui ne fait que
        poser une question) volerait les questions à la fiche fille qui porte
        réellement la procédure."""
        assert classer(brute(
            ["Le chatbot demande de préciser si l'utilisateur a encore accès au compte."]
        )) == "menu"

    def test_un_echec_de_l_assistant_est_signale_et_jamais_servi(self):
        assert classer(brute(["Pouvez-vous dire cela autrement ? Je ne comprends pas."])) \
            == "no_answer"
        assert classer(brute(["Aucune réponse spécifique n'est fournie."])) == "no_answer"

    def test_une_entree_vide_est_un_menu(self):
        assert classer(brute([])) == "menu"

    def test_un_menu_porteur_de_lien_reste_exploitable(self):
        """Une désambiguïsation qui finit par donner le lien officiel répond
        quand même à l'usager."""
        assert classer(brute(["Le chatbot affiche d'abord une désambiguïsation."],
                             ["https://eservices.tgr.gov.ma/my/reclamation"])) == "ok"


class TestComposition:
    def test_les_liens_officiels_sont_conserves(self):
        """Un lien perdu, c'est un usager qui n'atteint jamais le service."""
        sortie = composer_solution(brute(["Activez votre compte via ce lien"],
                                         ["https://eservices.tgr.gov.ma/my/auth/activate"]))
        assert "https://eservices.tgr.gov.ma/my/auth/activate" in sortie

    def test_une_reponse_reduite_a_un_lien_reste_une_reponse(self):
        sortie = composer_solution(brute([], ["https://eservices.tgr.gov.ma/my/mesquittances"]))
        assert sortie.strip()


@pytest.fixture(scope="module")
def fiches():
    import os

    from config import DOCX_ASSISTANT
    if not os.path.exists(DOCX_ASSISTANT):
        pytest.skip("guide de l'assistant absent de data/raw/")
    from prepare_assistant import build_fiches
    return build_fiches()


@pytest.mark.lourd
class TestIntegriteDuGuide:
    """Vérifications sur le vrai document, s'il est présent."""

    def test_les_identifiants_sont_uniques(self, fiches):
        ids = [f["id"] for f in fiches]
        assert len(ids) == len(set(ids))

    def test_aucune_fiche_ne_duplique_la_reponse_d_une_autre(self, fiches):
        """Deux fiches au contenu identique se partagent les voix du consensus."""
        assert reponses_dupliquees(fiches) == []

    def test_la_creation_de_compte_est_couverte(self, fiches):
        """La question qui a révélé le bug : elle DOIT avoir sa fiche."""
        creation = [f for f in fiches if "création de compte" in f["probleme"].lower()]
        assert creation, "aucune fiche ne couvre la création de compte"
        assert creation[0]["status"] == "ok"
        assert any("créer" in v.lower() for v in creation[0]["variantes"])

    def test_toute_fiche_avec_reponse_est_trouvable(self, fiches):
        """Sans variante, une fiche n'est atteignable que par son libellé
        exact — autant dire jamais, un usager n'écrit pas des étiquettes."""
        muettes = [f["id"] for f in fiches if f["status"] == "ok" and not f["variantes"]]
        assert muettes == []
