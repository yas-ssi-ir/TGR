"""
Machine à états de la relecture humaine.

Ce qu'une pastille verte affirme : « la TGR certifie que CE texte est juste ».
Chacun des cas ci-dessous correspond à une façon dont cette affirmation a
cessé d'être vraie en cours de route.
"""
import pytest

import revision


@pytest.fixture
def base(monkeypatch):
    """Une base en mémoire : les tests ne doivent pas toucher aux vraies
    réponses servies aux usagers."""
    etat = {"F.1": {"fr": "texte initial", "ar": "نص", "probleme": "P"}}
    monkeypatch.setattr(revision, "charger_reponses", lambda: etat)
    monkeypatch.setattr(revision, "enregistrer_reponses", lambda d: etat.update(d))
    return etat


class TestValidation:
    def test_enregistrer_sans_valider_ne_valide_pas(self, base):
        revision.enregistrer("F.1", "corrigé", "", valider=False)
        assert base["F.1"]["fr"] == "corrigé"
        assert not base["F.1"].get("validee")

    def test_valider_appose_la_signature_et_sa_date(self, base):
        revision.enregistrer("F.1", "", "", valider=True)
        assert base["F.1"]["validee"] is True
        assert base["F.1"]["validee_le"]

    def test_la_validation_peut_etre_retiree(self, base):
        """RÉGRESSION — un clic de travers sur « Valider » n'avait aucune
        marche arrière : la fiche restait certifiée par la TGR, et la seule
        issue était de retoucher le fichier des réponses à la main."""
        revision.enregistrer("F.1", "", "", valider=True)
        revision.enregistrer("F.1", "", "", valider=False, devalider=True)
        assert base["F.1"]["validee"] is False
        assert base["F.1"]["validee_le"] == ""

    def test_corriger_un_texte_deja_valide_retire_la_signature(self, base):
        """Le vert certifie UN texte, pas une fiche. Si le texte change, la
        TGR n'a jamais vu le nouveau : la fiche repart à relire."""
        revision.enregistrer("F.1", "premier jet", "", valider=True)
        revision.enregistrer("F.1", "texte remanié", "", valider=False)
        assert base["F.1"]["fr"] == "texte remanié"
        assert base["F.1"]["validee"] is False

    def test_une_fiche_inconnue_est_refusee(self, base):
        assert revision.enregistrer("INEXISTANTE", "x", "y", valider=True) is False


class TestChampsVides:
    def test_un_champ_laisse_vide_ne_efface_pas_le_texte(self, base):
        """La page envoie toujours les deux champs. Un arabe vide ne doit pas
        effacer la traduction existante."""
        revision.enregistrer("F.1", "nouveau français", "", valider=False)
        assert base["F.1"]["ar"] == "نص"


class TestArabeDefaillant:
    """« ar_defaillante » signale une traduction que le modèle a abandonnée.
    Un relecteur qui écrit l'arabe lui-même lève ce constat."""

    def test_traduire_a_la_main_leve_le_constat_d_echec(self, base):
        base["F.1"]["ar_defaillante"] = True
        revision.enregistrer("F.1", "", "نص عربي مكتوب بخط اليد للاختبار", valider=False)
        assert "ar_defaillante" not in base["F.1"]

    def test_un_texte_qui_n_est_pas_de_l_arabe_ne_leve_rien(self, base):
        """Coller du français dans le champ arabe ne règle pas le problème :
        l'usager arabophone recevrait un texte qu'il ne peut pas lire."""
        base["F.1"]["ar_defaillante"] = True
        revision.enregistrer("F.1", "", "Texte laisse en francais", valider=False)
        assert base["F.1"]["ar_defaillante"] is True


class TestTroisEtats:
    """« à relire » / « relue » / « validée ».

    RÉGRESSION — sans l'état intermédiaire, un rédacteur qui jugeait une
    réponse correcte n'avait aucun bouton pour l'enregistrer : le seul qui
    faisait avancer le compteur était « Valider ». Il signait donc à la place
    de la TGR. C'est arrivé sur la fiche 1.1.1.
    """

    def test_marquer_relue_ne_vaut_pas_signature(self, base):
        revision.enregistrer("F.1", "", "", valider=False, relue=True, relecteur="Yassir")
        assert base["F.1"]["relue"] is True
        assert base["F.1"]["relue_par"] == "Yassir"
        assert not base["F.1"].get("validee")

    def test_la_signature_porte_un_nom_et_une_date(self, base):
        """Une pastille verte qui ne dit pas qui l'a posée ne prouve rien."""
        revision.enregistrer("F.1", "", "", valider=True, relecteur="Agent TGR")
        assert base["F.1"]["validee_par"] == "Agent TGR"
        assert base["F.1"]["validee_le"]

    def test_valider_implique_relue(self, base):
        revision.enregistrer("F.1", "", "", valider=True, relecteur="Agent TGR")
        assert base["F.1"]["relue"] is True

    def test_corriger_le_texte_perime_les_deux_constats(self, base):
        """« Relue » et « validée » disent tous deux « j'ai lu CE texte » :
        ils ne suivent pas le texte suivant."""
        revision.enregistrer("F.1", "premier jet", "", valider=True, relecteur="Agent TGR")
        revision.enregistrer("F.1", "texte remanié", "", valider=False)
        assert base["F.1"]["validee"] is False
        assert base["F.1"]["relue"] is False
        assert base["F.1"]["validee_par"] == ""

    def test_retirer_la_signature_laisse_la_relecture(self, base):
        """Dévalider dit « je ne signe pas », pas « je n'ai rien lu »."""
        revision.enregistrer("F.1", "", "", valider=True, relecteur="Agent TGR")
        revision.enregistrer("F.1", "", "", valider=False, devalider=True)
        assert base["F.1"]["validee"] is False
        assert base["F.1"]["relue"] is True

    def test_un_nom_trop_long_est_coupe(self, base):
        revision.enregistrer("F.1", "", "", valider=True, relecteur="X" * 200)
        assert len(base["F.1"]["validee_par"]) == 60
