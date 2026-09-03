"""
Tests du banc d'essai — l'outil de mesure doit lui-même être digne de confiance.

Un script de test dont la logique de jugement est fausse produit des chiffres
rassurants ET faux : il déclare vert un système cassé, ou rouge un système sain.
C'est le pire des deux mondes, parce que plus personne ne va vérifier derrière.

Ces tests portent donc sur les deux décisions que prend `banc_essai.py` :
  - quels cas poser (charger_cas)
  - comment juger la réponse obtenue (juger)

Ils tournent sans la pile ML : aucun modèle n'est chargé.
"""
import pytest

from banc_essai import HORS_SUJET, charger_cas, juger, langue_de


class TestDetectionDeLangue:
    """Le rapport ventile les résultats par langue. Une détection fausse
    déplacerait des questions d'un bloc à l'autre et fausserait les taux."""

    @pytest.mark.parametrize("question,attendu", [
        ("Comment créer un compte sur eServices ?", "fr"),
        ("Je souhaite obtenir une attestation de salaire", "fr"),
        ("كيف أحذف حسابي؟", "ar"),
        ("لم أتوصل برمز التفعيل", "ar"),
        ("bghit ndir chikaya", "dj"),
        ("ma9dertch ndkhol l compte dyali", "dj"),
        ("site dyal TGR wa9ef ma khedamch", "dj"),
    ])
    def test_langue_reconnue(self, question, attendu):
        assert langue_de(question) == attendu

    def test_le_francais_a_chiffres_reste_du_francais(self):
        """Une date ou un numéro d'article ne fait pas de la darija."""
        for q in ("Bien déjà déclaré pour 2025 et les années antérieures",
                  "selon l'article 35 du décret n° 2-22-431",
                  "certificat d'imposition P1007"):
            assert langue_de(q) == "fr", q


def reponse(statut, fiche=None):
    """Fabrique un retour d'agent, comme answer() le produit."""
    return {"statut": statut,
            "sources": [{"id": fiche, "categorie": "Test", "distance": 0.2}] if fiche else []}


class TestJugement:
    """Le verdict porte sur ce que voit l'usager, pas sur un état interne."""

    CAS = {"question": "Comment créer un compte ?", "attendu": "AST.1",
           "categorie": "Compte", "langue": "fr"}
    CAS_REFUS = {"question": "Donne-moi une recette de tajine", "attendu": None,
                 "categorie": "Hors périmètre", "langue": "fr"}

    def test_bonne_fiche_servie(self):
        assert juger(self.CAS, reponse("SUCCESS", "AST.1"))[0] == "OK"

    def test_mauvaise_fiche_servie(self):
        verdict, detail = juger(self.CAS, reponse("SUCCESS", "AST.23"))
        assert verdict == "MAUVAISE_FICHE"
        assert "AST.23" in detail, "le détail doit nommer la fiche réellement servie"

    def test_question_du_portail_refusee(self):
        """Le pire défaut possible : éconduire un usager légitime."""
        assert juger(self.CAS, reponse("OUT_OF_SCOPE"))[0] == "REFUS_A_TORT"

    def test_hors_sujet_correctement_refuse(self):
        assert juger(self.CAS_REFUS, reponse("OUT_OF_SCOPE"))[0] == "OK"

    def test_hors_sujet_accepte_est_une_anomalie(self):
        """Un assistant d'administration qui répond hors de son domaine est
        plus dangereux qu'un assistant incomplet."""
        assert juger(self.CAS_REFUS, reponse("SUCCESS", "AST.46"))[0] == "ACCEPTE_A_TORT"

    def test_reponse_sans_source_est_signalee(self):
        """Une réponse sans source n'est pas ancrée : elle ne peut pas être
        comptée comme correcte, même si le statut est SUCCESS."""
        assert juger(self.CAS, reponse("INSUFFICIENT_KNOWLEDGE"))[0] == "SANS_SOURCE"

    def test_aucun_verdict_ok_par_defaut(self):
        """Garde-fou : tout état non prévu doit ressortir, jamais passer pour OK."""
        for statut in ("KNOWN_BUG", "NOT_GROUNDED", "SUCCESS"):
            verdict = juger(self.CAS, reponse(statut))[0]
            assert verdict != "OK", statut


@pytest.mark.lourd
class TestChargementDesCas:
    """Sur le vrai corpus — saute si la chaîne de préparation n'a pas tourné."""

    @staticmethod
    def cas(rapide=False, langue="tout"):
        c = charger_cas(rapide, langue)
        if not c:
            pytest.skip("corpus absent — chaîne de préparation non jouée")
        return c

    def test_le_hors_sujet_est_inclus_dans_le_test_complet(self):
        """C'est le seul contrôle qu'un corpus ne peut pas fournir lui-même :
        s'il disparaît, on ne teste plus jamais les refus."""
        cas = self.cas()
        refus = [c for c in cas if c["attendu"] is None]
        assert len(refus) == len(HORS_SUJET)

    def test_chaque_cas_porte_une_attente(self):
        for c in self.cas():
            assert c["question"].strip()
            assert "attendu" in c and "langue" in c

    def test_le_filtre_de_langue_ne_laisse_rien_passer(self):
        for lg in ("ar", "dj"):
            assert all(c["langue"] == lg for c in self.cas(langue=lg)), lg

    def test_le_mode_rapide_est_un_sous_ensemble(self):
        """--rapide doit alléger, pas changer ce qui est testé."""
        rapide = {c["question"] for c in self.cas(rapide=True)}
        complet = {c["question"] for c in self.cas()}
        assert rapide < complet

    def test_aucun_doublon_de_question_dans_une_meme_fiche(self):
        vus = set()
        for c in self.cas():
            cle = (c["attendu"], c["question"])
            assert cle not in vus, f"cas en double : {cle}"
            vus.add(cle)
