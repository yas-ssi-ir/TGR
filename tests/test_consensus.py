"""
Tests du VOTE DE CONSENSUS — le cœur des décisions de l'assistant.

C'est ce mécanisme qui choisit la fiche à servir. Une erreur ici donne à
l'usager la réponse d'un autre problème que le sien.

Marqués « lourd » : importer agent_rag charge la pile ML (torch, chromadb).
"""
import pytest

pytestmark = pytest.mark.lourd


def passage(fiche_id, distance, status="ok"):
    """Fabrique un passage comme le retriever en renvoie."""
    return {
        "text": f"texte de la fiche {fiche_id}",
        "distance": distance,
        "source": "reclamations_xlsx",
        "fiche_id": fiche_id,
        "categorie": "Test",
        "status": status,
        "fichier": "",
    }


class TestRecevabilite:
    def test_deux_chunks_concordants_font_un_consensus(self):
        from agent_rag import fiche_consensus
        c = fiche_consensus([passage("2.1.1", 0.30), passage("2.1.1", 0.34)])
        assert c is not None
        assert c["fiche_id"] == "2.1.1"
        assert c["votes"] == 2

    def test_un_chunk_isole_et_lointain_ne_prouve_rien(self):
        from agent_rag import fiche_consensus
        assert fiche_consensus([passage("2.1.1", 0.36)]) is None

    def test_un_chunk_isole_mais_tres_proche_suffit(self):
        from agent_rag import fiche_consensus
        c = fiche_consensus([passage("5.1", 0.22)])
        assert c is not None and c["fiche_id"] == "5.1"

    def test_les_chunks_trop_lointains_ne_votent_pas(self):
        from agent_rag import fiche_consensus
        assert fiche_consensus([passage("4.3.2", 0.55), passage("4.3.2", 0.60)]) is None

    def test_les_extraits_de_pdf_ne_votent_pas(self):
        """Un chunk de PDF n'a pas de fiche : il ne peut pas voter."""
        from agent_rag import fiche_consensus
        pdf = passage("", 0.25)
        pdf["fichier"] = "FAQ_TGR.pdf"
        assert fiche_consensus([pdf, pdf]) is None

    def test_aucun_passage(self):
        from agent_rag import fiche_consensus
        assert fiche_consensus([]) is None


class TestProximitePrioritaire:
    """RÉGRESSION — trancher au nombre de voix favorisait mécaniquement les
    fiches les mieux dotées en variantes, au détriment d'une fiche nettement
    plus proche. Mesuré : 5 % de confusions, ramenées à 2 % après correction.
    """

    def test_la_fiche_la_plus_proche_gagne_meme_avec_moins_de_voix(self):
        from agent_rag import fiche_consensus
        passages = [
            passage("2.2.2", 0.34), passage("2.2.2", 0.35), passage("2.2.2", 0.36),
            passage("2.2.1", 0.20), passage("2.2.1", 0.21),
        ]
        c = fiche_consensus(passages)
        assert c["fiche_id"] == "2.2.1", "la fiche la plus proche doit l'emporter"


class TestDepartageBugConnu:
    """Répondre « problème connu, contactez le support » alors qu'une solution
    documentée existe est la pire des deux erreurs pour l'usager."""

    def test_une_solution_proche_prime_sur_un_bug_connu(self):
        # Distances sous DIST_CONSENSUS_MULTI_MAX (0.31, voir config.py) : le
        # départage doit rester opérant dans la fenêtre où le fast-path français
        # est désormais recevable, pas seulement dans l'ancienne fenêtre à 0.37.
        from agent_rag import fiche_consensus
        passages = [
            passage("4.1.2", 0.27, status="no_answer"),
            passage("4.1.2", 0.28, status="no_answer"),
            passage("4.2.2", 0.29), passage("4.2.2", 0.30),
        ]
        c = fiche_consensus(passages)
        assert c["fiche_id"] == "4.2.2"

    def test_un_bug_connu_garde_sa_question_si_rien_ne_le_talonne(self):
        from agent_rag import fiche_consensus
        passages = [
            passage("1.3.1", 0.25, status="no_answer"),
            passage("1.3.1", 0.26, status="no_answer"),
            passage("5.1", 0.36),
        ]
        c = fiche_consensus(passages)
        assert c["fiche_id"] == "1.3.1"


class TestMargeTranslangue:
    """RÉGRESSION — une question en arabe est structurellement plus « loin »
    du corpus français sans être hors sujet. Sans marge, « ما هي مشاكل رمز
    التحقق؟ » était rejetée alors que 5 de ses 6 voisins désignaient la même
    fiche, à 0.40.
    """

    def test_sans_marge_les_distances_arabes_sont_rejetees(self):
        from agent_rag import fiche_consensus
        assert fiche_consensus([passage("2.2.2", 0.40), passage("2.2.2", 0.41)]) is None

    def test_avec_marge_le_consensus_est_reconnu(self):
        from agent_rag import fiche_consensus
        from config import DIST_OFFSET_TRANSLANGUE
        c = fiche_consensus([passage("2.2.2", 0.40), passage("2.2.2", 0.41)],
                            marge=DIST_OFFSET_TRANSLANGUE)
        assert c is not None and c["fiche_id"] == "2.2.2"

    def test_la_marge_s_applique_a_l_arabe_uniquement(self):
        from agent_rag import marge_ecriture
        lexique_francais = {"compte", "passe", "attestation"}
        assert marge_ecriture("Mon mot de passe", lexique_francais) == 0.0
        assert marge_ecriture("كيف أحذف حسابي؟", lexique_francais) > 0.0


class TestOutils:
    def test_detection_de_la_langue(self):
        from agent_rag import langue_de
        assert langue_de("كيف أحذف حسابي؟") == "ar"
        assert langue_de("Mon mot de passe") == "fr"
        assert langue_de("") == "fr"

    def test_dedoublonnage_des_sources(self):
        """Une fiche et ses variantes pointent le même problème : l'usager ne
        doit voir qu'une source, avec sa meilleure distance."""
        from agent_rag import dedupe_sources
        sources = dedupe_sources([passage("2.1.1", 0.30), passage("2.1.1", 0.20),
                                  passage("5.1", 0.25)])
        assert len(sources) == 2
        distances = {s["fiche_id"]: s["distance"] for s in sources}
        assert distances["2.1.1"] == 0.20
