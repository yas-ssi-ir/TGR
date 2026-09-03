"""
Ce que la lettre de réclamation a le droit de promettre.

Le module n'expose que deux gestes : DÉPOSER et SUIVRE. Toute formule qui
en suggère un troisième laisse l'usager attendre devant une porte qui
n'existe pas.

Ces contrôles lisent le fichier source plutôt que d'importer le module :
reclamation_handler tire le client LLM et le retriever, absents de l'étape
rapide de la CI.
"""
import os
import re

import pytest

SOURCE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                      "src", "reclamation_handler.py")


@pytest.fixture(scope="module")
def code():
    with open(SOURCE, encoding="utf-8") as f:
        return f.read()


class TestPromessesTenables:
    """RÉGRESSION — la lettre invitait à « répondre à cette réclamation ».
    Aucune interface ne le permet : l'onglet Suivi ne fait que lire, et il
    n'existe aucune route pour répondre à un dossier."""

    @pytest.mark.parametrize("promesse", [
        r"r[ée]pondez\s+[àa]\s+cette\s+r[ée]clamation",
        r"r[ée]pondre\s+[àa]\s+cette\s+r[ée]clamation",
        r"الرد على هذه الشكاية",
    ])
    def test_la_lettre_ne_promet_pas_une_reponse_impossible(self, code, promesse):
        trouve = re.search(promesse, code, re.IGNORECASE)
        assert not trouve, (
            "la lettre invite à répondre à la réclamation, ce que le portail "
            "ne permet pas — proposer plutôt le suivi ou un nouveau dépôt")

    def test_la_lettre_oriente_vers_ce_qui_existe(self, code):
        assert "onglet" in code and "Suivi" in code, \
            "la lettre doit indiquer où suivre le dossier"
        assert "nouvelle réclamation" in code, \
            "la lettre doit indiquer le recours réellement disponible"

    def test_les_deux_langues_sont_traitees(self, code):
        """Un usager arabophone reçoit la même lettre, pas une version
        oubliée lors d'une correction."""
        assert "التتبع" in code, "la clôture arabe n'oriente pas vers le suivi"
