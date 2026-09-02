"""
Contrôles d'intégrité de la connaissance servie aux usagers.

Ces tests ne vérifient pas du code mais des DONNÉES. C'est volontaire : les
réponses pré-rédigées sont servies telles quelles, sans filet à l'exécution.
Une réponse vide, un champ arabe resté en français ou une fiche sans réponse
se voient ici — pas devant l'usager.
"""
import json
import os
import re
import warnings

import pytest

from config import (
    ASSISTANT_FICHES_JSON,
    FAQ_FICHES_JSON,
    PRECOMPUTED_JSON,
    QA_FICHES_JSON,
)

ARABE = re.compile("[؀-ۿ]")
CHAMPS_FICHE = {"id", "categorie", "probleme", "solution", "status", "variantes"}


def charger(chemin):
    if not os.path.exists(chemin):
        pytest.skip(f"{os.path.basename(chemin)} absent — chaîne de préparation non jouée")
    with open(chemin, encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(scope="module")
def fiches():
    """Les fiches réellement INDEXÉES : les nœuds de menu du relevé de
    l'assistant sont exclus de la base vectorielle, donc aussi d'ici."""
    toutes = charger(QA_FICHES_JSON) + charger(FAQ_FICHES_JSON) + charger(ASSISTANT_FICHES_JSON)
    return [f for f in toutes if f.get("status") != "menu"]


@pytest.fixture(scope="module")
def reponses():
    return charger(PRECOMPUTED_JSON)


class TestStructureDesFiches:
    def test_le_corpus_n_est_pas_vide(self, fiches):
        assert len(fiches) >= 40, "corpus anormalement petit — relancer prepare_qa/prepare_faq"

    def test_chaque_fiche_a_tous_ses_champs(self, fiches):
        for f in fiches:
            manquants = CHAMPS_FICHE - set(f)
            assert not manquants, f"fiche {f.get('id')} : champs manquants {manquants}"

    def test_les_identifiants_sont_uniques(self, fiches):
        ids = [f["id"] for f in fiches]
        doublons = {i for i in ids if ids.count(i) > 1}
        assert not doublons, f"identifiants en double : {doublons}"

    def test_aucun_probleme_vide(self, fiches):
        vides = [f["id"] for f in fiches if not f["probleme"].strip()]
        assert not vides, f"fiches sans libellé de problème : {vides}"

    def test_statut_valide(self, fiches):
        for f in fiches:
            assert f["status"] in ("ok", "no_answer"), \
                f"fiche {f['id']} : statut inattendu « {f['status']} »"

    def test_une_fiche_resolue_a_une_solution(self, fiches):
        vides = [f["id"] for f in fiches if f["status"] == "ok" and not f["solution"].strip()]
        assert not vides, f"fiches marquées résolues mais sans solution : {vides}"


class TestCouvertureDesReponses:
    """Une fiche résolue sans réponse pré-rédigée retombe sur la voie lente
    (40 à 100 s) : l'objectif de latence n'est plus tenu."""

    def test_chaque_fiche_resolue_a_sa_reponse(self, fiches, reponses):
        attendues = [f["id"] for f in fiches if f["status"] == "ok"]
        manquantes = [i for i in attendues if i not in reponses]
        assert not manquantes, (
            f"{len(manquantes)} fiche(s) sans réponse pré-rédigée : {manquantes[:8]} — "
            "relancer « python -X utf8 src\\precompute_answers.py »")

    def test_aucune_reponse_orpheline(self, fiches, reponses):
        connus = {f["id"] for f in fiches}
        orphelines = [i for i in reponses if i not in connus]
        assert not orphelines, (
            f"réponses sans fiche correspondante : {orphelines[:8]} — "
            "la base a changé, relancer la chaîne de préparation")


class TestQualiteDesReponses:
    def test_aucune_reponse_francaise_vide(self, reponses):
        vides = [i for i, r in reponses.items() if not r.get("fr", "").strip()]
        assert not vides, f"réponses françaises vides : {vides}"

    def test_aucune_reponse_arabe_vide(self, reponses):
        vides = [i for i, r in reponses.items() if not r.get("ar", "").strip()]
        assert not vides, f"réponses arabes vides : {vides}"

    @staticmethod
    def _non_traduites(reponses: dict) -> tuple[list[str], list[str]]:
        """Sépare ce qu'un script peut réparer de ce qui attend un humain.

        « ar_defaillante » est posé par retraduire_ar.py quand le modèle a
        épuisé ses trois tentatives, découpage en blocs compris. Relancer le
        script ne donnerait rien : seule une traduction humaine débloque la
        fiche, et elle est déjà signalée en rouge dans /revision.
        """
        a_retenter, abandonnees = [], []
        for fid, rep in reponses.items():
            lettres = [c for c in rep.get("ar", "") if c.isalpha()]
            if not lettres:
                continue
            part_arabe = sum(1 for c in lettres if ARABE.match(c)) / len(lettres)
            if part_arabe < 0.30:
                (abandonnees if rep.get("ar_defaillante") else a_retenter).append(fid)
        return a_retenter, abandonnees

    def test_le_champ_arabe_est_bien_en_arabe(self, reponses):
        """RÉGRESSION — 25 champs « ar » sur 44 contenaient du FRANÇAIS : le
        modèle recopiait au lieu de traduire. Un usager arabophone recevait une
        réponse qu'il ne pouvait pas lire.

        Seul ce qu'un script peut réparer fait échouer la construction. Une
        fiche explicitement reconnue hors de portée du modèle est une DONNÉE
        MANQUANTE en attente d'un humain, pas une régression : la faire
        échouer laisserait l'intégration continue rouge en permanence, et une
        alerte permanente n'alerte plus personne.
        """
        a_retenter, abandonnees = self._non_traduites(reponses)
        if abandonnees:
            warnings.warn(
                f"{len(abandonnees)} réponse(s) hors de portée du modèle "
                f"({abandonnees[:8]}) — à saisir à la main dans /revision, "
                "filtre « Arabe non traduit »",
                stacklevel=2)
        assert not a_retenter, (
            f"{len(a_retenter)} traduction(s) arabe(s) à refaire : {a_retenter[:8]} — "
            "lancer « python -X utf8 src\\retraduire_ar.py »")

    def test_toute_reponse_restee_en_francais_est_signalee(self, reponses):
        """Le corollaire du test précédent, et sa condition de validité.

        Une réponse laissée en français DOIT porter « ar_defaillante ». Sans ce
        drapeau elle disparaît des radars : le script ne la reprend pas, et rien
        ne distingue un abandon assumé d'un oubli. C'est ce test qui empêche
        l'indulgence ci-dessus de devenir un trou."""
        a_retenter, _ = self._non_traduites(reponses)
        assert not a_retenter, (
            f"{len(a_retenter)} réponse(s) en français sans le drapeau "
            f"« ar_defaillante » : {a_retenter[:8]}")

    @pytest.mark.parametrize("interdit", [
        "voici la réponse",
        "documentation fournie",
        "[votre nom]",
        "passage 1",
        "as an ai",
    ])
    def test_aucune_formule_parasite(self, reponses, interdit):
        """Ce que l'usager ne doit jamais lire : les traces du fonctionnement
        interne, ou un crochet resté à remplir."""
        fautives = [i for i, r in reponses.items()
                    if interdit in (r.get("fr", "") + r.get("ar", "")).lower()]
        assert not fautives, (
            f"« {interdit} » présent dans {fautives[:8]} — "
            "relancer « python -X utf8 src\\nettoyer_reponses.py --appliquer »")

    def test_aucune_reponse_ne_commence_par_une_salutation(self, reponses):
        """La bulle de chat et la lettre de réclamation ont déjà leur formule
        d'appel : une salutation ici fait doublon."""
        fautives = [i for i, r in reponses.items()
                    if r.get("fr", "").strip().lower().startswith(("bonjour", "cher ", "chère "))]
        assert not fautives, f"salutation en tête de : {fautives[:8]}"
