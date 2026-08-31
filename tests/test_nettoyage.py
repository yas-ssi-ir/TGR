"""
Tests du nettoyage des réponses pré-rédigées.

Le modèle 3B ajoute des préambules et des salutations malgré les consignes.
Ces textes étant servis tels quels à l'usager, le nettoyage n'est pas
cosmétique : il conditionne la crédibilité de la réponse.
"""
from nettoyer_reponses import nettoyer


class TestPreambules:
    def test_retire_l_annonce_du_modele(self):
        # cas réel observé, faute de frappe du modèle comprise
        texte = "Voici la réponse officiée en français :\n\nVotre mot de passe a expiré."
        assert nettoyer(texte) == "Votre mot de passe a expiré."

    def test_retire_un_preambule_reformule(self):
        texte = "Voici la réponse claire et concise à votre problème :\nContactez votre perception."
        assert "Voici" not in nettoyer(texte)


class TestSalutations:
    def test_retire_la_salutation_isolee(self):
        assert nettoyer("Bonjour,\n\nVotre demande a échoué.") == "Votre demande a échoué."

    def test_retire_cher_utilisateur(self):
        assert nettoyer("Cher utilisateur,\n\nLe lien a expiré.") == "Le lien a expiré."

    def test_empile_preambule_et_salutation(self):
        """Retirer le premier fait apparaître le second : le nettoyage boucle."""
        texte = "Cher utilisateur,\n\nVoici la réponse :\n\nLe compte est bloqué."
        assert nettoyer(texte) == "Le compte est bloqué."

    def test_retire_la_salutation_dans_une_etape_numerotee(self):
        texte = ("1. Bonjour, je comprends que vous avez changé de téléphone. "
                 "2. Récupérez le code depuis votre compte Google.")
        resultat = nettoyer(texte)
        assert "Bonjour" not in resultat
        assert "Récupérez le code" in resultat


class TestRenumerotation:
    def test_la_liste_repart_a_un(self):
        """Une étape supprimée ne doit pas laisser « 1. 2. Texte »."""
        texte = "1. Bonjour, je vous remercie de votre message. 2. Contactez le support."
        resultat = nettoyer(texte)
        assert not resultat.startswith("1. 2.")
        assert resultat.startswith("1.")


class TestRenvoisInternes:
    def test_retire_la_mention_de_la_documentation(self):
        """L'usager ne voit aucune « documentation fournie » : la citer trahit
        le fonctionnement interne du système."""
        texte = "Selon la documentation fournie, le lien expire au bout d'une heure."
        resultat = nettoyer(texte)
        assert "documentation" not in resultat.lower()
        assert "expire au bout d'une heure" in resultat


class TestSignature:
    def test_retire_la_signature(self):
        """La lettre de réclamation ajoute déjà la sienne : sinon, doublon."""
        texte = "Contactez votre perception.\n\nLe Support eServices TGR"
        assert nettoyer(texte) == "Contactez votre perception."

    def test_retire_la_signature_et_ce_qui_la_suit(self):
        """RÉGRESSION — « Cordialement, » suivi de « [Votre Nom] » échappait au
        filtre : l'usager lisait un crochet à remplir."""
        texte = "Le lien expire au bout d'une heure.\n\nCordialement,\n[Votre Nom]"
        resultat = nettoyer(texte)
        assert resultat == "Le lien expire au bout d'une heure."

    def test_retire_un_crochet_isole(self):
        assert "[" not in nettoyer("Contactez le support. [À compléter]")


class TestRobustesse:
    def test_texte_vide(self):
        assert nettoyer("") == ""
        assert nettoyer(None) == ""

    def test_texte_deja_propre_inchange(self):
        propre = "Votre mot de passe a expiré. Cliquez sur « Mot de passe oublié »."
        assert nettoyer(propre) == propre

    def test_ne_vide_jamais_un_texte_utile(self):
        texte = "Bonjour, je comprends. Contactez votre perception de rattachement."
        assert "perception" in nettoyer(texte)
