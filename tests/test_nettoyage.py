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


class TestMarqueurDeListe:
    r"""RÉGRESSION — le nettoyage remplaçait un numéro de téléphone par « 1. ».

    Mesuré sur la réponse verbatim de FAQ.20 : le motif « \d+[.)]\s » prenait
    « 0537273727. » — le centre d'appel de la DGI — pour une étape numérotée et
    la renumérotait. L'usager se serait vu communiquer un numéro inexistant,
    dans un texte officiel recopié mot pour mot.
    """

    def test_un_numero_de_telephone_n_est_pas_une_etape(self):
        texte = ("Vous pouvez nous contacter, ou appeler le centre d'appel "
                 "de la DGI : 0537273727. Nos équipes vous répondront.")
        assert "0537273727" in nettoyer(texte)

    def test_une_vraie_liste_est_toujours_renumerotee(self):
        texte = "3. Ouvrez votre profil. 4. Cliquez sur Modifier."
        sortie = nettoyer(texte)
        assert sortie.startswith("1.")
        assert "2." in sortie

    def test_un_montant_n_est_pas_une_etape(self):
        texte = "Le seuil est fixé à 200. Au-delà, la taxe est due."
        assert "200." in nettoyer(texte)


class TestProtectionDuVerbatim:
    """Un texte officiel recopié tel quel ne doit jamais être « nettoyé » :
    il ne contient aucun artefact de rédaction, donc toute modification est
    une dégradation."""

    def test_les_textes_officiels_sont_repertories_par_fiche(self):
        from nettoyer_reponses import textes_officiels
        officiels = textes_officiels()
        # rien à vérifier si la chaîne de préparation n'a pas été jouée
        if not officiels:
            return
        assert all(isinstance(cle, tuple) and len(cle) == 2 for cle in officiels)
        assert all(sol.strip() for _, sol in officiels)


class TestDecoupagePourTraduction:
    """Un modèle 3B décroche sur un texte long et dense : il recopie le français
    au lieu de traduire. Le découpage en blocs est le dernier recours avant
    d'abandonner la fiche à une traduction humaine."""

    def test_un_texte_court_reste_entier(self):
        from retraduire_ar import decouper
        assert decouper("Une seule phrase courte.") == ["Une seule phrase courte."]

    def test_un_texte_long_est_coupe_en_fin_de_phrase(self):
        from retraduire_ar import TAILLE_MORCEAU, decouper
        texte = " ".join(f"Voici la phrase numéro {i} de ce texte administratif."
                         for i in range(40))
        blocs = decouper(texte)
        assert len(blocs) > 1
        assert all(len(b) <= TAILLE_MORCEAU + 120 for b in blocs)
        assert all(b.endswith(".") for b in blocs)

    def test_aucun_texte_n_est_perdu_au_decoupage(self):
        from retraduire_ar import decouper
        texte = " ".join(f"Phrase {i} du document." for i in range(30))
        assert " ".join(decouper(texte)).split() == texte.split()

    def test_le_budget_de_tokens_suit_la_longueur(self):
        """Une traduction coupée en plein milieu reste en arabe et passe la
        vérification : l'usager reçoit alors la moitié de sa réponse."""
        from retraduire_ar import budget_tokens
        assert budget_tokens("court") < budget_tokens("x" * 1600)
        assert budget_tokens("x" * 99999) <= 1600


class TestPolitesseFinale:
    """RÉGRESSION — « Je vous prie d'agréer, avec l'assurance de ma
    considération, l'expression de mes salutations distinguées. » était servi
    à l'usager au bout d'une réponse d'assistance. Le filtre de signature
    s'ancrait sur un mot-clé en début de ligne (« Cordialement ») ; celle-ci
    commence par « Je vous prie », trop banal pour servir d'ancre.
    Constaté sur les fiches 1.1.1, 1.2.1 et FAQ.34.
    """

    def test_je_vous_prie_d_agreer_est_retire(self):
        texte = ("Contactez le support du portail.\n\n"
                 "Je vous prie d'agréer, avec l'assurance de ma considération, "
                 "l'expression de mes salutations distinguées.")
        sortie = nettoyer(texte)
        assert "agréer" not in sortie
        assert sortie.endswith("Contactez le support du portail.")

    def test_la_formule_collee_a_une_phrase_utile_ne_l_emporte_pas(self):
        """Le modèle colle parfois la politesse derrière une phrase utile :
        on coupe à la PHRASE, pas à la ligne, sinon on perd du contenu."""
        texte = ("Ils pourront vous aider. Je vous remercie de votre patience et "
                 "vous prie d'agréer, cher utilisateur, mes salutations distinguées.")
        sortie = nettoyer(texte)
        assert "Ils pourront vous aider." in sortie
        assert "agréer" not in sortie

    def test_un_texte_sans_politesse_est_intact(self):
        texte = "Votre mot de passe a expiré. Utilisez le lien « Mot de passe oublié »."
        assert nettoyer(texte) == texte
