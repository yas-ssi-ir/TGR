# Assistant IA — Portail eServices TGR

**Agentic RAG** (pattern *Corrective RAG*) pour l'assistant conversationnel du portail
eServices de la Trésorerie Générale du Royaume du Maroc.
100 % local, 100 % open source, **sans GPU**, **sans appel réseau à l'exécution**.

Répond aux questions des usagers **et** traite les réclamations déposées —
en **français**, en **arabe** et en **darija**.

---

## Sommaire

- [Le problème](#le-problème)
- [Le parti pris : le LLM ne rédige jamais en direct](#le-parti-pris--le-llm-ne-rédige-jamais-en-direct)
- [Architecture](#architecture)
- [Comment l'assistant décide, sans LLM](#comment-lassistant-décide-sans-llm)
- [Le multilingue](#le-multilingue--français-arabe-darija)
- [Stack technique](#stack-technique)
- [Installation](#installation)
- [Chaîne de préparation des données](#chaîne-de-préparation-des-données)
- [Lancement](#lancement)
- [Espace agent : la relecture n'est pas optionnelle](#espace-agent--la-relecture-nest-pas-optionnelle)
- [Module réclamations](#module-réclamations)
- [Vérification et tests](#vérification-et-tests)
- [Intégration continue](#intégration-continue)
- [Déploiement](#déploiement)
- [Résultats mesurés](#résultats-mesurés)
- [Structure du projet](#structure-du-projet)
- [Les seuils, et pourquoi ces valeurs](#les-seuils-et-pourquoi-ces-valeurs)
- [Limites connues](#limites-connues)

---

## Le problème

Un assistant d'administration publique a trois contraintes que la plupart des
démonstrations de RAG ignorent :

1. **Il ne doit jamais inventer.** Une procédure administrative inventée envoie
   un usager faire la queue pour rien. Mieux vaut un « je ne sais pas » honnête.
2. **Il doit répondre vite, sur du matériel ordinaire.** Pas de GPU, pas de
   cloud : un CPU de bureau. Or un modèle 3B y écrit à ~9 tokens/seconde, soit
   40 à 100 secondes par réponse.
3. **Il doit parler la langue des usagers.** Au Maroc, cela veut dire le
   français, l'arabe, et la darija — souvent écrite en alphabet latin
   (« bghit nmse7 compte dyali »).

Ce projet traite les trois, et la troisième contrainte s'avère être le vrai
sujet technique : les seuils de distance d'un modèle d'embedding multilingue
ne discriminent pas ce qu'on croit.

---

## Le parti pris : le LLM ne rédige jamais en direct

Aucun réglage ne change la physique du CPU. Le système **retire donc le LLM du
chemin de la réponse** pour tout ce qui est déjà connu :

| Situation | Qui répond | Latence |
|---|---|---|
| Question connue | Réponse officielle **pré-rédigée hors ligne** | **~0,1 s** |
| Question déjà posée une fois | Cache sémantique | ~0,3 s |
| Question hors sujet | Cascade de filtres déterministes | ~0,1 s |
| Question inédite | LLM en streaming, puis mise en cache | 30–60 s, une seule fois |

Le LLM travaille **hors ligne**, une fois : il rédige la réponse officielle de
chaque fiche, en français et en arabe. À l'exécution, ces textes sont servis
verbatim. Plus rapide, moins cher, et **sans risque d'invention** — à condition
d'avoir été relus par un agent (voir [Espace agent](#espace-agent--la-relecture-nest-pas-optionnelle)).

---

## Architecture

```
                          question de l'usager
                                   │
                    ┌──────────────▼──────────────┐
                    │  0. CACHE SÉMANTIQUE        │  similarité ≥ 0,96
                    │     déjà répondu ?          │──── oui ──▶ ~0,3 s
                    └──────────────┬──────────────┘
                                   │ non
                    ┌──────────────▼──────────────┐
                    │  1. RETRIEVE                │  top-25 ChromaDB
                    │     + vote des voisins      │  (le consensus sert
                    └──────────────┬──────────────┘   de preuve au filtre)
                                   │
                    ┌──────────────▼──────────────┐
                    │  2. GUARDRAIL               │  4 signaux déterministes
                    │     dans le périmètre TGR ? │──── non ──▶ refus poli, ~0,1 s
                    └──────────────┬──────────────┘
                                   │ oui
                    ┌──────────────▼──────────────┐
                    │  3. VOIE RAPIDE             │  réponse officielle
                    │     fiche pré-rédigée ?     │──── oui ──▶ ~0,1 s, zéro LLM
                    └──────────────┬──────────────┘
                                   │ non → voie lente
                    ┌──────────────▼──────────────┐
                    │  4. GRADE (LLM, 1 appel)    │  pertinence des passages
                    └──────────────┬──────────────┘
                       ┌───────────┴───────────┐
                  0 pertinent              ≥1 pertinent
                       │                       │
            ┌──────────▼─────────┐  ┌──────────▼─────────┐
            │ 5. REWRITE (×1)    │  │ 6. GENERATE        │  streaming SSE
            │    puis RETRIEVE   │  └──────────┬─────────┘
            └──────────┬─────────┘             │
                  toujours rien      ┌──────────▼─────────┐
                       │             │ 7. VERIFY          │  anti-hallucination
                       ▼             └──────────┬─────────┘
              FALLBACK honnête                  ▼
           (orientation support)        réponse + sources
```

Chaque étape est **tracée et affichée en direct** dans l'interface : l'usager
voit l'agent raisonner, et un auditeur voit pourquoi telle réponse a été servie.

---

## Comment l'assistant décide, sans LLM

**Le constat qui structure tout le projet.** `multilingual-e5` écrase toutes les
distances entre 0,22 et 0,45, qu'une question soit pertinente ou non :

| Question | Distance | Verdict attendu |
|---|---|---|
| `كيف أحذف حسابي؟` (« comment supprimer mon compte ? ») | 0,344 | **dans** le périmètre |
| `donne-moi une recette de tajine` | 0,356 | **hors** périmètre |

Douze millièmes séparent une question légitime d'une question de cuisine.
**Un seuil absolu ne peut pas trancher.** Le système s'appuie donc sur quatre
signaux déterministes (0 ms chacun), du plus sûr au plus faible :

### 1. Zéro mot commun avec la documentation → hors sujet certain

Le vocabulaire du domaine est **construit automatiquement** depuis le corpus
indexé (`src/lexique.py`) — aucune liste écrite à la main, donc rien à
maintenir quand la base évolue. Une question dont aucun mot significatif
n'appartient à ce vocabulaire ne peut pas concerner le portail.

> **Piège rencontré en production.** Certains mots sont à la fois du vocabulaire
> métier et des mots de tous les jours : *recette* (recette de perception) et
> *gestion*. Ils validaient le test d'appartenance au domaine, laissant passer
> des questions hors sujet vers l'étape suivante — qui, elle, ne raisonne pas
> sur le sens. Ces mots sont désormais neutralisés dans les deux sens.

### 2. Consensus des voisins → le vrai discriminant

Pour une vraie question, plusieurs chunks d'une **même fiche** (la fiche et ses
variantes de formulation) remontent ensemble dans le top-25 : c'est la signature
d'une correspondance réelle. Pour une question hors sujet, les résultats sont
dispersés entre fiches sans rapport.

La décision se prend en deux temps, **et l'ordre compte** :

- **Recevabilité** — une fiche n'est candidate que si ≥ 2 de ses chunks
  concordent, ou si un seul remonte mais très proche (≤ 0,30).
- **Choix** — parmi les recevables, **la plus proche** l'emporte. Trancher au
  nombre de voix favoriserait mécaniquement les fiches les mieux dotées en
  variantes, au détriment d'une fiche plus proche mais moins fournie.

Deux garde-fous complètent le vote :

- **Anti-antonymes** — les embeddings sont aveugles à l'opposition de sens :
  « créer un compte » et « supprimer un compte » sont vectoriellement voisins.
  Une fiche décrivant *exclusivement* l'action inverse de celle demandée est
  écartée, si proche soit-elle.
- **Départage bug / solution** — une fiche « problème connu sans solution » ne
  doit pas l'emporter de justesse sur une fiche qui, elle, porte un correctif.

### 3. Terme métier non ambigu employé → dans le périmètre

Liste courte et explicite (`TERMES_NOYAU`) : *compte*, *mot de passe*, *CNIE*,
*quittance*, *adhésion*, *soumission*… Les mots à double sens en sont exclus
à dessein.

### 4. Ni terme métier, ni source proche → hors sujet

Le LLM n'arbitre **que** ce qui échappe aux quatre signaux.

---

## Le multilingue : français, arabe, darija

La documentation est en français. Une question posée dans une autre écriture est
structurellement plus « loin » dans l'espace vectoriel — non parce qu'elle est
hors sujet, mais parce que l'embedding franchit une barrière linguistique. Deux
marges distinctes, calibrées séparément :

| Écriture | Marge | Mesure qui la justifie |
|---|---|---|
| Arabe | +0,07 | une question légitime place 5 de ses 6 voisins sur la bonne fiche, mais à 0,40–0,41 |
| Darija latine | +0,03 | les 14 questions darija du corpus sont toutes à ≤ 0,351 — une marge de 0,07 serait du gaspillage de tolérance |

La marge darija a été calibrée **sur la décision finale** (question légitime
acceptée / hors-sujet refusé), pas sur la seule distance :

| Marge | Darija légitimes acceptées | Refusées à tort | Hors-sujet admis |
|---|---|---|---|
| 0,00 | 12/14 | 1 | 0/6 |
| **0,03** | **13/14** | **0** | **2/6** ← retenu |
| 0,07 | 14/14 | 0 | 3/6 |

0,03 supprime le refus à tort — **le pire des défauts : un usager marocain
éconduit par sa propre administration** — en exposant moins que 0,07.

La darija en alphabet latin est détectée par trois signaux volontairement
stricts, pour ne jamais attraper du français : vocabulaire fermé (*bghit*,
*kifach*, *dyali*…), motif de négation `ma…ch` (*mabqitch*), et chiffre employé
comme lettre (*n9der*, *3andi*).

---

## Stack technique

| Brique | Choix | Pourquoi |
|---|---|---|
| Embeddings | `intfloat/multilingual-e5-base` (CPU) | un seul modèle couvre FR / AR / darija ; préfixes `query:` / `passage:` obligatoires |
| Base vectorielle | ChromaDB, persistée sur disque | aucun service externe, aucune donnée usager qui sort |
| LLM | Ollama + `qwen2.5:3b-instruct` | bon en FR/AR, ~2 Go, tourne sur CPU |
| Orchestration | LangChain + graphe CRAG maison | le graphe est explicite et traçable, pas une boîte noire |
| API | FastAPI + streaming SSE | l'usager voit l'agent raisonner en direct |
| Interface | HTML/CSS/JS + Tailwind v4 **embarqué localement** | un CDN signalerait chaque visite d'usager à un tiers |

Aucune dépendance réseau à l'exécution : une fois les modèles téléchargés, le
système fonctionne hors ligne.

---

## Installation

```powershell
# 1. Dépendances Python (une fois)
py -3.11 -m venv venv
venv\Scripts\python.exe -m pip install -r requirements.txt

# 2. LLM local (une fois) — https://ollama.com/download
ollama pull qwen2.5:3b-instruct

# 3. Ressources de l'interface (une fois, code tiers non versionné)
python -X utf8 eval\installer_tailwind.py
python -X utf8 eval\installer_polices.py
```

---

## Chaîne de préparation des données

À rejouer dans cet ordre après toute modification des sources :

```powershell
python -X utf8 src\prepare_qa.py           # xlsx des réclamations → fiches + variantes
python -X utf8 src\prepare_faq.py          # FAQ PDF → fiches Q/R
python -X utf8 src\prepare_assistant.py    # relevé .docx de l'assistant → fiches
python -X utf8 src\ingestion.py            # tout → ChromaDB                    (~2 min)
python -X utf8 src\precompute_answers.py   # réponses officielles FR + AR       (~30 min)
python -X utf8 src\nettoyer_reponses.py --appliquer   # retire préambules et salutations
python -X utf8 src\retraduire_ar.py        # retraduit les champs arabes défaillants
```

`precompute_answers.py` et `retraduire_ar.py` sont **interruptibles** : ils
sauvegardent après chaque fiche et reprennent où ils se sont arrêtés.

**Pourquoi une étape de nettoyage.** Un modèle 3B n'obéit pas parfaitement à une
consigne de format : il ajoute des préambules (« Voici la réponse officielle : »),
des salutations, des signatures. Plutôt que de le relancer en espérant mieux, on
corrige mécaniquement — c'est déterministe, gratuit et reproductible.

---

## Lancement

`/revision` exige un mot de passe (voir plus bas) — le définir avant de démarrer :

```powershell
$env:REVISION_PASSWORD = "choisissez-un-mot-de-passe"
python -X utf8 src\main.py
```

| URL | Rôle |
|---|---|
| http://127.0.0.1:8000 | assistant (onglets *Assistant* et *Réclamation*) |
| http://127.0.0.1:8000/revision | **espace agent** — relecture, validation, questions en attente |
| http://127.0.0.1:8000/health | état ChromaDB + Ollama |
| http://127.0.0.1:8000/docs | documentation OpenAPI |

Avant une démonstration, préchauffer le cache :

```powershell
python -X utf8 src\warmup_cache.py
```

---

## Espace agent : la relecture n'est pas optionnelle

Une réponse pré-rédigée est servie **sans vérification au moment de la question**.
Tant qu'un agent ne l'a pas relue, c'est le texte d'un modèle 3B qui engage
l'administration.

L'espace agent met chaque réponse **en regard de sa note source**, et gère trois
états :

| État | Signification |
|---|---|
| **à relire** | personne n'a encore comparé la réponse à sa source |
| **relue** | un agent a comparé, il n'y avait rien à reprendre |
| **validée** | un agent **nommé** engage l'administration sur ce texte |

Trois règles de conception, chacune née d'un défaut réel :

- **Toute modification du texte retire la validation ET la relecture.** Une fiche
  changée doit être re-signée : sinon un texte modifié continue de s'afficher
  comme certifié.
- **La signature est nominative.** Une validation anonyme n'engage personne.
- **La dévalidation existe.** Un relecteur se trompe de bouton ; sans marche
  arrière, la seule issue serait de retoucher le fichier à la main.

L'interface indique aussi à l'usager, dans la réponse elle-même, si le texte
qu'il lit a été validé par un agent ou s'il n'a encore été relu par personne.
C'est toute la différence entre une réponse **officielle** et une réponse
seulement **vraisemblable**.

**Questions non traitées.** Quand une question relève bien du portail mais que
le système ne peut pas y répondre correctement (fiche connue sans solution, ou
réponse générée qui échoue la vérification d'ancrage), elle est conservée dans
`data/processed/questions_en_attente.json` et listée dans une section dédiée de
`/revision` — plutôt que perdue silencieusement. L'usager en est informé dans
la réponse elle-même : *« Votre question a été transmise à nos équipes pour
traitement. »*

---

## Module réclamations

Calqué sur le portail réel (onglets *Dépôt* et *Suivi*) :

```
dépôt d'une réclamation
   │
[1. CLASSIFIER]  nature détectée automatiquement
   │
[2. CHERCHER]    problème déjà connu ? (consensus vectoriel, sans LLM)
   │
   ├─ solution connue  → réponse automatique proposée, sous réserve de
   │                     validation par un agent
   ├─ problème connu   → escalade technique
   └─ inconnu          → escalade agent
   │
[3. ENREGISTRER] référence de suivi REC-AAAA-NNNNN
```

**Aucune réponse n'est jamais finale sans supervision humaine.** Et la lettre
envoyée ne promet que ce que le portail sait réellement faire : suivre un
dossier, ou en déposer un nouveau en rappelant la référence.

---

## Vérification et tests

```powershell
python -X utf8 src\verifier_tout.py       # verdict global PASSE / ÉCHOUE   (~3 min)
python -X utf8 src\audit_couverture.py    # la voie rapide trouve-t-elle la bonne fiche ?
python -X utf8 src\banc_essai.py          # banc d'essai complet du corpus
python -X utf8 src\bench_latence.py       # latence + justesse, 18 questions réelles
python -X utf8 src\diagnostic.py "ma question"   # trace pas à pas d'une question
```

`verifier_tout.py` enchaîne quatre contrôles et rend un verdict unique :
intégrité des données, couverture de la voie rapide, latence + justesse, et
**avancement de la relecture humaine**.

### Suite de tests

```powershell
venv\Scripts\python.exe -m pip install -r requirements-dev.txt

venv\Scripts\python.exe -m ruff check src tests    # analyse statique
venv\Scripts\python.exe -m pytest                  # toute la suite
venv\Scripts\python.exe -m pytest -m "not lourd"   # logique + données (~2 s)
```

Les tests couvrent **la logique de décision**, là où une régression serait
invisible : filtre lexical, vote de consensus, nettoyage des réponses, et
intégrité des données servies (aucune réponse vide, aucun champ arabe resté en
français, aucun crochet `[Votre Nom]` oublié).

Plusieurs sont des **tests de non-régression** : ils reproduisent des défauts
réellement rencontrés — l'arabe rejeté par le filtre lexical, la fiche la mieux
dotée en variantes qui l'emportait sur la plus proche, la signature à trous
servie à l'usager, et le mot à double sens qui validait le périmètre.

> **Note Windows.** Les commandes passent par `python -m` à dessein : la
> stratégie de contrôle d'application bloque les exécutables fraîchement
> téléchargés, si bien que `venv\Scripts\pytest.exe` échoue là où
> `python -m pytest` fonctionne.

---

## Intégration continue

`.github/workflows/ci.yml`, en trois étages — le premier doit échouer vite :

| Étage | Contenu | Durée |
|---|---|---|
| Analyse statique et logique | ruff + tests sans la pile ML | ~40 s |
| Moteur de décision | tests du consensus + couverture | ~4 min |
| Image de déploiement | l'image Docker se construit et démarre | ~2 min |

---

## Déploiement

Sur un serveur, sans rien installer d'autre que Docker :

```bash
docker compose up -d --build
docker compose exec ollama ollama pull qwen2.5:3b-instruct   # une seule fois
```

→ http://localhost:8000

Deux services séparés : `ollama` (le modèle) et `assistant` (l'application).
Les séparer permet de redéployer l'assistant **sans retélécharger les 2 Go du
modèle**. Le dossier `data/` est monté en volume : la base vectorielle et
surtout **les réponses relues par les agents** survivent aux redéploiements.

Tout reste sur la machine : aucune donnée d'usager ne sort du réseau.

---

## Résultats mesurés

**Corpus indexé** — 605 chunks issus de 139 fiches (4 sources) et
**125 réponses officielles** pré-rédigées en français et en arabe.

**Banc d'essai du corpus complet** (`src/banc_essai.py`) :

```
Ensemble du corpus  381 / 395
Questions en arabe   67 / 67
Questions en darija  28 / 28
```

**Latence** — voie rapide ~0,1 s, cache ~0,3 s, refus hors périmètre ~0,1 s.
La voie lente (question inédite) reste à 30–60 s sur CPU, une seule fois : la
réponse est ensuite mise en cache.

**Gain le plus net du projet** : en retirant la solution complète des chunks de
variantes, une question en darija est passée de **124 s à 0,24 s** — elle ne
retrouvait plus sa propre fiche, noyée dans le français environnant.

Pour régénérer ces chiffres :

```powershell
python -X utf8 src\banc_essai.py
python -X utf8 eval\evaluate.py     # comparaison RAG classique vs agentic
```

---

## Structure du projet

```
data/raw/                             sources documentaires (PDF, DOCX, XLSX)
data/processed/qa_fiches.json         fiches issues de l'historique des réclamations
data/processed/faq_fiches.json        fiches issues de la FAQ officielle
data/processed/assistant_fiches.json  fiches issues du relevé de l'assistant existant
data/processed/reponses_precalculees.json   réponses officielles FR + AR
data/processed/questions_en_attente.json   questions non traitées (non versionné, se remplit à l'usage)
data/chroma_db/                       base vectorielle (non versionnée, régénérable)

src/config.py               toutes les constantes et seuils, chacun commenté
                            avec la mesure qui l'a produit
src/lexique.py              vocabulaire du domaine, marges de langue,
                            garde-fou anti-antonymes
src/prepare_qa.py           xlsx → fiches + variantes usager (FR / darija)
src/prepare_faq.py          FAQ PDF → fiches Q/R
src/prepare_assistant.py    relevé .docx → fiches (découpage par styles Word)
src/ingestion.py            fiches + documents → ChromaDB
src/retriever.py            recherche sémantique (préfixes E5)
src/llm.py                  client Ollama (generate / stream / decide)
src/agent_rag.py            ★ l'agent : consensus, guardrail, voie rapide, CRAG
src/rag_classic.py          baseline, pour la comparaison chiffrée
src/reclamation_handler.py  dépôt / suivi / validation des réclamations
src/semantic_cache.py       cache sémantique + invalidation par empreinte
src/precompute_answers.py   pré-rédaction hors ligne des réponses officielles
src/nettoyer_reponses.py    retire préambules, salutations, signatures
src/retraduire_ar.py        traduction arabe vérifiée, avec réessai
src/revision.py             relecture humaine (données + validation)
src/main.py                 API FastAPI + streaming SSE
src/verifier_tout.py        vérification complète, verdict PASSE / ÉCHOUE
src/audit_couverture.py     audit de la voie rapide (sans LLM)
src/banc_essai.py           banc d'essai du corpus complet
src/bench_latence.py        latence + justesse sur questions réelles
src/diagnostic.py           trace pas à pas d'une question
src/warmup_cache.py         préchauffage du cache avant une démonstration

static/index.html           interface assistant + réclamations
static/revision.html        espace agent (relecture et validation)
eval/                       golden dataset, évaluation comparative, installeurs
eval/mesure_consensus_multichunk.py   outil de mesure pour recalibrer DIST_CONSENSUS_MULTI_MAX
tests/                      suite de tests (logique, nettoyage, données)

pyproject.toml              configuration ruff + pytest
.github/workflows/ci.yml    intégration continue
Dockerfile / docker-compose.yml       déploiement (assistant + ollama)
```

---

## Les seuils, et pourquoi ces valeurs

Aucun seuil n'a été choisi au jugé : chacun est commenté dans `src/config.py`
avec la mesure qui l'a produit.

| Constante | Valeur | Rôle |
|---|---|---|
| `DIST_AUTO_IN` | 0,33 | ≤ → passage pertinent sans demander au LLM |
| `DIST_AUTO_OUT` | 0,45 | ≥ → hors périmètre sans demander au LLM |
| `DIST_HORS_SUJET` | 0,34 | seuil du refus déterministe |
| `CONSENSUS_K` | 25 | profondeur de recherche pour le vote des voisins |
| `CONSENSUS_MIN_CHUNKS` | 2 | chunks concordants pour emporter la décision |
| `DIST_CANDIDATE_MAX` | 0,37 | au-delà, un chunk ne vote plus |
| `DIST_CONSENSUS_MULTI_MAX` | 0,31 | en français, ≥ 2 chunks ne suffisent plus au-delà — sinon le LLM n'est presque jamais sollicité sur une question inédite (exempté en arabe/darija) |
| `DIST_SOLO_ACCEPT` | 0,30 | un chunk unique n'est accepté que très proche |
| `DIST_DEPARTAGE` | 0,04 | départage « bug connu » vs fiche porteuse de solution |
| `DIST_OFFSET_TRANSLANGUE` | 0,07 | marge accordée aux questions en arabe |
| `DIST_OFFSET_DARIJA` | 0,03 | marge accordée à la darija en alphabet latin |
| `CACHE_SIM_MIN` | 0,96 | similarité minimale pour servir depuis le cache |
| `CHUNK_SOLUTION_MAX` | 0 | solution recopiée dans un chunk de variante — aucune |

---

## Limites connues

- **La relecture humaine reste à faire.** Tant que le compteur de l'espace agent
  n'est pas complet, le contenu servi engage un modèle 3B, pas l'administration.
  C'est la limite la plus importante du projet, et elle est volontairement
  visible dans l'interface.
- **Qualité de l'arabe** : traduction automatique d'un petit modèle, vérifiée
  mécaniquement (l'écriture est bien arabe) mais pas relue par un arabophone.
- **Quelques fiches sans solution documentée** : la procédure n'existe dans
  aucune source. L'assistant oriente alors vers le dépôt d'une réclamation —
  comportement correct : mieux vaut un aiguillage honnête qu'une procédure
  inventée.
- **Compromis darija assumé** : la marge de 0,03 laisse passer certaines
  questions hors sujet en darija. Le refus à tort d'une vraie question a été
  jugé plus grave, mesures à l'appui.
- **Limite du modèle d'embedding** : sur certaines formulations, un mot saillant
  pèse plus que le reste de la phrase et place une fiche voisine devant la
  bonne. Ce n'est ni un défaut des données ni des seuils. Les deux vraies
  réponses seraient un modèle plus fort (`e5-large`) ou un **reranker**
  cross-encodeur — l'un et l'autre coûtent de la latence et demandent une
  revalidation complète : c'est une v2, pas un correctif.
- **Un seul utilisateur à la fois** sur les questions inédites : Ollama sérialise
  les requêtes. Sans effet sur la voie rapide, qui n'appelle pas le LLM.
- **`DIST_CONSENSUS_MULTI_MAX` a un coût mesuré** : environ 0,8 % des questions qui
  trouvaient leur fiche instantanément passent désormais par le LLM (jusqu'à
  30 s). Compromis assumé pour qu'une question réellement inédite reçoive un
  vrai jugement plutôt qu'une fausse réponse rapide — voir `config.py`.
- **Réclamations stockées en JSON**, sans authentification : maquette
  fonctionnelle, en attente d'un accès base de données (`src/ingestion_postgres.py`).
- **Un PDF source est scanné** : ses pages ne sont pas exploitées, faute d'OCR.

---

## Contexte

Projet réalisé dans le cadre d'un stage à la Trésorerie Générale du Royaume.
L'assistant fonctionne intégralement en local : aucune donnée d'usager n'est
transmise à un service tiers, à aucun moment.
