# 🏛️ Assistant IA TGR — Agentic RAG

Assistant du portail **eServices TGR** (https://eservices.tgr.gov.ma), en **Agentic RAG**
(pattern Corrective RAG) — 100 % local, 100 % open source, **sans GPU**.

Répond aux questions des usagers **et** traite les réclamations déposées, en français,
en arabe et en darija.

## ⚡ Le parti pris : le LLM ne rédige jamais en direct

Sur CPU, un modèle 3B écrit à ~9 tokens/seconde : une réponse coûte 40 à 100 secondes.
Aucun réglage ne change cette physique. Le système **retire donc le LLM du chemin de la
réponse** pour tout ce qui est déjà connu :

| Situation | Qui répond | Latence mesurée |
|---|---|---|
| Question connue (96 % des cas) | Réponse officielle **pré-rédigée hors ligne** | **~0,1 s** |
| Question déjà posée | Cache sémantique | ~0,3 s |
| Question hors sujet | Cascade de filtres déterministes | ~0,1 s |
| Question inédite | LLM en streaming, puis mise en cache | lente une fois |

Le LLM travaille **hors ligne**, une fois, tranquillement : il rédige les réponses
officielles de chaque fiche (FR + AR). À l'exécution, elles sont servies telles quelles.
Plus rapide, moins cher, **et sans risque d'invention** — à condition d'avoir été relues
(voir « Espace agent » plus bas).

## 🧭 Comment l'assistant décide (sans LLM)

Constat mesuré : `multilingual-e5` écrase toutes les distances entre 0,22 et 0,45.
**Un seuil absolu ne sépare pas le pertinent du hors-sujet** — « كيف أحذف حسابي؟ »
(légitime) est à 0,344, « recette de tajine » (hors sujet) à 0,356.

Le système s'appuie donc sur quatre signaux déterministes, du plus sûr au plus faible :

1. **Zéro mot commun** avec toute la documentation → hors sujet certain
   (lexique construit automatiquement depuis le corpus, `src/lexique.py`)
2. **Consensus des voisins** — plusieurs chunks d'une même fiche remontent ensemble :
   signature d'une vraie correspondance. Parmi les fiches recevables, **la plus proche**
   l'emporte (classer au nombre de voix favoriserait les fiches les mieux dotées en variantes)
3. **Terme métier non ambigu** employé dans la question
4. **Ni terme métier, ni source proche** → hors sujet

Le LLM n'arbitre que ce qui échappe aux quatre.

Les questions écrites dans une langue absente de la documentation (arabe) bénéficient
d'une **marge translangue** : leurs distances sont structurellement plus hautes sans être
pour autant hors sujet.

## 🧱 Stack

| Brique | Choix |
|---|---|
| Embeddings | `intfloat/multilingual-e5-base` (CPU, préfixes `passage:` / `query:`) |
| Base vectorielle | ChromaDB (persistée localement) |
| LLM | Ollama + `qwen2.5:3b-instruct` (CPU) |
| Orchestration | LangChain + graphe CRAG maison |
| API | FastAPI + streaming SSE |
| Interface | HTML/JS vanilla (glassmorphism, RTL arabe) |

## 🚀 Installation

```powershell
# 1. Dépendances (une fois)
py -3.11 -m venv venv
venv\Scripts\python.exe -m pip install -r requirements.txt

# 2. LLM local (une fois) — https://ollama.com/download/windows
ollama pull qwen2.5:3b-instruct
```

## 🔄 Chaîne de préparation

À rejouer dans cet ordre après toute modification des sources :

```powershell
python -X utf8 src\prepare_qa.py          # xlsx des réclamations → fiches + variantes
python -X utf8 src\prepare_faq.py         # FAQ_TGR.pdf → 21 fiches Q/R
python -X utf8 src\ingestion.py           # tout → ChromaDB              (~2 min)
python -X utf8 src\precompute_answers.py  # réponses officielles FR + AR (~30 min, une fois)
python -X utf8 src\nettoyer_reponses.py --appliquer   # retire préambules et salutations
python -X utf8 src\retraduire_ar.py       # retraduit les champs arabes défaillants
```

`precompute_answers.py` et `retraduire_ar.py` sont **interruptibles** : ils sauvegardent
après chaque fiche et reprennent où ils se sont arrêtés.

## ▶️ Lancement

```powershell
python -m uvicorn src.main:app --port 8000
```

- http://127.0.0.1:8000 — assistant (onglets *Assistant* et *Demande-Réclamation*)
- http://127.0.0.1:8000/revision — **espace agent** : relecture des réponses officielles

## 🛡️ Espace agent — la relecture n'est pas optionnelle

Une réponse pré-rédigée est servie **sans vérification au moment de la question**.
Tant qu'un agent TGR ne l'a pas relue, c'est le texte d'un modèle 3B qui engage
l'administration. L'espace agent met chaque réponse en regard de sa **note source**
et signale :

- **risque élevé** — la note interne est trop courte, le modèle a dû combler les blancs :
  c'est là que naissent les procédures inventées
- **arabe non traduit** — le champ arabe contient encore du français

Tant que le compteur n'affiche pas 44/44, le système n'est pas prêt pour la production.

## 🧪 Vérification

```powershell
python -X utf8 src\verifier_tout.py       # verdict global PASSE / ÉCHOUE (~3 min)
python -X utf8 src\bench_latence.py       # latence + justesse sur 18 questions réelles
python -X utf8 src\audit_couverture.py    # % de questions connues servies instantanément
python -X utf8 eval\evaluate.py           # comparaison RAG classique vs agentic
```

`verifier_tout.py` contrôle : intégrité des données, couverture de la voie rapide,
latence médiane, justesse du périmètre, et avancement de la relecture humaine.

## 🔁 Qualité et intégration continue

```powershell
venv\Scripts\python.exe -m pip install -r requirements-dev.txt

venv\Scripts\python.exe -m ruff check src tests         # analyse statique
venv\Scripts\python.exe -m ruff check src tests --fix   # corrige l'automatisable
venv\Scripts\python.exe -m pytest                       # toute la suite
venv\Scripts\python.exe -m pytest -m "not lourd"        # logique + données (~2 s)
```

> **Windows** : les commandes passent par `python -m` à dessein. La stratégie de
> contrôle d'application bloque les exécutables fraîchement téléchargés —
> `venv\Scripts\pytest.exe` échoue avec « Une stratégie de contrôle
> d'application a bloqué ce fichier », alors que `python -m pytest` fonctionne.

**Ce que couvrent les tests** — la logique de décision, là où une régression
serait invisible : filtre lexical, vote de consensus, nettoyage des réponses,
et l'**intégrité des données servies** (aucune réponse vide, aucun champ arabe
resté en français, aucun crochet `[Votre Nom]` oublié).

Trois de ces tests sont des **tests de non-régression** : ils reproduisent des
défauts réellement rencontrés — l'arabe rejeté par le filtre lexical, la fiche
la mieux dotée en variantes qui l'emportait sur la plus proche, et la signature
« Cordialement, [Votre Nom] » servie à l'usager.

**Avant chaque commit** (facultatif mais recommandé) :

```powershell
venv\Scripts\python.exe -m pre_commit install
```

**Intégration continue** (`.github/workflows/ci.yml`), en trois étages :

| Étage | Contenu | Durée |
|---|---|---|
| Analyse statique et logique | ruff + tests sans la pile ML | ~40 s |
| Moteur de décision | tests du consensus + couverture | ~4 min |
| Image de déploiement | l'image Docker se construit et démarre | ~2 min |

## 🐳 Déploiement

Sur un serveur, sans rien installer d'autre que Docker :

```bash
docker compose up -d --build
docker compose exec ollama ollama pull qwen2.5:3b-instruct   # une seule fois
```

→ http://localhost:8000

Deux services : `ollama` (le modèle) et `assistant` (l'application). Les séparer
permet de redéployer l'assistant **sans retélécharger les 2 Go du modèle**.
Le dossier `data/` est monté en volume : la base vectorielle et surtout les
**réponses relues par les agents** survivent aux redéploiements.

Tout reste sur la machine — aucune donnée d'usager ne sort du réseau TGR.

## 📊 Résultats mesurés

```
Couverture voie rapide : 96 % des questions connues servies en ~0,1 s
Confusions de fiche    :  2 %
Latence                : moyenne 0,12 s | max 0,21 s | 18/18 sous 2 s
Justesse du périmètre  : 17/18
```

Comparaison sur le golden dataset (`eval/RESULTATS.md`) :

| | RAG classique | Agentic RAG |
|---|---|---|
| Bonnes réponses | 9/14 | 10/14 |
| **Hallucinations** | **3** | **1** |
| **Latence moyenne** | **37,7 s** | **2,8 s** |

Les trois questions hors périmètre (capitale de la France, recette de tajine,
crédit immobilier) reçoivent une réponse inventée du RAG classique ; l'agent en
refuse deux sur trois, en un dixième de seconde.

## 📂 Structure

```
data/raw/request_response.xlsx        réclamations sources
data/raw/FAQ_TGR.pdf                  FAQ officielle TGR (14 p.)
data/processed/qa_fiches.json         30 fiches issues des réclamations
data/processed/faq_fiches.json        21 fiches issues de la FAQ
data/processed/reponses_precalculees.json   44 réponses officielles FR + AR
data/chroma_db/                       base vectorielle

src/config.py               toutes les constantes et seuils, commentés
src/lexique.py              vocabulaire du domaine (filtre hors-sujet, 0 ms)
src/prepare_qa.py           xlsx → fiches + variantes usager (FR / darija)
src/prepare_faq.py          FAQ PDF → fiches Q/R
src/ingestion.py            fiches + PDF → ChromaDB
src/retriever.py            recherche sémantique (préfixes E5)
src/llm.py                  wrapper Ollama (generate / stream / decide)
src/agent_rag.py            ⭐ agent : consensus, guardrail, voie rapide, CRAG
src/rag_classic.py          baseline pour la comparaison chiffrée
src/reclamation_handler.py  dépôt / suivi / validation des réclamations
src/semantic_cache.py       cache sémantique + invalidation par empreinte
src/precompute_answers.py   pré-rédaction hors ligne des réponses officielles
src/nettoyer_reponses.py    retire préambules, salutations, signatures
src/retraduire_ar.py        traduction arabe vérifiée (avec réessai)
src/revision.py             relecture humaine (données + validation)
src/main.py                 API FastAPI + SSE
src/verifier_tout.py        vérification complète, verdict PASSE / ÉCHOUE
src/audit_couverture.py     audit de la voie rapide (sans LLM)
src/bench_latence.py        banc d'essai latence + justesse
src/warmup_cache.py         préchauffage du cache avant une démo
static/index.html           interface assistant + réclamations
static/revision.html        espace agent (relecture)
eval/                       golden dataset + évaluation comparative

tests/                      suite de tests (logique, nettoyage, données)
pyproject.toml              configuration ruff + pytest
requirements-dev.txt        outils de qualité
.pre-commit-config.yaml     vérifications avant chaque commit
.github/workflows/ci.yml    intégration continue
Dockerfile                  image de l'application
docker-compose.yml          déploiement (assistant + ollama)
.env.example                modèle de configuration
```

## 🎬 Scénario de démo

1. `Mon mot de passe ne marche plus` → réponse sourcée, **0,1 s**
2. `J'ai changé de téléphone et je n'ai plus mes codes` → fiche MFA, **0,1 s**
3. `كيف أحذف حسابي؟` → réponse en arabe, **0,1 s**
4. `ma9dertch ndkhol l compte dyali` → darija comprise, **0,1 s**
5. `Quelle est la capitale de la France ?` → refus poli, **0,1 s**
6. Onglet **Demande-Réclamation** → lettre officielle + référence `REC-2026-…`, **0,1 s**
7. **Espace agent** → montrer la traçabilité : note source, réponse, validation

## ⚠️ Limites connues

- **La relecture humaine reste à faire** (44 réponses). Sans elle, le contenu engage
  un modèle 3B, pas la TGR.
- **Qualité de l'arabe** : traduction automatique d'un petit modèle, à corriger dans
  l'espace agent.
- **7 fiches sans solution** (« BUG », « Incompréhensible » dans le xlsx source) :
  donnée manquante côté TGR, pas un problème technique.
- **`Guide_Reclamations_En_Ligne.pdf` est scanné** : 12 pages non exploitées faute d'OCR
  (`tesseract` n'est pas installé).
- **Cas limite** : « Comment obtenir un crédit immobilier ? » est traité comme une
  question fiscale sur l'immobilier (la réponse redirige correctement, mais le statut
  devrait être « hors périmètre »). Resserrer les seuils pour ce cas casse des
  comportements corrects — mesuré, donc laissé en l'état.
- **Limite du modèle d'embedding** : sur « J'ai cliqué sur le lien de réinitialisation
  reçu par email mais il ne marche pas », `multilingual-e5-base` place la fiche
  « modification d'email » (0.331) devant la fiche « lien de réinitialisation expiré »
  (0.344) — le mot « email » pèse plus que le reste de la phrase. Vérifié : ce n'est
  pas un défaut des données (les variantes des deux fiches sont spécifiques et
  correctes), ni des seuils. Les deux vraies réponses seraient un modèle plus fort
  (`e5-large`) ou un **reranker** cross-encodeur appliqué aux candidats — l'un et
  l'autre coûtent de la latence et demandent une revalidation complète : à traiter
  comme une v2, pas comme un correctif.
  On a délibérément renoncé à ajouter cette formulation aux variantes de la fiche :
  cela aurait réparé le test sans rien améliorer pour les usagers, et surtout privé
  l'évaluation de toute valeur de preuve.
- **Un seul utilisateur à la fois** sur les questions inédites : Ollama sérialise les
  requêtes. Sans effet sur la voie rapide, qui n'appelle pas le LLM.
- **Réclamations stockées en JSON**, sans authentification : maquette fonctionnelle,
  en attente de l'accès PostgreSQL de la DSI (`src/ingestion_postgres.py`).
