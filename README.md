---
title: Mre Ai
emoji: 🌍
colorFrom: purple
colorTo: pink
sdk: gradio
sdk_version: 6.24.0
python_version: '3.12'
app_file: app.py
pinned: false
short_description: An AI-driven assistant and RAG pipeline
---

# MRE AI (Sanad AI) — Assistant pour les Marocains Résidant à l'Étranger

Application web destinée aux **Marocains Résidant à l'Étranger (MRE)** :

1. **Assistant RAG sectoriel** : répond en français et en arabe **exclusivement à partir d'une base
   vectorielle de documents officiels marocains** (IGOC 2024, loi 12-309, CGI 2025, statistiques
   sectorielles…), avec **citations des sources** (document + page) et **réponses en streaming**.
2. **Observatoire des Talents MRE** : annuaire de profils marocains à l'étranger, alimenté
   manuellement et par découverte automatique (RSS, Tavily, ORCID + extraction LLM).
3. **Opportunités d'investissement** : fiches extraites des documents `data/investissement/`
   et validées par un LLM.

```
PDF/.md → clean_pdf.py → ingest.py → ChromaDB (bge-m3, local) → LLM (OpenRouter) → SSE → navigateur
```

## Fonctionnalités

- **Chat RAG streaming** : réponse affichée token par token (Server-Sent Events).
- **Secteurs** : immobilier, agriculture, industrie, tourisme, finance, douane, investissement
  (un sous-dossier de `data/` = un secteur, utilisé comme filtre de recherche).
- **Embeddings locaux** `BAAI/bge-m3` (FR/AR) : aucun appel réseau pour la vectorisation.
- **Anti-hallucination** : filtre par secteur, seuil de similarité, prompt strict, sources citées.
- **Comptes utilisateurs** : inscription → validation par un administrateur → accès aux pages
  privées (chat, observatoire, opportunités). Dashboard admin pour gérer comptes et profils.
- **Ingestion idempotente** : relancer met à jour les chunks d'un fichier sans dupliquer.

## Stack technique

| Couche | Technologie |
|---|---|
| API & web | FastAPI, Uvicorn, Jinja2, Pydantic |
| RAG | ChromaDB (persistant, cosine), langchain-text-splitters |
| Embeddings | `BAAI/bge-m3` (local, CPU ou GPU) |
| Extraction PDF | PyMuPDF (pymupdf4llm) |
| LLM | OpenRouter (`OPENROUTER_MODEL`, défaut `openrouter/free`) |
| Auth & données | SQLAlchemy (SQLite `instance/talents.db`), bcrypt, sessions signées |
| Client web | Fetch + ReadableStream, `marked` + `DOMPurify` (Markdown → HTML nettoyé) |

## Arborescence

```
├── main.py            # API FastAPI : pages web, chat SSE, Observatoire, admin
├── rag_engine.py      # Moteur RAG (retrieval ChromaDB) + opportunités d'investissement
├── ingest.py          # Vectorisation .md/.pdf dans ChromaDB (idempotent)
├── ingest_invest.py   # Ré-ingestion + validation LLM des fiches d'investissement
├── clean_pdf.py       # Prétraitement PDF → Markdown propre
├── config.py          # Configuration pydantic-settings (.env)
├── database.py        # Modèles SQLAlchemy (User, Post, ProfileCache)
├── auth.py            # Hash des mots de passe, session, messages flash
├── talent_sync.py     # Cycle de découverte de talents (bouton admin + worker 24h)
├── observatoire/      # Agents et sources de l'Observatoire (Tavily, RSS, ORCID, LLM)
├── app.py             # Point d'entrée Hugging Face Spaces (Gradio + ZeroGPU)
├── data/<secteur>/    # Documents sources, un dossier par secteur
├── chroma_db/         # Base vectorielle persistante (Git LFS)
├── templates/, static/ # Interface web
├── tests/             # Tests pytest
└── docs/              # Notes de conception et audits
```

## Installation

```powershell
git lfs install
git lfs pull          # indispensable : chroma_db/, PDF et images sont stockés en Git LFS
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements-dev.txt
```

> Sans `git lfs pull`, `chroma_db/chroma.sqlite3` n'est qu'un fichier pointeur de quelques octets
> et l'application échoue avec `file is not a database`.

## Configuration

Copier `.env.example` en `.env` puis renseigner au minimum :

```env
OPENROUTER_API_KEY=sk-or-...
SECRET_KEY=<valeur aléatoire : python -c "import secrets; print(secrets.token_hex(32))">
ADMIN_EMAIL=admin@example.com
ADMIN_PASSWORD=<mot de passe fort>
```

`ADMIN_EMAIL` / `ADMIN_PASSWORD` créent (ou promeuvent) le compte administrateur au démarrage.

Mot de passe oublié ou aucun administrateur ? Créer/réinitialiser un compte admin
(mot de passe demandé de façon masquée) :

```powershell
python create_admin.py vous@example.com
```
Les autres variables sont documentées dans [.env.example](.env.example) et [config.py](config.py).

## Remplir la base de connaissances

**1. (Optionnel) Prétraiter les PDFs bruts** (en-têtes/pieds répétitifs, numéros de page, artefacts OCR) :

```powershell
python clean_pdf.py data
```

**2. Vectoriser** dans ChromaDB (le modèle et la collection sont lus depuis `.env`) :

```powershell
python ingest.py                                   # tout data/
python ingest.py data/immobilier/12309.pdf         # un fichier
python ingest.py data/finance data/douane          # plusieurs secteurs
```

- Le secteur d'un document est le nom de son dossier sous `data/`.
- `chunk_size=3000`, `chunk_overlap=300` ; métadonnées `source`, `page`, `sector`.
- Une version `.md` nettoyée prévaut sur le `.pdf` du même nom.
- **Durée** : `bge-m3` sur CPU ≈ 5-10 s par chunk. Les ~140 tableaux statistiques de
  `data/immobilier/` représentent ~3 500 chunks : préférer une machine avec GPU
  (`EMBEDDING_DEVICE=auto` utilise CUDA si disponible).

## Lancer l'application

```powershell
.\venv\Scripts\python.exe -m uvicorn main:app --port 8000
```

| Route | Accès | Description |
|---|---|---|
| `/` | public | Accueil |
| `/login`, `/register` | public | Connexion / demande d'accès |
| `/chat` | compte validé | Assistant RAG (streaming) |
| `/observatory` | compte validé | Observatoire des Talents |
| `/opportunities` | compte validé | Opportunités d'investissement |
| `/admin` | admin | Dashboard, profils, demandes d'accès |
| `/docs` | public | Documentation OpenAPI |
| `/health` | public | État : modèle, clé API, nombre de chunks |

*(Ouvrir `http://127.0.0.1:8000`, pas `0.0.0.0`.)*

## Contrat API

Les endpoints ci-dessous exigent une **session connectée avec accès validé** (cookie de session) :
`401` sans session, `403` si le compte est en attente ou bloqué.

### `POST /api/v1/chat/real-estate` — streaming SSE

```json
{ "query": "Quels sont les droits d'enregistrement pour un MRE qui achète ?", "sector": "real_estate" }
```

`sector` ∈ `real_estate`, `agriculture`, `industry`, `tourism`, `finance`, `douane`, `investissement`
(défaut `real_estate`). Réponse `text/event-stream` :

```
data: {"type": "sources", "sources": [{"source": "CGI 2025 - Fiscalité Immobilière (Extraits MRE).pdf", "page": 1}]}
data: {"type": "token", "content": "## Droits d'enregistrement"}
...
data: {"type": "done"}
```

Événements : `sources` → `token`* → `done`, ou `error` (quota 429, timeout, LLM injoignable).
Erreurs avant le flux : `401`/`403` (accès), `422` (requête ou secteur invalide), `503` (clé API absente).

### `POST /api/chat`
Même flux SSE, format de requête `{ "message": "...", "sector": "real_estate" }`.

### `POST /api/observatory-chat`
`{ "message": "ingénieure IA au Canada" }` → `{ "response", "criteria", "matches": [...] }`
(profils approuvés, triés par score hybride mots-clés + sémantique).

## Déploiement

- **Hugging Face Spaces** : `app.py` monte la FastAPI sous une mini-app Gradio (exigence ZeroGPU).
- **Docker** (Render, Railway…) : `Dockerfile` fourni, écoute sur `$PORT` (7860 par défaut).
- En production, définir `SECRET_KEY`, `ADMIN_EMAIL`, `ADMIN_PASSWORD` et `OPENROUTER_API_KEY`
  dans les secrets de l'hébergeur.

## Tests

```powershell
python -m pytest
```

## Limitations connues

- Les modèles gratuits OpenRouter ont des quotas journaliers (erreur `429` → message dans le chat).
- L'agent de chat de l'Observatoire fonctionne en mode déterministe (recherche + sélection sans
  tool-calling LLM réel, voir `observatoire/langchain_compat.py`).
- La base SQLite (`instance/`) n'est pas persistante sur les hébergements éphémères.

## Licence

Voir [LICENSE](LICENSE).
