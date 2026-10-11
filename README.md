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

# MRE AI — Agent Immobilier pour les Marocains Résidant à l'Étranger

Application de conseil **immobilier et fiscale** pour les **Marocains Résidant à l'Étranger (MRE)**.
Un agent IA (RAG) répond aux questions juridiques en français et en arabe, **exclusivement à partir
d'une base vectorielle de textes officiels marocains** (IGOC 2024, loi 12-309, CGI 2025, etc.),
avec **citations des sources** (document + page) et **réponses en streaming temps réel**.

```
PDF → clean_pdf.py → Markdown propre → ingest.py → ChromaDB (bge-m3) → LLM (OpenRouter) → SSE → navigateur
```

## Fonctionnalités

- **Chat RAG streaming** : réponse qui s'affiche **token par token** dans la bulle (Server-Sent Events,
  aucune attente de la fin de la requête).
- **Interface web complète** servie par FastAPI : accueil, chat, observatoire, login/register/admin.
- **Base de connaissances locale** : aucun appel réseau pour l'embedding — confidentialité des données.
- **Multilingue FR/AR** : embeddings `BAAI/bge-m3`.
- **Anti-hallucination** : réponse restreinte aux chunks pertinents, seuil de similarité, sources citées.
- **Idempotent** : l'ingestion remplace les chunks d'un fichier modifié sans jamais dupliquer.

## Stack technique

| Couche | Technologie |
|---|---|
| API & web | FastAPI, Uvicorn, Jinja2, Pydantic |
| RAG | ChromaDB (persistant, cosine), langchain-text-splitters |
| Embeddings | `BAAI/bge-m3` (local, CPU, FR+AR) |
| Extraction PDF | PyMuPDF (pymupdf4llm) |
| LLM | `stealth/ox-alpha` via OpenRouter (SSE) |
| Client web | Fetch + ReadableStream, `marked.js` (Markdown→HTML) |
| Divers | loguru, python-dotenv |

## Arborescence

```
├── main.py            # API FastAPI : routes web + chat SSE + compat. frontend
├── rag_engine.py      # Moteur RAG : retrieval ChromaDB + construction du prompt
├── models.py          # Contrats API Pydantic
├── config.py          # Configuration pydantic-settings (.env)
├── ingest.py          # Vectorisation .md/.pdf dans ChromaDB (idempotent)
├── app.py             # Point d'entrée Hugging Face Spaces (SDK Gradio)
├── requirements.txt
├── .env               # OPENROUTER_API_KEY=sk-or-... (secret, non commité)
├── data/              # Documents sources (PDF bruts + .md nettoyés)
├── chroma_db/         # Base vectorielle persistante (générée par ingest.py)
├── templates/         # Interface web historique
└── static/            # Images (logo, etc.)
```

## Installation

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Configuration

Créer le fichier `.env` :

```env
OPENROUTER_API_KEY=sk-or-...
```

Variables optionnelles : `OPENROUTER_MODEL` (défaut `poolside/laguna-s-2.1:free`),
`CHROMA_PERSIST_DIR`, `CHROMA_COLLECTION`, `EMBEDDING_MODEL`, `SIMILARITY_THRESHOLD`, `TOP_K`.

## Remplir la base de connaissances

**1. Prétraiter les PDFs bruts** (en-têtes/pieds répétitifs, numéros de page, tables des matières, artefacts OCR) :

```powershell
python clean_pdf.py data
```

**2. Vectoriser** dans ChromaDB :

```powershell
python ingest.py                    # tous les documents de data/
python ingest.py data/loi.pdf      # un fichier précis
```

- Extraction Markdown page par page (`pymupdf4llm`), échecs de pages tracés.
- `chunk_size=3000`, `chunk_overlap=300` (grand contexte LLM).
- Métadonnées par chunk : `source`, `page`, `sector` (filtre RAG).
- **Idempotent** : relancer met à jour les chunks sans dupliquer. Une version `.md` nettoyée
  prévaut sur son `.pdf` d'origine (les anciens chunks sont purgés).

## Lancer l'application

```powershell
.\venv\Scripts\python.exe -m uvicorn main:app --port 8000
```

| Route | Description |
|---|---|
| `http://127.0.0.1:8000/` | Accueil |
| `http://127.0.0.1:8000/chat` | Chat avec l'agent immobilier (streaming) |
| `http://127.0.0.1:8000/docs` | Documentation interactive de l'API |
| `http://127.0.0.1:8000/health` | État : modèle, clé API, nb de chunks |

*(Ne pas ouvrir `0.0.0.0` dans un navigateur : utiliser `127.0.0.1`)*

## Contrat API

### `POST /api/v1/chat/real-estate` — réponse en streaming (SSE)

```json
{ "query": "Quels sont les droits d'enregistrement pour un MRE qui achète ?" }
```

Réponse : flux **Server-Sent Events** (`Content-Type: text/event-stream`), aucun JSON global attendu.

```
data: {"type": "sources", "sources": [{"source": "CGI 2025 - Fiscalité Immobilière (Extraits MRE)", "page": 1}]}
data: {"type": "token", "content": "### Droits d'enregistrement"}
data: {"type": "token", "content": " pour un MRE..."}
...
data: {"type": "done"}
```

Événements : `sources` (premier, documents RAG utilisés) → `token` (Markdown pur) → `done`.
Défaillance en cours de flux : `error` (quota 429, timeout, LLM injoignable).
Erreurs avant le début du stream : HTTP classique (`503` clé absente, `422` requête invalide).

### `POST /api/chat` — compatibilité
Même flux SSE brut (mêmes événements). Utilisé par l'interface historique des templates
(bulles, quick replies) : le JavaScript lit le `ReadableStream` et injecte les tokens
en temps réel dans le DOM, sans buffer côté serveur.

### `POST /api/observatory-chat` — stub
Structure de réponse compatible (observatoire non implémenté : résultats vides).

## Comportement du chat (frontend)

- La bulle du bot est créée immédiatement ; chaque événement `token` est rendu **instantanément**
  (effet machine à écrire) — aucun gel de l'interface, même sur des réponses longues.
- Conversion Markdown → HTML côté client avec `marked.js` (CDN, avec text/plain en fallback hors-ligne).
- Les **sources** citées s'affichent sous la bulle (`Source : loi 12-309 · p.5`).

## Anti-hallucination

1. Filtrage métadonnée `sector` (rag_engine.py).
2. Seuil de similarité : chunks trop éloignés écartés.
3. Prompt strict : citations obligatoires, interdiction d'inventer, Markdown pur.
4. Événement `sources` en tête de chaque réponse (traçabilité).

## Exemple de questions exploitables

- « Quelles conditions pour ouvrir un compte en dirhams convertibles ? » (IGOC 2024)
- « Quels droits d'enregistrement pour un MRE qui achète ? » (CGI extraits)
- « Quelles opérations sur un compte de capital ? » (IGOC 2024)

## Limitations / roadmap

- **Observatoire** : route présente (structure vide), pas de base de profils — à compléter.
- **Authentification** : `/login`, `/register`, `/admin` sont des coquilles (pas de sessions ni utilisateurs).
- **Multi-agents** : l'interface propose agriculture/industrie/tourisme, le backend est mono-agent (immobilier).
- Quota du modèle gratuit OpenRouter (~200 req/jour).

## Licence

Propriété des auteurs. Usage personnel/démo uniquement.
