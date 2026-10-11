# MRE AI / Sanad AI — Codebase Résumé & Audit des Agents

Date de l'audit : 2026-08-11 — Analyse statique uniquement (aucun code exécuté).

---

## 1. Vue d'ensemble

Application de conseil **immobilier et fiscal pour les Marocains Résidant à l'Étranger (MRE)**, servie par **FastAPI**, avec deux univers :

1. **Agent RAG juridique** : chat en streaming (SSE) qui répond exclusivement à partir d'une base vectorielle ChromaDB de textes officiels marocains (CGI, IGOC, loi 12-309...), multilingue FR/AR, avec citations de sources.
2. **Observatoire des Talents MRE** : base SQLAlchemy de profils de talents marocains à l'étranger + agents LangChain de découverte web (RSS / Tavily / ORCID) et de chat de recherche de profils, + dashboard admin avec authentification.

```
PDF → clean_pdf.py → .md propre → ingest.py → ChromaDB (bge-m3) → LLM OpenRouter (SSE) → navigateur
```

---

## 2. Arborescence et rôle des modules

| Fichier | Rôle |
|---|---|
| `main.py` | API FastAPI : routes web, chat SSE RAG, compat ancien frontend, routes Observatoire, auth admin, dashboard, `/health`. |
| `rag_engine.py` | Moteur RAG : retrieval ChromaDB filtré (`sector=="immobilier"`) + seuil de similarité + formatage du contexte. |
| `models.py` | Schémas Pydantic (`ChatRequest`). |
| `config.py` | Configuration pydantic-settings (`.env`) : clés OpenRouter, modèle, ChromaDB, seuil, CORS. |
| `ingest.py` | Vectorisation `.md`/`.pdf` → ChromaDB (idempotent, bge-m3 local, chunks 3000/300). |
| `clean_pdf.py` | Prétraitement PDF bruts → `.md` propres (nettoyage regex + fréquence de lignes). |
| `auth.py` | Hash bcrypt, session cookie signée (Starlette SessionMiddleware), flash messages. |
| `database.py` | SQLAlchemy : modèles `User` et `Post`, engine SQLite, `init_db`, `get_db`. |
| `talent_sync.py` | Orchestration sync talents (root, importé par `main.py`). |
| `observatoire/` | Agents et outils de l'Observatoire (voir §4). |
| `templates/` + `static/` | Interface web historique (Jinja2), conservée telle quelle. |
| `tests/` | 1 test unitaire (tokenisation mots-clés). |
| `data/` | PDFs juridiques, `talent_sources.json` (RSS), `talents_seed.json` (profils de secours). |

---

## 3. Résumé du fonctionnement

### 3.1 Pipeline RAG juridique
1. **Ingestion** : `clean_pdf.py` nettoie les PDFs (numéros de page, en-têtes/pieds répétitifs, tables des matières, artefacts OCR) et écrit des `.md` avec marqueurs `<!-- PAGE N -->`. `ingest.py` découpe (3000/300), embarque en local avec `BAAI/bge-m3` (1024 dims, CPU, FR+AR) et stocke dans ChromaDB. **Idempotent** : un fichier modifié remplace ses chunks sans dupliquer ; un `.md` prévaut sur son `.pdf`.
2. **Retrieval** : `rag_engine.retrieve()` filtre par métadonnée `sector == "immobilier"`, interroge par `query_embeddings` (jamais `query_texts`, dimensions incompatibles) et écarte les chunks au-delà du seuil de distance cosine (`SIMILARITY_THRESHOLD`, défaut 0.60).
3. **Génération** : `main._stream_llm()` envoie prompt système + contexte à OpenRouter (`stealth/ox-alpha`) en streaming et relaye chaque token en SSE (`sources` → `token`* → `done` | `error`).

### 3.2 Interface web & compat
- `url_for` Flask re-créé en global Jinja (`_ENDPOINT_URLS`), `get_flashed_messages` re-créé à partir de la session.
- `/api/chat` renvoie le flux SSE brut (lisible par le JS historique) ; `/api/v1/chat/real-estate` est l'API moderne.

### 3.3 Observatoire des Talents
- Modèle `Post` : profil talent (titre, secteur, pays, tags, années d'expérience, sources, `auto_discovered`).
- `/observatory`, `/admin` (dashboard), `/login`/`/register`/`/logout` (session signée).
- `/api/observatory-chat` : agent qui interroge la base (voir verdict §5).
- `/admin/talents/sync` + worker 24h (`scheduler.start_daily_sync`, thread daemon, `ENABLE_DAILY_SYNC`).

---

## 4. Les « agents » du projet (Observatoire)

| Agent / composant | Fichier | Nature |
|---|---|---|
| Chat Observatoire | `observatory_agent.py` | Agent tool-calling LangChain : `list_options`, `search_profiles`, `select_matches`. |
| Scout de découverte | `talent_scout_agent.py` | Agent tool-calling : `search_sector`, `search_news`, `search_rss`, `search_orcid`, `check_duplicate`, `validate_and_store_profile`. |
| Extracteur de profils | `talent_extractor_llm.py` | Appel LLM OpenRouter → JSON structuré (nom, pays, secteur, tags...). |
| Source ORCID | `orcid_source.py` | Collecte déterministe via API ORCID (sans LLM). |
| Recherche web | `talent_web_search.py` | Tavily (recherche par domaine + articles de presse). |
| Retry multi-clés | `retryable_openrouter.py` + `openrouter_key_pool.py` | Rotation round-robin sur 2 clés OpenRouter (401/429). |
| Compat LangChain | `langchain_compat.py` | **Stub de remplacement** d'`AgentExecutor`/`create_tool_calling_agent`. |
| Scheduler | `scheduler.py` | Worker daemon 24h. |

---

## 5. Audit : les agents fonctionnent-ils ? (analyse statique)

### ✅ Agent RAG immobilier (`rag_engine.py` + `main._stream_llm`)
**Fonctionnel.** Pipeline cohérent : filtrage métadonnées → seuil → prompt → SSE. Dégrade proprement (collection vide → message honnête ; pas de clé → HTTP 503 ; 429/timeout → événement `error`). Prérequis : collection ChromaDB remplie + `OPENROUTER_API_KEY`.

### ⚠️ Agent Chat Observatoire (`observatory_agent.py`)
**NE fait pas de vrai tool-calling.** `main.py` importe `create_tool_calling_agent` depuis `observatoire/langchain_compat.py`, qui est un **stub** : son `_FallbackToolCallingAgent.invoke()` ne traite QUE les noms d'outils du scout (`search_sector`, `search_news`, `search_rss`, `search_orcid`, `check_duplicate`, `validate_and_store_profile`). Les outils de l'Observatoire (`list_options`, `search_profiles`, `select_matches`) ne correspondent à aucun de ces noms → le stub ne fait rien et retourne un message générique *« Fallback d'agent activé »*. `selected` reste vide → `matches: []`, réponse non pertinente.
→ **À réparer** : soit réécrire le stub pour les 3 outils de l'Observatoire, soit installer le vrai `langchain`/`langchain-openai` et supprimer le stub.

### ⚠️ Agent Scout de découverte (`talent_scout_agent.py`)
Le **tour LLM lui-même est également stubbé** (même import `langchain_compat`), mais le module contient un **fallback déterministe robuste** (lignes 354-404) : si `discovered` est vide après `executor.invoke`, il appelle les outils directement (tous les domaines + news/RSS/ORCID), puis valide chaque URL en cache via `validate_and_store_profile`, et se replie sur `data/talents_seed.json`.
→ **Fonctionnel en mode fallback**, sous réserve que : Tavily/RSS/ORCID joignables ET le pool OpenRouter (`OPENROUTER_API_KEY_1`/`_2`) configuré. Sans clés, tout se dégrade silencieusement (candidats rejetés / aucune découverte).

### ✅ Agent Extracteur de profils (`talent_extractor_llm.py`)
**Fonctionnel.** Appel `httpx` → OpenRouter (`OPENROUTER_EXTRACTOR_MODEL`), parse JSON tolérant (`raw_decode`), garde-fous solides (confidence ≥ 0.6, nom réel unique, pays étranger confirmé). En cas d'échec ou de pool vide → retourne `None` (traité comme rejet par l'appelant). Aucun crash possible.

### ✅ Source ORCID (`orcid_source.py`)
**Fonctionnel** (déterministe, pas de LLM). Précision décente : résidence étrangère via affiliations + origine marocaine via études au Maroc/bio/nom. Dépend de `requests` et du réseau. Garde-fou déduplication par URL.

### ⚠️ Retry multi-clés (`retryable_openrouter.py` / `openrouter_key_pool.py`)
**Logique OK** (rotation 401/429, `max_retries=0` pour éviter le double-retry), **mais** : `openrouter_key_pool` lève `RuntimeError` si aucune clé `_1`/`_2` — appelants concernés gèrent l'exception (dégradation propre). Et il dépend de `openai` + `langchain-openai` (voir §6).

### ✅ Scheduler (`scheduler.py`)
**Fonctionnel** : thread daemon, stop propre via `threading.Event`, session DB par cycle.

---

## 6. Problèmes bloquants / à corriger

1. **`requirements.txt` incomplet** — l'application ne démarre pas avec une installation fraîche. Modules importés mais NON déclarés :
   - `bcrypt` (importé en tête de `auth.py`, lui-même importé par `main.py`) → `ModuleNotFoundError` au démarrage.
   - `openai` et `langchain-openai` (importés en tête de `retryable_openrouter.py`, importé par `observatory_agent.py` → `main.py`) → `ModuleNotFoundError` au démarrage.
   - `requests` (importé en tête de `orcid_source.py`) → dépendance transitive probable via chromadb/posthog, mais non garantie.
   - `tavily-python` (import PAUSIBLE dans `talent_web_search.py`, try/except → non bloquant).
   - `sqlalchemy` : présent en transitif via chromadb, mais non déclaré.
   - `langchain-core` : transitif via `langchain-huggingface`/`langchain-text-splitters`.
   → Ajouter au minimum : `bcrypt`, `openai`, `langchain-openai`, `requests`, `tavily-python`.

2. **`database.create_default_admin_if_missing()` (ligne 66)** : construit `User(email=..., password_hash=..., country="Maroc", sector_interest="real_estate")` alors que le modèle `User` n'a que `id/email/password_hash/created_at` → `TypeError` garanti si jamais appelé. Fonction morte (aucun appel), mais à corriger ou supprimer.

3. **`sync_talents_dynamic` dupliqué** : même fonction dans `talent_sync.py` (root) et `observatoire/scheduler.py`. `main.py` importe la version root ; `scheduler.py` utilise la sienne. Risque de divergence, à unifier.

4. **Agent Chat Observatoire non fonctionnel** (voir §5) : le stub `langchain_compat` ne couvre pas les outils de `observatory_agent.py`. Soit implémenter le vrai `create_tool_calling_agent`, soit adapter le fallback.

5. **`PostView` (main.py:149)** : code mort (aucune utilisation trouvée).

6. **Commentaires GROQ/« Groq »** obsolètes dans `talent_monitor.py`/`talent_scout_agent.py` (portage Groq→OpenRouter) : purement cosmétique.

---

## 7. Points forts observés

- Ingestion idempotente avec migration PDF→.md sans doublon.
- Anti-hallucination RAG en 4 couches (filtre métadonnées, seuil, prompt strict, événement `sources`).
- Dégradation gracieuse partout : clés absentes, quota 429, timeout, pages en échec → jamais de crash exposé au client.
- Déduplication intelligente (Levenshtein `SequenceMatcher`, normalisation Unicode).
- Séparation claire clé RAG (`OPENROUTER_API_KEY`) vs pool Observatoire (`_1`/`_2`).
- Fallback déterministe du scout (RSS/Tavily/ORCID + seeds) même si l'agent LLM échoue.

---

## 8. Conclusion

- **Agent RAG** : opérationnel.
- **Extracteur LLM, ORCID, Scheduler** : opérationnels.
- **Scout de découverte** : fonctionne via son fallback déterministe (le raisonnement LLM est un stub).
- **Chat Observatoire** : **non fonctionnel** (stub incompatible).
- **Démarrage global** : dépend d'ajouts manquants dans `requirements.txt` (`bcrypt`, `openai`, `langchain-openai`, `requests`, `tavily-python`).

Priorité de correction : 1) requirements.txt → 2) stub du chat Observatoire → 3) `create_default_admin_if_missing` / duplication `sync_talents_dynamic`.
