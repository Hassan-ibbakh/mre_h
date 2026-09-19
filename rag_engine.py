"""Moteur RAG de l'agent immobilier MRE (Sanad AI).

Responsabilités :
    - Interroger la collection ChromaDB alimentée par `ingest.py`.
    - Filtrer par métadonnées (`sector`, optionnel) AVANT l'injection
      dans le contexte : seule la donnée du secteur demandé est injectée.
    - Appliquer un seuil de similarité : les chunks faibles sont écartés
      (anti-hallucination).
    - Formater le contexte Markdown injecté dans le prompt final.

Note d'architecture : la collection est peuplée avec des embeddings explicites
(bge-m3, 1024 dimensions). Il faut donc interroger avec `query_embeddings`,
jamais avec `query_texts` (la fonction d'embedding par défaut de ChromaDB
produirait des vecteurs de dimensions incompatibles).
"""

from __future__ import annotations

import json
import hashlib
import logging
import re
from dataclasses import dataclass
from datetime import date

import httpx
from chromadb import PersistentClient
from langchain_huggingface import HuggingFaceEmbeddings

from config import Settings, get_settings, resolve_device
from observatoire.openrouter_key_pool import get_next_openrouter_key_with_name, pool_size

logger = logging.getLogger("sanad.rag")
_OPPORTUNITY_VALIDATION_CACHE: dict[str, dict | None] = {}
_OPPORTUNITY_VALIDATION_VERSION = "5"


class OpportunityQuotaExceeded(Exception):
    """Signale que le quota LLM ne permet plus de valider ce lot."""


class OpportunityValidationFailed(Exception):
    """Signale une réponse LLM exploitable mais non publiable."""


class OpportunityValidationUnavailable(Exception):
    """Signale une erreur technique empêchant le LLM de rendre un verdict."""

_OPPORTUNITY_TERMS = re.compile(
    r"\b(opportunit\w*|projet\w*|investissement\w*|investir|appel\s+à\s+candidatures|zone\s+industrielle)\b",
    re.IGNORECASE,
)
_REGION_NAMES = (
    "Casablanca-Settat", "Souss-Massa", "Tanger-Tétouan-Al Hoceïma",
    "Rabat-Salé-Kénitra", "Fès-Meknès", "Marrakech-Safi", "Oriental",
    "Béni Mellal-Khénifra", "Drâa-Tafilalet", "Guelmim-Oued Noun",
    "Laâyoune-Sakia El Hamra", "Dakhla-Oued Ed-Dahab",
)
_SECTOR_LABELS = {
    "agriculture": "Agriculture", "agroalimentaire": "Agroalimentaire",
    "industrie": "Industrie", "tourisme": "Tourisme", "immobilier": "Immobilier",
    "énergie": "Énergie", "energies": "Énergie", "mines": "Mines",
    "services": "Services", "technologie": "Technologie", "commerce": "Commerce",
    "finance": "Finance",
}
_FRENCH_MONTHS = {
    "janvier": 1, "février": 2, "fevrier": 2, "mars": 3, "avril": 4,
    "mai": 5, "juin": 6, "juillet": 7, "août": 8, "aout": 8,
    "septembre": 9, "octobre": 10, "novembre": 11, "décembre": 12,
    "decembre": 12,
}
_FRENCH_DATE_RE = re.compile(
    r"\b(\d{1,2})\s+(" + "|".join(_FRENCH_MONTHS) + r")\s+(20\d{2})\b",
    re.IGNORECASE,
)
_NUMERIC_DATE_RE = re.compile(r"\b(\d{1,2})[/-](\d{1,2})[/-](20\d{2})\b")
_URL_RE = re.compile(r"https?://[^\s<>\]\[\)]+", re.IGNORECASE)


def _display_source_name(source: str) -> str:
    name = source.rsplit("/", 1)[-1].rsplit("\\", 1)[-1]
    name = re.sub(r"\.(md|pdf)$", "", name, flags=re.IGNORECASE)
    return re.sub(r"[_-]+", " ", name).strip().title()


def _parse_json_response(content: str) -> dict:
    """Accepte un objet JSON brut ou encapsulé dans un bloc Markdown."""
    cleaned = content.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", cleaned, flags=re.IGNORECASE)
    value = json.loads(cleaned)
    if not isinstance(value, dict):
        raise ValueError("La réponse LLM n'est pas un objet JSON")
    return value


def _has_expired_deadline(content: str) -> bool:
    """Écarte les appels dont la date limite explicite est déjà dépassée."""
    for line in content.splitlines():
        normalized = line.casefold()
        if not any(marker in normalized for marker in (
            "date limite", "date de clôture", "clôture des candidatures",
            "cloture des candidatures", "dernier délai", "dernier delai",
            "délai de dépôt", "delai de depot",
        )):
            continue
        french_match = _FRENCH_DATE_RE.search(line)
        numeric_match = _NUMERIC_DATE_RE.search(line)
        try:
            if french_match:
                day, month_name, year = french_match.groups()
                deadline = date(int(year), _FRENCH_MONTHS[month_name.casefold()], int(day))
            elif numeric_match:
                day, month, year = numeric_match.groups()
                deadline = date(int(year), int(month), int(day))
            else:
                continue
        except ValueError:
            continue
        if deadline < date.today():
            return True
    return False


def _extract_application_url(content: str) -> str | None:
    """Retourne une URL explicitement associée à une inscription/candidature."""
    for match in _URL_RE.finditer(content):
        context = content[max(0, match.start() - 180):match.start()].casefold()
        if any(keyword in context for keyword in (
            "inscription", "préinscription", "preinscription", "candidature",
            "candidater", "formulaire", "soumission",
        )):
            return match.group(0).rstrip(".,;:!?")
    return None


def _validate_opportunity_with_llm(content: str, source: str, settings: Settings) -> dict | None:
    """Valide une fiche et retourne uniquement les champs autorisés à l'affichage."""
    if not settings.OPENROUTER_API_KEY:
        logger.warning("Validation LLM impossible pour '%s' : clé OpenRouter absente.", source)
        return None
    cache_key = hashlib.sha256(
        f"{settings.OPENROUTER_OPPORTUNITY_MODEL}\0{content}".encode("utf-8")
    ).hexdigest()
    if cache_key in _OPPORTUNITY_VALIDATION_CACHE:
        return _OPPORTUNITY_VALIDATION_CACHE[cache_key]

    prompt = f"""Analyse le document suivant comme un vérificateur d'opportunités d'investissement au Maroc.
Réponds uniquement avec un objet JSON contenant exactement:
{{"is_investment_opportunity": true/false, "title": "...", "summary": "...", "investment_rationale": "...", "sector": "...", "region": "..."}}

Mets is_investment_opportunity à true seulement si le document décrit un projet, un actif ou une activité
concrète qu'un investisseur peut réellement financer, créer, reprendre ou exploiter. Un article général,
une statistique, une réglementation, une aide ou un appel à candidature sans projet investissable est false.
Un appel d'offres de prestation, d'audit, de conseil ou de fourniture destiné à une organisation est false :
ce n'est pas une opportunité d'investissement pour un porteur de projet. N'accepte pas un document seulement
parce qu'il contient les mots « investissement » ou « projet ».
En revanche, un appel à projets, une subvention, un programme d'accompagnement ou de financement est true
s'il vise explicitement des entrepreneurs ou entreprises, indique une action concrète (candidater, créer,
étendre ou financer une activité) et présente un bénéfice ou des conditions identifiables. Dans ce cas,
explique précisément ces éléments dans summary et investment_rationale.
La date d'aujourd'hui est le {date.today().isoformat()}. Si le document donne une date limite ou une période
déjà expirée, is_investment_opportunity est false : n'affiche jamais une opportunité terminée comme active.
N'invente aucune information. title, summary et investment_rationale doivent être en français et dérivés
uniquement du document. summary explique clairement l'opportunité en 2 phrases au maximum. Pour une réponse
true, investment_rationale explique en une phrase ce qu'un investisseur peut concrètement faire et pourquoi
le dossier est publiable. Pour une réponse false, laisse les autres champs vides.
sector doit être le secteur principal. region doit être une région marocaine explicitement mentionnée, sinon une chaîne vide.

{content[:7000]}"""
    try:
        try:
            attempts = pool_size()
        except RuntimeError:
            attempts = 1
        for _ in range(attempts):
            try:
                _, api_key = get_next_openrouter_key_with_name()
            except RuntimeError:
                api_key = settings.OPENROUTER_API_KEY
            response = httpx.post(
                f"{settings.OPENROUTER_BASE_URL}/chat/completions",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": settings.OPENROUTER_OPPORTUNITY_MODEL,
                    "messages": [
                        {"role": "system", "content": "Tu es un classifieur strict. Ne réponds qu'en JSON valide."},
                        {"role": "user", "content": prompt},
                    ],
                    "temperature": 0,
                    "max_tokens": 500,
                    "response_format": {"type": "json_object"},
                },
                timeout=30.0,
            )
            if response.status_code == 429:
                logger.warning("Quota OpenRouter atteint pour une clé pendant '%s'.", source)
                continue
            response.raise_for_status()
            payload = response.json()
            raw = ((payload.get("choices") or [{}])[0].get("message") or {}).get("content")
            result = _parse_json_response(raw or "")
            break
        else:
            raise OpportunityQuotaExceeded from None
    except (httpx.HTTPError, ValueError, json.JSONDecodeError, KeyError, TypeError) as exc:
        logger.warning("Validation LLM échouée pour '%s' : %s", source, exc)
        raise OpportunityValidationUnavailable from exc

    if result.get("is_investment_opportunity") is not True:
        _OPPORTUNITY_VALIDATION_CACHE[cache_key] = None
        raise OpportunityValidationFailed
    title = str(result.get("title") or "").strip()
    summary = str(result.get("summary") or "").strip()
    investment_rationale = str(result.get("investment_rationale") or "").strip()
    sector = str(result.get("sector") or "").strip()
    region = str(result.get("region") or "").strip() or None
    if not title or not summary or not investment_rationale or not sector:
        logger.warning("Validation LLM incomplète pour '%s'.", source)
        _OPPORTUNITY_VALIDATION_CACHE[cache_key] = None
        raise OpportunityValidationFailed
    sector = _SECTOR_LABELS.get(sector.casefold(), sector.title())
    if region and not any(name.casefold() == region.casefold() for name in _REGION_NAMES):
        region = next((name for name in _REGION_NAMES if name.casefold() in region.casefold()), None)
    validated = {
        "title": title[:100],
        "excerpt": summary[:400],
        "validation_rationale": investment_rationale[:300],
        "sector": sector,
        "region": region,
    }
    _OPPORTUNITY_VALIDATION_CACHE[cache_key] = validated
    return validated


def extract_opportunities(records: list[dict], limit: int = 8) -> list[dict]:
    """Construit des cartes à partir des documents RAG, sans inventer de chiffres.

    Plusieurs chunks d'un même document sont regroupés afin qu'une source ne
    produise qu'une seule opportunité sur la page d'accueil.
    """
    grouped: dict[str, dict] = {}
    for record in records:
        metadata = record.get("metadata") or {}
        source = str(metadata.get("source") or "").strip()
        content = str(record.get("document") or "").strip()
        if not source or not content or not _OPPORTUNITY_TERMS.search(content):
            continue
        entry = grouped.setdefault(source, {
            "source": source,
            "sector": metadata.get("sector") or "investissement",
            "content": [],
        })
        entry["content"].append(content)

    opportunities = []
    for entry in grouped.values():
        content = "\n".join(entry["content"])
        headings = re.findall(r"(?m)^#{1,3}\s+(.+?)\s*$", content)
        title = headings[0].strip(" #") if headings else _display_source_name(entry["source"])
        region = next((name for name in _REGION_NAMES if name.lower() in content.lower()), None)
        match = _OPPORTUNITY_TERMS.search(content)
        excerpt_start = max(0, (match.start() if match else 0) - 70)
        excerpt = re.sub(r"\s+", " ", content[excerpt_start:excerpt_start + 300]).strip()
        if excerpt_start > 0:
            excerpt = "..." + excerpt
        opportunities.append({
            "title": title[:100],
            "sector": str(entry["sector"]),
            "region": region,
            "excerpt": excerpt,
            "source": entry["source"],
            "application_url": _extract_application_url(content),
        })

    return opportunities[:limit]


def get_rag_opportunities(settings: Settings, limit: int = 8) -> list[dict]:
    """Lit les documents indexés pour alimenter la vitrine d'opportunités."""
    client = PersistentClient(path=settings.CHROMA_PERSIST_DIR)
    collection = client.get_or_create_collection(
        name=settings.CHROMA_COLLECTION,
        metadata={"hnsw:space": "cosine"},
    )
    if collection.count() == 0:
        return []
    data = collection.get(include=["documents", "metadatas"])
    ids = data.get("ids") or []
    records = [
        {"id": record_id, "document": document, "metadata": metadata}
        for record_id, document, metadata in zip(
            ids, data.get("documents") or [], data.get("metadatas") or []
        )
        if str((metadata or {}).get("sector") or "").casefold() == "investissement"
    ]
    candidates = extract_opportunities(records, limit=limit * 3)
    validated = []
    for candidate in candidates:
        source_records = [
            record for record in records
            if str((record.get("metadata") or {}).get("source") or "").strip() == candidate["source"]
        ]
        saved = (source_records[0].get("metadata") or {}) if source_records else {}
        source_content = "\n".join(
            str(record.get("document") or "") for record in source_records
        )
        validation_is_current = (
            saved.get("opportunity_validation_version") == _OPPORTUNITY_VALIDATION_VERSION
        )
        if _has_expired_deadline(source_content):
            collection.update(
                ids=[str(record["id"]) for record in source_records],
                metadatas=[{
                    **(record.get("metadata") or {}),
                    "opportunity_validated": "false",
                    "opportunity_validation_version": _OPPORTUNITY_VALIDATION_VERSION,
                } for record in source_records],
            )
            continue
        if saved.get("opportunity_validated") == "true" and validation_is_current:
            classification = {
                "title": str(saved.get("opportunity_title") or candidate["title"]),
                "excerpt": str(saved.get("opportunity_excerpt") or candidate["excerpt"]),
                "validation_rationale": str(saved.get("opportunity_validation_rationale") or ""),
                "sector": str(saved.get("opportunity_sector") or candidate["sector"]),
                "region": str(saved.get("opportunity_region") or "") or None,
            }
            validated.append({**candidate, **classification})
            if len(validated) >= limit:
                break
            continue
        if saved.get("opportunity_validated") == "false" and validation_is_current:
            continue
        try:
            classification = _validate_opportunity_with_llm(source_content, candidate["source"], settings)
        except OpportunityQuotaExceeded:
            logger.warning("Validation des opportunités interrompue : quota OpenRouter atteint.")
            break
        except OpportunityValidationUnavailable:
            # Une indisponibilité réseau/API n'est pas un refus éditorial. La
            # fiche reste à évaluer lors d'un prochain passage.
            logger.warning("Validation différée pour '%s' : LLM indisponible.", candidate["source"])
            continue
        except OpportunityValidationFailed:
            collection.update(
                ids=[str(record["id"]) for record in source_records],
                metadatas=[{
                    **(record.get("metadata") or {}),
                    "opportunity_validated": "false",
                    "opportunity_validation_version": _OPPORTUNITY_VALIDATION_VERSION,
                }
                           for record in source_records],
            )
            continue
        if classification:
            validated.append({**candidate, **classification})
            saved_metadata = {
                **(source_records[0].get("metadata") or {}),
                "opportunity_validated": "true",
                "opportunity_validation_version": _OPPORTUNITY_VALIDATION_VERSION,
                "opportunity_title": classification["title"],
                "opportunity_excerpt": classification["excerpt"],
                "opportunity_validation_rationale": classification["validation_rationale"],
                "opportunity_sector": classification["sector"],
                "opportunity_region": classification["region"] or "",
            }
            collection.update(
                ids=[str(record["id"]) for record in source_records],
                metadatas=[saved_metadata] * len(source_records),
            )
        if len(validated) >= limit:
            break
    return validated


@dataclass
class RetrievedChunk:
    """Chunk récupéré par le RAG (usage interne au moteur)."""

    source: str
    page: int | None
    content: str


class ImmobilierRAGEngine:
    """Moteur de recherche sémantique sur la base juridique immobilier MRE."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        device = resolve_device(settings.EMBEDDING_DEVICE)
        logger.info("Embeddings sur périphérique '%s' (EMBEDDING_DEVICE=%s).",
                    device, settings.EMBEDDING_DEVICE)
        self._embeddings = HuggingFaceEmbeddings(
            model_name=settings.EMBEDDING_MODEL,
            model_kwargs={"device": device},
            encode_kwargs={"normalize_embeddings": True},
        )
        self._client = PersistentClient(path=settings.CHROMA_PERSIST_DIR)
        self._collection = self._client.get_or_create_collection(
            name=settings.CHROMA_COLLECTION,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(
            "Collection '%s' prête (%d chunk(s)).",
            settings.CHROMA_COLLECTION,
            self._collection.count(),
        )

    @property
    def collection_count(self) -> int:
        return self._collection.count()

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        sector: str | None = None,
    ) -> list[RetrievedChunk]:
        """Recherche sémantique + filtre métadonnées optionnel + seuil de similarité.

        `sector` : si fourni (ex. "immobilier", "finance", "tourisme"), ne retient
        que les chunks de ce secteur. Sinon (défaut), recherche dans TOUS les
        secteurs présents dans la collection (ingest.py déduit le secteur du
        dossier data/<secteur>/).
        """
        k = top_k or self.settings.TOP_K
        if self._collection.count() == 0:
            logger.warning("Collection vide : lancez `python ingest.py` pour alimenter la base.")
            return []

        query_vector = self._embeddings.embed_query(query)

        query_kwargs: dict = {
            "query_embeddings": [query_vector],
            "n_results": k,
            "include": ["documents", "metadatas", "distances"],
        }
        if sector:
            query_kwargs["where"] = {"sector": sector}

        results = self._collection.query(**query_kwargs)

        documents = (results.get("documents") or [[]])[0]
        metadatas = (results.get("metadatas") or [[]])[0]
        distances = (results.get("distances") or [[]])[0]

        chunks: list[RetrievedChunk] = []
        for doc, meta, dist in zip(documents, metadatas, distances):
            # Distance cosine ChromaDB : plus bas = plus proche (1 - cos).
            if dist > self.settings.SIMILARITY_THRESHOLD:
                continue
            chunks.append(
                RetrievedChunk(
                    source=meta.get("source", "inconnu"),
                    page=meta.get("page"),
                    content=doc,
                )
            )

        logger.info("RAG : %d source(s) retenue(s) sur %d candidat(s).", len(chunks), len(documents))
        return chunks

    def build_context(self, chunks: list[RetrievedChunk]) -> str:
        """Formate les chunks en contexte Markdown sourcé pour le prompt LLM."""
        blocks = [
            f"[Document: {chunk.source}, page {chunk.page or 'N/A'}]\n{chunk.content}"
            for chunk in chunks
        ]
        return "\n\n".join(blocks)


_engine: ImmobilierRAGEngine | None = None


def get_rag_engine() -> ImmobilierRAGEngine:
    """Singleton paresseux du moteur RAG (chargement lourd : bge-m3)."""
    global _engine
    if _engine is None:
        _engine = ImmobilierRAGEngine(get_settings())
    return _engine
