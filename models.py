"""Schémas Pydantic du contrat API Sanad AI.

Deux familles de schémas :

1) Chat juridique immobilier (JSON In -> Markdown streamé) :
   La requête entrante est validée par Pydantic. La réponse, elle, est un flux
   Server-Sent Events (SSE) de Markdown pur — plus aucune validation stricte de
   JSON côté LLM (voir main.py).

2) Observatoire des Talents MRE (JSON In -> JSON Out) :
   Schémas de requête/réponse pour `/api/observatory-chat`, alignés sur le
   scoring hybride mots-clés + sémantique de main.py (`_search_observatory_profiles`).
   Ils documentent le contrat exact (utile pour l'OpenAPI /docs et pour figer
   la forme des données côté frontend), sans remplacer les endpoints qui
   restent en JSONResponse « libre » pour rester 100% compatibles avec le
   JavaScript existant.
"""

from typing import Optional

from pydantic import BaseModel, Field

# Doit rester synchronisé avec SECTORS / SECTOR_LABELS dans main.py.
REAL_ESTATE_SECTORS = ["real_estate", "agriculture", "industry", "tourism"]

# Doit rester synchronisé avec OBSERVATORY_DOMAIN_LABELS dans main.py.
OBSERVATORY_DOMAINS = [
    "tech", "health", "education", "agriculture",
    "industry", "finance", "creative", "social", "other",
]


# ---------------------------------------------------------------------------
# Chat juridique immobilier — API moderne (/api/v1/chat/real-estate)
# ---------------------------------------------------------------------------
class ChatRequest(BaseModel):
    """Corps de la requête entrante."""

    query: str = Field(
        ..., min_length=3, max_length=4000, description="Question sur l'immobilier marocain"
    )
    country: Optional[str] = Field(
        None, description="Pays de résidence du MRE (personnalisation de la réponse)"
    )
    sector: Optional[str] = Field(
        None,
        description=(
            "Secteur RAG pour filtrer la base de connaissances "
            f"({', '.join(REAL_ESTATE_SECTORS)}). Omis = recherche non filtrée."
        ),
    )


# ---------------------------------------------------------------------------
# Observatoire des Talents MRE — agent intelligent (/api/observatory-chat)
# ---------------------------------------------------------------------------
class ObservatoryChatRequest(BaseModel):
    """Corps de la requête entrante (miroir de LegacyObservatoryRequest dans
    main.py, exposé ici pour la doc OpenAPI et une éventuelle route /api/v1)."""

    message: str = Field(
        ..., min_length=1, max_length=500,
        description="Requête en langage naturel décrivant le profil de talent recherché",
    )


class TalentProfile(BaseModel):
    """Un profil de talent tel que retourné par `_search_observatory_profiles`
    (scoring hybride mots-clés + sémantique, embeddings batchés)."""

    id: int
    title: str = Field(..., description="Nom ou intitulé du profil")
    sector_label: str = Field(..., description="Libellé du domaine (ex: Technologie)")
    country: Optional[str] = Field(None, description="Pays de résidence")
    years_experience: Optional[int] = None
    tags: list[str] = Field(default_factory=list, description="Compétences / mots-clés")
    match_score: int = Field(..., ge=0, le=100, description="Score de pertinence 0-100")
    review_status: str = "approved"
    is_masked: bool = False
    description: str = ""
    source_name: str = Field("", description="Source de découverte du profil (sync auto)")
    source_url: str = ""
    image_url: str = ""


class ObservatoryCriteria(BaseModel):
    """Critères extraits/appliqués par l'agent LangChain lors de la recherche."""

    keywords: list[str] = Field(default_factory=list)
    min_years_experience: Optional[int] = None
    sector: Optional[str] = Field(
        None, description=f"Un des domaines : {', '.join(OBSERVATORY_DOMAINS)}"
    )
    country: Optional[str] = None


class ObservatoryChatResponse(BaseModel):
    """Réponse de `/api/observatory-chat` : texte généré par l'agent +
    les meilleurs profils (déjà dédupliqués, triés par match_score décroissant)."""

    response: str = Field(..., description="Réponse en langage naturel de l'agent")
    criteria: ObservatoryCriteria
    matches: list[TalentProfile] = Field(default_factory=list, max_length=15)


class TalentSyncResult(BaseModel):
    """Résultat d'un cycle de synchronisation (`/admin/talents/sync`, manuel
    ou automatique toutes les 24h)."""

    created: int = Field(..., ge=0, description="Nombre de nouveaux profils ajoutés")