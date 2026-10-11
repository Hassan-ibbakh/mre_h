"""Configuration centralisée du backend Sanad AI (pydantic-settings).

Les variables sont lues depuis le fichier `.env` à la racine du projet.
Chaque valeur peut être surchargée par une variable d'environnement du même nom.
"""

from functools import lru_cache
from typing import List

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Configuration du microservice Sanad AI (agent immobilier MRE)."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # ----- LLM (OpenRouter) -----
    OPENROUTER_API_KEY: str = Field(
        default="",
        description="Clé API OpenRouter (https://openrouter.ai/keys). Vide = chat désactivé avec 503.",
    )
    OPENROUTER_BASE_URL: str = Field(
        default="https://openrouter.ai/api/v1",
        description="Endpoint OpenAI-compatible d'OpenRouter.",
    )
    OPENROUTER_MODEL: str = Field(
        default="openrouter/free",
        description="Routeur gratuit OpenRouter, qui sélectionne un modèle disponible.",
    )
    OPENROUTER_OPPORTUNITY_MODEL: str = Field(
        default="openrouter/free",
        description="Modèle OpenRouter utilisé pour valider et classifier les opportunités d'investissement.",
    )
    LLM_TEMPERATURE: float = Field(default=0.1, ge=0.0, le=1.0)
    LLM_MAX_TOKENS: int = Field(default=4096, gt=0, le=32768)
    SECRET_KEY: str = Field(default="dev_secret_key_change_in_production")
    ADMIN_EMAIL: str = Field(default="", description="Email configuré pour créer l'administrateur initial.")
    ADMIN_PASSWORD: str = Field(default="", description="Mot de passe configuré pour créer l'administrateur initial.")
    # ----- Observatoire des Talents MRE (agents LangChain) -----
    OPENROUTER_API_KEY_1: str = Field(
        default="",
        description="1ère clé OpenRouter dédiée au pool de l'Observatoire (extraction + agents).",
    )
    OPENROUTER_API_KEY_2: str = Field(
        default="",
        description="2ème clé OpenRouter dédiée au pool de l'Observatoire (extraction + agents).",
    )
    OPENROUTER_API_KEY_3: str = Field(
        default="",
        description="3ème clé OpenRouter dédiée au pool de l'Observatoire (extraction + agents).",
    )
    OPENROUTER_API_KEY_4: str = Field(
        default="",
        description="4ème clé OpenRouter dédiée au pool de l'Observatoire (extraction + agents).",
    )
    OPENROUTER_EXTRACTOR_MODEL: str = Field(
        default="openrouter/free",
        description="Modèle OpenRouter gratuit pour extraire/valider un profil de talent (JSON structuré).",
    )
    OPENROUTER_AGENT_MODEL: str = Field(
        default="openrouter/free",
        description="Modèle OpenRouter gratuit pour les agents tool-calling (scout + observatory). "
        "Doit supporter le tool-calling sur OpenRouter.",
    )
    TAVILY_API_KEY: str = Field(
        default="",
        description="Clé API Tavily pour la découverte web de talents MRE.",
    )

    # ----- RAG / ChromaDB -----
    CHROMA_PERSIST_DIR: str = Field(default="chroma_db", description="Dossier de persistance ChromaDB.")
    CHROMA_COLLECTION: str = Field(
        default="immobilier_mre",
        description="Collection dédiée au droit immobilier marocain pour les MRE.",
    )
    EMBEDDING_MODEL: str = Field(
        default="BAAI/bge-m3",
        description="Embeddings locaux multilingues FR/AR (1024 dimensions, compatibles avec la collection existante).",
    )
    EMBEDDING_BATCH_SIZE: int = Field(
        default=64,
        ge=1,
        le=512,
        description="Taille de lot d'embedding (CPU : 64-128 accélère nettement).",
    )
    EMBEDDING_DEVICE: str = Field(
        default="auto",
        description="Périphérique d'embedding : 'auto' (cuda si dispo, sinon cpu), 'cpu', 'cuda'.",
    )
    TOP_K: int = Field(default=6, ge=1, le=20, description="Nombre de chunks récupérés par le RAG.")
    SIMILARITY_THRESHOLD: float = Field(
        default=0.60,
        ge=0.0,
        le=1.0,
        description="Seuil de distance cosine ChromaDB (plus bas = plus proche). "
        "Les chunks au-delà du seuil sont écartés : anti-hallucination. "
        "Observé sur bge-m3 : pertinence forte < 0.45, moyenne < 0.60.",
    )

    # ----- API -----
    CORS_ORIGINS: List[str] = Field(
        default_factory=lambda: ["http://localhost:3000", "http://127.0.0.1:8000"],
        description="Origines CORS autorisées (format JSON dans .env, ex: [\"http://x\"]).",
    )
    ENABLE_DAILY_SYNC: bool = Field(
        default=False,
        description="Activer la synchronisation automatique des talents toutes les 24h."
    )

    # ----- Cache & Deduplication (Observatoire) -----
    ENABLE_PROFILE_CACHE: bool = Field(
        default=True,
        description="Activer le cache des profils/URLs pour éviter la réutilisation. "
        "Réduit ~50% de la consommation tokens en éliminant les doublons ORCID.",
    )
    CACHE_RETENTION_DAYS: int = Field(
        default=30,
        ge=1,
        le=365,
        description="Nombre de jours à conserver une URL en cache après traitement. "
        "Après cette période, la URL est purgée et peut être retraitée.",
    )
    SKIP_CACHED_PROFILES: bool = Field(
        default=True,
        description="Si True, passer les URLs déjà en cache (approved/rejected). "
        "Si False, retraiter même si en cache (debug/dev mode).",
    )
    CACHE_CLEANUP_INTERVAL_DAYS: int = Field(
        default=7,
        ge=1,
        description="Intervalle (jours) entre les nettoyages du cache.",
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Singleton de configuration (recalculé à chaud pour les tests)."""
    return Settings()


def resolve_device(device: str | None = None) -> str:
    """Résout 'auto'/'vide' vers 'cuda' si un GPU est disponible (sinon 'cpu').

    Si explicitement demandé ('cuda'), vérifie que CUDA est réellement disponible
    avant de l'utiliser (sinon retombe sur 'cpu')."""
    requested = (device or "").strip().lower()
    if requested in ("", "auto", "cuda"):
        try:
            import torch
            if torch.cuda.is_available():
                return "cuda"
        except ImportError:
            pass
    return "cpu"
