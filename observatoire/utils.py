"""Utilitaires partagés pour l'observatoire des talents."""

from __future__ import annotations

import hashlib
import logging
import re
import unicodedata

import numpy as np

logger = logging.getLogger("sanad.ai")

# Stop-words français courants -- a exclure du scoring mots-cles pour ne pas
# diluer le score avec des mots grammaticaux sans valeur discriminante.
STOPWORDS_FR = {
    "le", "la", "les", "un", "une", "des", "de", "du", "au", "aux", "en",
    "et", "ou", "a", "avec", "sans", "pour", "par", "sur", "dans", "chez",
    "je", "tu", "il", "elle", "on", "nous", "vous", "ils", "elles",
    "ce", "cet", "cette", "ces", "qui", "que", "quoi", "dont",
    "est", "sont", "etre", "avoir", "ayant", "ete",
    "recherche", "cherche", "veux", "voudrais", "besoin", "profil",
}


def normalize_text(value: str | None) -> str:
    """Normalise le texte pour la recherche : minuscules, sans accents."""
    if not value:
        return ""
    normalized = unicodedata.normalize("NFKD", value)
    return "".join(ch for ch in normalized if not unicodedata.combining(ch)).lower()


def tokenize_keywords(value: str | None, drop_stopwords: bool = True) -> list[str]:
    """Convertit une chaîne de mots-clés en liste de tokens propres.
    Par défaut, exclut les stop-words français pour ne garder que les
    termes réellement discriminants (compétences, secteur, etc.)."""
    if not value:
        return []
    normalized = normalize_text(value)
    tokens = [token for token in re.split(r"[\s,]+", normalized) if token]
    if drop_stopwords:
        tokens = [t for t in tokens if t not in STOPWORDS_FR and len(t) > 1]
    return tokens


# ---------------------------------------------------------------------------
# Recherche sémantique légère pour l'Observatoire des Talents.
# ---------------------------------------------------------------------------

_EMBED_MODEL = None
_PROFILE_CACHE: dict[int, tuple[str, np.ndarray]] = {}  # post_id -> (text_hash, embedding)


def _get_embed_model():
    global _EMBED_MODEL
    if _EMBED_MODEL is None:
        from sentence_transformers import SentenceTransformer
        logger.info("Chargement du modèle d'embeddings (bge-m3) pour l'Observatoire...")
        _EMBED_MODEL = SentenceTransformer("BAAI/bge-m3")
    return _EMBED_MODEL


def _text_hash(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()


def embed_query(query: str) -> np.ndarray | None:
    if not query or not query.strip():
        return None
    try:
        model = _get_embed_model()
        return np.asarray(model.encode(query, normalize_embeddings=True), dtype=np.float32)
    except Exception:
        logger.exception("Embedding indisponible, repli sur le scoring mots-clés seul.")
        return None


def embed_profile(post_id: int, text: str) -> np.ndarray | None:
    """Conserve pour compatibilité (usage ponctuel, un seul profil -- ex.
    lors de l'ajout manuel d'un profil unique). Pour une recherche sur
    plusieurs profils, utilise embed_profiles_batch() : un seul appel modèle
    au lieu d'un par profil, bien plus rapide (~1min -> quelques secondes
    sur 45 profils, hors chargement initial du modèle)."""
    h = _text_hash(text)
    cached = _PROFILE_CACHE.get(post_id)
    if cached and cached[0] == h:
        return cached[1]
    try:
        model = _get_embed_model()
        vec = np.asarray(model.encode(text, normalize_embeddings=True), dtype=np.float32)
    except Exception:
        logger.exception("Embedding profil %s indisponible.", post_id)
        return None
    _PROFILE_CACHE[post_id] = (h, vec)
    return vec


def embed_profiles_batch(items: list[tuple[int, str]]) -> dict[int, np.ndarray]:
    """Encode en un seul appel modèle tous les profils dont l'embedding n'est
    pas déjà en cache. À utiliser pour toute recherche portant sur plusieurs
    profils (c'est le cas de _search_observatory_profiles dans main.py)."""
    results: dict[int, np.ndarray] = {}
    to_encode: list[tuple[int, str]] = []

    for post_id, text in items:
        h = _text_hash(text)
        cached = _PROFILE_CACHE.get(post_id)
        if cached and cached[0] == h:
            results[post_id] = cached[1]
        else:
            to_encode.append((post_id, text))

    if to_encode:
        try:
            model = _get_embed_model()
            texts = [t for _, t in to_encode]
            vecs = model.encode(texts, normalize_embeddings=True, batch_size=32, show_progress_bar=False)
            for (post_id, text), vec in zip(to_encode, vecs):
                arr = np.asarray(vec, dtype=np.float32)
                _PROFILE_CACHE[post_id] = (_text_hash(text), arr)
                results[post_id] = arr
        except Exception:
            logger.exception("Embedding batch indisponible, profils concernés ignorés (scoring mots-clés seul).")

    return results


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b))


def invalidate_profile(post_id: int) -> None:
    _PROFILE_CACHE.pop(post_id, None)