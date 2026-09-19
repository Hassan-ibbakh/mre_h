"""
Pool de cles OpenRouter dediees a l'Observatoire des Talents MRE (extraction
+ agents tool-calling). Distinct de OPENROUTER_API_KEY, utilisee par ailleurs
pour le RAG juridique de Sanad AI.

Round-robin sur les cles OPENROUTER_API_KEY_1 a OPENROUTER_API_KEY_4, chacune
associee au nom de sa variable d'environnement pour pouvoir identifier dans
les logs laquelle est en faute en cas de 401/429.
"""
import itertools
import logging
from typing import List, Tuple

from config import get_settings

logger = logging.getLogger("sanad.ai")


def _load_pool() -> List[Tuple[str, str]]:
    settings = get_settings()
    candidates = [
        ("OPENROUTER_API_KEY_1", settings.OPENROUTER_API_KEY_1),
        ("OPENROUTER_API_KEY_2", settings.OPENROUTER_API_KEY_2),
        ("OPENROUTER_API_KEY_3", settings.OPENROUTER_API_KEY_3),
        ("OPENROUTER_API_KEY_4", settings.OPENROUTER_API_KEY_4),
    ]
    seen = set()
    pool = []
    for env_name, key in candidates:
        if key and key not in seen:
            pool.append((env_name, key))
            seen.add(key)
    if not pool:
        raise RuntimeError(
            "Aucune cle OpenRouter disponible pour le pool de l'Observatoire. "
            "Definissez au moins OPENROUTER_API_KEY_1 dans .env."
        )
    logger.info(
        "Pool de cles OpenRouter (Observatoire) initialise avec %s cle(s) distincte(s) : %s",
        len(pool), ", ".join(name for name, _ in pool),
    )
    return pool


_pool = None
_cycle = None


def _ensure_loaded():
    global _pool, _cycle
    if _pool is None:
        _pool = _load_pool()
        _cycle = itertools.cycle(_pool)


def get_next_openrouter_key() -> str:
    """Retourne la prochaine cle du pool (round-robin)."""
    _ensure_loaded()
    _, key = next(_cycle)
    return key


def get_next_openrouter_key_with_name() -> Tuple[str, str]:
    """Retourne (nom_variable_env, valeur_cle) pour la prochaine cle du pool."""
    _ensure_loaded()
    return next(_cycle)


def pool_size() -> int:
    _ensure_loaded()
    return len(_pool)