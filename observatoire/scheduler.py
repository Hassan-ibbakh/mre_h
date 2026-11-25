"""Worker de synchronisation périodique de l'Observatoire des Talents MRE.

La logique de synchronisation elle-même vit dans `talent_sync.py` ; ce module
ne fait que la relancer à intervalle régulier dans un thread daemon.
"""
import logging
import threading
from typing import Callable

from sqlalchemy.orm import Session

from database import SessionLocal

logger = logging.getLogger("sanad.ai")


def start_daily_sync(sync_fn: Callable[[Session], int], interval_hours: int = 24) -> threading.Event:
    """Démarre un worker en arrière-plan qui relance `sync_fn` périodiquement.

    Retourne un `threading.Event` : le positionner (`.set()`) arrête le worker."""
    stop_event = threading.Event()

    def _runner() -> None:
        while not stop_event.is_set():
            try:
                db = SessionLocal()
                try:
                    sync_fn(db)
                finally:
                    db.close()
            except Exception:
                logger.exception("Erreur pendant la synchronisation planifiée des talents")
            stop_event.wait(interval_hours * 3600)

    thread = threading.Thread(target=_runner, daemon=True, name="talent-sync-worker")
    thread.start()
    return stop_event
