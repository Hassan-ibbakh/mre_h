"""Synchronisation de l'Observatoire des Talents MRE (portage FastAPI)."""
import logging
import threading
import time
from typing import Callable

from sqlalchemy.orm import Session

from database import Post, SessionLocal, classify_mre_review_status
from observatoire.talent_monitor import discover_talents_from_sources

logger = logging.getLogger("sanad.ai")


def sync_talents_dynamic(db: Session) -> int:
    all_posts = db.query(Post).all()
    existing_urls = {p.source_url for p in all_posts if p.source_url}
    existing_titles = {p.title.strip().lower() for p in all_posts}

    discovered = discover_talents_from_sources(
        "data/talent_sources.json",
        existing_urls=existing_urls,
        existing_titles=existing_titles,
    )

    created = 0
    for item in discovered:
        title = item["title"].strip()
        if not title or title.lower() in existing_titles:
            continue
        raw_confidence_score = item.get("confidence_score")
        if raw_confidence_score is not None:
            try:
                confidence_score = max(0, min(100, round(float(raw_confidence_score))))
            except (TypeError, ValueError):
                confidence_score = None
        else:
            try:
                raw_confidence = float(item.get("confidence"))
                confidence_score = round(raw_confidence * 100 if raw_confidence <= 1 else raw_confidence)
                confidence_score = max(0, min(100, confidence_score))
            except (TypeError, ValueError):
                confidence_score = None

        review_status = classify_mre_review_status(confidence_score, item.get("country"))
        post = Post(
            title=title,
            sector=item.get("sector") or "other",
            country=item.get("country") or "Maroc",
            expertise_tags=item.get("expertise_tags", "Talent, Monitor"),
            description=item.get("description", ""),
            years_experience=item.get("years_experience"),
            is_active=review_status == "approved",
            review_status=review_status,
            confidence_score=confidence_score,
            source_url=item.get("url", ""),
            source_name=item.get("source", "Source externe"),
            image_url=item.get("image_url", ""),
            auto_discovered=True,
        )
        db.add(post)
        existing_titles.add(title.lower())
        created += 1

    if created:
        db.commit()
    return created


def start_daily_sync(sync_fn: Callable[[Session], int], interval_hours: int = 24):
    """Démarre un worker en arrière-plan qui relance la synchronisation périodiquement."""
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