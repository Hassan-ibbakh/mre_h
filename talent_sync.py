"""Synchronisation de l'Observatoire des Talents MRE (implémentation unique).

Utilisée à la fois par le bouton admin (POST /admin/talents/sync) et par le
worker périodique (observatoire.scheduler.start_daily_sync).
"""

import logging

from sqlalchemy.orm import Session

from database import Post, classify_mre_review_status
from observatoire.talent_monitor import discover_talents_from_sources

logger = logging.getLogger("sanad.ai")

TALENT_SOURCES_FILE = "data/talent_sources.json"


def _confidence_score(item: dict) -> int | None:
    """Normalise la confiance de l'extracteur en score 0-100.

    Accepte `confidence_score` (0-100) ou `confidence` (0-1 ou 0-100).
    Retourne None si la valeur est absente ou illisible."""
    raw = item.get("confidence_score")
    if raw is None:
        raw = item.get("confidence")
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return None
    if value <= 1:
        value *= 100
    return max(0, min(100, round(value)))


def sync_talents_dynamic(db: Session) -> int:
    """Lance la découverte autonome de talents (RSS + Tavily + ORCID) et insère
    en base les nouveaux profils jugés pertinents par l'extracteur LLM.

    Retourne le nombre de profils créés."""
    all_posts = db.query(Post).all()
    existing_urls = {p.source_url for p in all_posts if p.source_url}
    existing_titles = {p.title.strip().lower() for p in all_posts}

    discovered = discover_talents_from_sources(
        TALENT_SOURCES_FILE,
        existing_urls=existing_urls,
        existing_titles=existing_titles,
    )

    created = 0
    for item in discovered:
        title = (item.get("title") or "").strip()
        if not title or title.lower() in existing_titles:
            continue
        confidence_score = _confidence_score(item)
        country = (item.get("country") or "").strip() or None
        review_status = classify_mre_review_status(confidence_score, country)
        db.add(Post(
            title=title,
            sector=item.get("sector") or "other",
            country=country,
            expertise_tags=item.get("expertise_tags", ""),
            description=item.get("description", ""),
            years_experience=item.get("years_experience"),
            # Seuls les profils approuvés sont publiés ; « masked » reste inactif
            # (même convention que le bouton de bascule du dashboard admin).
            is_active=review_status == "approved",
            review_status=review_status,
            confidence_score=confidence_score,
            source_url=item.get("url", ""),
            source_name=item.get("source", "Source externe"),
            image_url=item.get("image_url", ""),
            auto_discovered=True,
        ))
        existing_titles.add(title.lower())
        created += 1

    if created:
        db.commit()

    logger.info("Sync talents : %s nouveau(x) profil(s) inséré(s).", created)
    return created
