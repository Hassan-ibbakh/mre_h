import logging

from sqlalchemy.orm import Session

from database import Post, classify_mre_review_status
from observatoire.talent_monitor import discover_talents_from_sources

logger = logging.getLogger("sanad.ai")


def sync_talents_dynamic(db: Session) -> int:
    """
    Lance la decouverte AUTONOME de talents (RSS + Tavily + ORCID via
    agents/talent_monitor.py), et insere en base les nouveaux profils juges
    pertinents par l'agent LLM d'extraction (OpenRouter).
    """
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
        confidence_score = max(0, min(100, round(float(item.get("confidence_score", 0)))))
        review_status = classify_mre_review_status(confidence_score, item.get("country"))
        post = Post(
            title=title,
            sector=item.get("sector") or "other",
            country=item.get("country") or "Maroc",
            expertise_tags=item.get("expertise_tags", "Talent, Monitor"),
            description=item.get("description", ""),
            years_experience=item.get("years_experience"),
            is_active=review_status in {"approved", "masked"},
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

    logger.info("Sync talents : %s nouveau(x) profil(s) insere(s).", created)
    return created