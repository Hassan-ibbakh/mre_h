#!/usr/bin/env python
"""Ré-ingère les fiches d'investissement puis déclenche leur validation LLM.

Usage :
    python ingest_invest.py

Équivaut au bouton « Ingérer les opportunités » du dashboard admin :
1. ingestion de data/investissement/ dans la collection ChromaDB configurée
   (les sources supprimées du dossier sont purgées de l'index) ;
2. validation des opportunités par le LLM (résultat mis en cache dans les
   métadonnées ChromaDB).
"""

from __future__ import annotations

from pathlib import Path

from loguru import logger

from config import get_settings
from ingest import ingest_path
from rag_engine import get_rag_opportunities

INVEST_DIR = Path("data/investissement")


def main() -> None:
    settings = get_settings()

    logger.info("Ingestion des fiches d'investissement depuis '{}'...", INVEST_DIR)
    files, chunks = ingest_path(
        INVEST_DIR,
        persist_dir=settings.CHROMA_PERSIST_DIR,
        collection_name=settings.CHROMA_COLLECTION,
        embedding_model=settings.EMBEDDING_MODEL,
        batch_size=settings.EMBEDDING_BATCH_SIZE,
        prune_missing=True,
    )
    logger.info("{} fichier(s), {} chunk(s) indexé(s).", files, chunks)

    logger.info("Validation LLM des opportunités...")
    opportunities = get_rag_opportunities(settings, limit=100)
    logger.info("{} opportunité(s) validée(s).", len(opportunities))
    for opportunity in opportunities:
        logger.info(
            "- {} | secteur : {} | région : {} | source : {}",
            opportunity["title"][:70],
            opportunity["sector"],
            opportunity.get("region") or "N/A",
            opportunity["source"],
        )


if __name__ == "__main__":
    main()
