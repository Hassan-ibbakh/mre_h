#!/usr/bin/env python
"""Supprime l'index ChromaDB local après confirmation explicite."""

import argparse
import shutil
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Supprime l'index ChromaDB local.")
    parser.add_argument("--yes", action="store_true", help="Confirme la suppression irréversible.")
    args = parser.parse_args()
    chroma_dir = Path("chroma_db").resolve()

    if not args.yes:
        parser.error("Ajoutez --yes pour confirmer la suppression de chroma_db.")
    if chroma_dir.is_dir():
        print(f"Suppression de {chroma_dir}...")
        shutil.rmtree(chroma_dir)
        print("ChromaDB nettoyée")
    else:
        print("ChromaDB déjà vide")


if __name__ == "__main__":
    main()
