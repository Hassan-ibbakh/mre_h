"""Prétraitement des PDFs juridiques bruts -> fichiers Markdown propres.

Usage :
    python clean_pdf.py                        # traite tous les PDFs de data/
    python clean_pdf.py data/IGOC\\ 2024.pdf   # un fichier précis
    python clean_pdf.py data/raw --out data    # un dossier d'entrée, sortie ailleurs

Pipeline :
    1. Extraction page par page avec pymupdf4llm (Markdown structuré).
    2. Nettoyage massif par regex + analyse de fréquence des lignes SUR TOUT
       LE DOCUMENT (une ligne quasi identique présente sur au moins la moitié
       des pages est un en-tête/pied de page ou un timbre officiel) :
       - numéros de page isolés, plages "3/45", libellés "Page N" (FR + AR) ;
       - en-têtes et pieds de page répétitifs, pages blanches déclarées ;
       - tables des matières / index : lignes à pointillés, titres isolés
         ("Sommaire", "Table des matières", "المحتوى", ...) ;
       - règles décoratives (-, =, *, ...) ;
       - caractères invisibles RTL/Unicode (marqueurs de direction).
    3. Écriture de data/<nom>.md avec des marqueurs <!-- PAGE N --> :
       ingest.py conserve ainsi la métadonnée "page" pour les citations légales.

Le fichier .md devient la source canonique : ingest.py ignore le .pdf d'origine
dès qu'un .md nettoyé du même nom existe dans le même dossier.
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import Counter
from pathlib import Path
from typing import List, Optional, Tuple

import fitz
import pymupdf4llm
from loguru import logger

DEFAULT_DATA_DIR = Path("data")
DEFAULT_MIN_REPETITIONS = 3

# Caractères invisibles fréquents dans les PDFs arabophones (marqueurs de direction, BOM)
_INVISIBLE = str.maketrans({"\u200e": "", "\u200f": "", "\ufeff": "", "\u200b": ""})

# --- Regex de suppression du bruit -------------------------------------------------
# Numéros de page isolés : "3", "3.", "- 12 -", "*45*"
_PAGE_NUMBER_RE = re.compile(r"^\s*[\*\.·\-\s]*\d{1,5}[\*\.·\-\s]*\s*$")
# Plages : "3/45"
_PAGE_RANGE_RE = re.compile(r"^\s*\d{1,5}\s*/\s*\d{1,5}\s*$")
# Libellés : "Page 3", "p. 12", "Page 3 sur 12", "صفحة 3"
_PAGE_LABEL_RE = re.compile(
    r"^\s*(?:page|p\.?|صفحة|ص)\s*\d{1,5}\s*(?:sur|de|/\s*\d{1,5})?\s*$",
    re.IGNORECASE,
)
# Pages blanches déclarées
_BLANK_PAGE_RE = re.compile(
    r"^\s*(?:page\s+blanche|this\s+page\s+intentionally\s+left\s+blank)\s*$",
    re.IGNORECASE,
)
# Règles décoratives : "-------", "=====", "****"
_RULE_RE = re.compile(r"^\s*[-—_=*·•#]{3,}\s*$")
# Puces vides laissées par l'extraction : "- ", "*", "•"
_EMPTY_BULLET_RE = re.compile(r"^\s*[-*•·]\s*$")
# Artefacts d'OCR d'images : "<!-- Start of picture text -->"
_PICTURE_TEXT_RE = re.compile(
    r"^\s*<!--\s*(?:start|end)\s+of\s+picture\s+text\s*-->\s*$",
    re.IGNORECASE,
)
# Lignes de table des matières : "1. Dispositions générales .......... 3"
_DOTTED_TOC_RE = re.compile(
    r"^\s*[\dIVXLC]+[.)]?\s+[\wÀ-ÿ\u0600-\u06FF][^.]*?\.{3,}\s*\d{1,4}\s*$"
)
# Titres d'index isolés sur une ligne
_TOC_HEADING_RE = re.compile(
    r"^\s*(?:sommaire|table\s+des\s+mati[eè]res|index|plan|المحتوى|الفهرس)\s*:?\.?\s*$",
    re.IGNORECASE,
)


def _norm(line: str) -> str:
    """Normalise une ligne : caractères invisibles retirés, espaces latéraux nettoyés."""
    return line.translate(_INVISIBLE).strip()


def collect_pdfs(paths: List[Path]) -> List[Path]:
    """Résout la liste des PDFs : fichiers directs ou dossiers parcourus récursivement."""
    pdfs: List[Path] = []
    for raw in paths:
        p = raw.expanduser().resolve()
        if p.is_dir():
            pdfs.extend(sorted(p.rglob("*.pdf")))
        elif p.is_file() and p.suffix.lower() == ".pdf":
            pdfs.append(p)
        else:
            logger.warning("Chemin ignoré (introuvable ou non-PDF) : {}", raw)
    return pdfs


def extract_pages(pdf_path: Path) -> List[Tuple[int, str]]:
    """Extrait chaque page du PDF en Markdown (échec d'une page tracé, non bloquant)."""
    try:
        doc = fitz.open(str(pdf_path))
    except Exception as exc:
        logger.error("Impossible d'ouvrir le PDF {} : {}", pdf_path, exc)
        return []

    total = doc.page_count
    logger.info("Extraction de {} page(s) depuis '{}'...", total, pdf_path.name)
    pages: List[Tuple[int, str]] = []
    failures = 0

    for page_no in range(total):
        try:
            md = pymupdf4llm.to_markdown(doc, pages=[page_no], page_numbers=False)
        except Exception as exc:
            failures += 1
            logger.warning(
                "Page {}/{} de '{}' EN ÉCHEC : {}",
                page_no + 1,
                total,
                pdf_path.name,
                exc,
            )
            continue

        if md and md.strip():
            pages.append((page_no + 1, md.strip()))
        else:
            failures += 1
            logger.warning(
                "Page {}/{} de '{}' : aucune donnée textuelle extraite (page scannée ?)",
                page_no + 1,
                total,
                pdf_path.name,
            )

    doc.close()
    if failures:
        logger.warning("Résumé : {}/{} pages en échec pour '{}'", failures, total, pdf_path.name)
    else:
        logger.info("{} page(s) extraites sans échec pour '{}'", total, pdf_path.name)
    return pages


def _keep_line(line: str, counts: Counter, threshold: int, stats: Counter) -> bool:
    """Décide si une ligne est du bruit (False = à supprimer)."""
    line = _norm(line)
    if not line:
        return True  # lignes vides conservées comme séparateurs

    if _PAGE_NUMBER_RE.fullmatch(line) or _PAGE_RANGE_RE.fullmatch(line) or _PAGE_LABEL_RE.fullmatch(line):
        stats["numeros_de_page"] += 1
        return False
    if _BLANK_PAGE_RE.fullmatch(line):
        stats["pages_blanches"] += 1
        return False
    if _RULE_RE.fullmatch(line):
        stats["regles_decoratives"] += 1
        return False
    if _EMPTY_BULLET_RE.fullmatch(line):
        stats["puces_vides"] += 1
        return False
    if _PICTURE_TEXT_RE.fullmatch(line):
        stats["artefacts_ocr"] += 1
        return False
    if _TOC_HEADING_RE.fullmatch(line) or _DOTTED_TOC_RE.fullmatch(line):
        stats["index"] += 1
        return False

    # En-têtes / pieds / timbres officiels : ligne répétée sur >= la moitié des pages.
    # Les lignes structurelles Markdown sont préservées (titres #, tableaux |,
    # citations >, listes "- " / "* ") — mais PAS le gras "**...**" : un en-tête
    # répété en gras doit rester détectable.
    if (
        counts.get(line, 0) >= threshold
        and len(line) <= 120
        and not line.startswith(("#", "|", ">", "- ", "* "))
        and any(ch.isalpha() for ch in line)
    ):
        stats["entetes_pieds_repetitifs"] += 1
        return False

    return True


def clean_document(pages: List[Tuple[int, str]], min_repetitions: int) -> Tuple[List[Tuple[int, str]], Counter]:
    """Nettoie tout le document et comptabilise le bruit supprimé par catégorie.

    La détection par fréquence porte sur le document COMPLET : un en-tête répété
    une fois par page ne dépasse jamais un seuil de fréquence si on nettoie page
    par page, d'où l'analyse globale ici.
    """
    n_pages = max(1, len(pages))
    threshold = max(min_repetitions, (n_pages + 1) // 2)

    all_lines: List[str] = []
    for _, md in pages:
        all_lines.extend(md.splitlines())

    counts = Counter(_norm(line) for line in all_lines if _norm(line))

    stats: Counter = Counter()
    cleaned: List[Tuple[int, str]] = []
    for page_no, md in pages:
        kept: List[str] = []
        for line in md.splitlines():
            if _keep_line(line, counts, threshold, stats):
                kept.append(line)
        text = re.sub(r"\n{3,}", "\n\n", "\n".join(kept)).strip()
        if text:
            cleaned.append((page_no, text))
    return cleaned, stats


def write_clean_md(pdf_path: Path, pages: List[Tuple[int, str]], out_dir: Path) -> Optional[Path]:
    """Écrit data/<nom>.md : chaque page précédée du marqueur <!-- PAGE N -->."""
    out_path = out_dir / f"{pdf_path.stem}.md"
    if not pages:
        logger.error("Aucun contenu propre pour '{}' — fichier .md non écrit.", pdf_path.name)
        return None
    blocks = [f"<!-- PAGE {page_no} -->\n{md}" for page_no, md in pages]
    out_path.write_text("\n\n".join(blocks) + "\n", encoding="utf-8")
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prétraitement des PDFs juridiques bruts vers des .md propres (nettoyage regex).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "paths",
        nargs="*",
        type=Path,
        default=[DEFAULT_DATA_DIR],
        help="Fichiers PDF ou dossiers à nettoyer (défaut : data/)",
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_DATA_DIR, help="Dossier de sortie des .md (défaut : data/)")
    parser.add_argument(
        "--min-repetitions",
        type=int,
        default=DEFAULT_MIN_REPETITIONS,
        help="Seuil de fréquence minimal pour considérer une ligne comme en-tête/pied répétitif",
    )
    args = parser.parse_args()

    out_dir = args.out.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    pdfs = collect_pdfs(args.paths)
    if not pdfs:
        logger.error("Aucun PDF trouvé dans les chemins fournis. Abandon.")
        sys.exit(1)

    for pdf in pdfs:
        try:
            pages = extract_pages(pdf)
            cleaned, stats = clean_document(pages, args.min_repetitions)
            out_path = write_clean_md(pdf, cleaned, out_dir)
            if out_path is None:
                continue
            chars_in = sum(len(md) for _, md in pages)
            chars_out = sum(len(md) for _, md in cleaned)
            logger.success(
                "'{}' -> '{}' : {} -> {} caractères ({} page(s) conservée(s))",
                pdf.name,
                out_path.name,
                chars_in,
                chars_out,
                len(cleaned),
            )
            for noise_type, count in stats.items():
                logger.info("    - {} supprimés : {}", noise_type, count)
        except Exception:
            logger.exception("Échec global du prétraitement pour '{}'", pdf)


if __name__ == "__main__":
    main()
