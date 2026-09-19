"""Ingestion de documents juridiques dans une base vectorielle ChromaDB locale.

Usage :
    python ingest.py                      # ingère tous les documents de data/
    python ingest.py data/loi.pdf         # ingère un fichier précis
    python ingest.py --collection mre     # choisit une autre collection

Décisions d'architecture (production) :
    - Sources : PDF bruts OU .md nettoyés par clean_pdf.py (source canonique).
      Quand un .md du même nom existe à côté d'un .pdf, seul le .md est ingéré
      (pas de double ingestion) et la métadonnée "page" est reconstruite depuis
      les marqueurs <!-- PAGE N -->.
    - Extraction PDF -> Markdown via pymupdf4llm : la structure du document
      (titres, listes, tableaux) est préservée pour un meilleur RAG.
    - Nettoyage des artefacts PDF : numéros de page isolés, en-têtes/pieds de
      page répétitifs détectés par fréquence de lignes + regex.
    - Découpage en gros blocs cohérents (chunk_size=3000, overlap=300) pensé
      pour un LLM à très grand contexte : des blocs juridiques complets,
      pas de petits fragments.
    - Embeddings locaux BAAI/bge-m3 : multilingue FR + AR nativement
      (indispensable pour le droit marocain), aucun appel réseau pour l'embedding.
    - Persistance ChromaDB dans ./chroma_db, collection dédiée à l'immobilier MRE.
    - Métadonnées : source (nom du fichier), sector (filtre anti-hallucination
      côté retrieval), page (numéro de page pour les citations légales).
    - Idempotent : relancer le script met à jour les chunks d'un fichier déjà
      ingéré au lieu de dupliquer.
    - Logging riche (loguru) : chaque page en échec d'extraction est tracée.
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import Counter
from pathlib import Path
from typing import List, Tuple

import pymupdf4llm
from chromadb import PersistentClient
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from loguru import logger

from config import resolve_device

# ---------------------------------------------------------------------------
# Constantes par défaut (surchargeables en ligne de commande)
# ---------------------------------------------------------------------------
DEFAULT_DATA_DIR = Path("data")
DEFAULT_PERSIST_DIR = "chroma_db"
DEFAULT_COLLECTION = "immobilier_mre"
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

CHUNK_SIZE = 3000       # LLM à très grand contexte : gros blocs cohérents
CHUNK_OVERLAP = 300     # chevauchement pour ne pas couper une disposition en deux
BATCH_SIZE = 100        # taille des lots d'insertion ChromaDB (mémoire maîtrisée)
MIN_CHUNK_CHARS = 20    # ignore les résidus de mise en page (bruit)
SECTOR_LABEL = "immobilier"  # métadonnée utilisée comme filtre RAG anti-hallucination

# ---------------------------------------------------------------------------
# Artefacts PDF typiques (numéros de page, en-têtes/pieds)
# ---------------------------------------------------------------------------
_PAGE_NUMBER_RE = re.compile(r"^\s*\d{1,5}\s*$")                      # "3"
_PAGE_RANGE_RE = re.compile(r"^\s*\d{1,5}\s*/\s*\d{1,5}\s*$")         # "3/45"
_PAGE_LABEL_RE = re.compile(r"^\s*(?:page|p\.?|صفحة)\s*\d{1,5}\s*$", re.IGNORECASE)

# Marqueur de page écrit par clean_pdf.py : "<!-- PAGE 12 -->"
_PAGE_MARKER_RE = re.compile(r"<!--\s*PAGE\s+(\d+)\s*-->")

_SUPPORTED_EXTENSIONS = {".pdf", ".md"}

def _make_splitter(chunk_size: int, chunk_overlap: int) -> RecursiveCharacterTextSplitter:
    """Splitter construit sur mesure (taille de chunk surchargeable en CLI :
    un chunk plus gros = moins d'embeddings = ingestion plus rapide)."""
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
        is_separator_regex=False,
    )

_SPLITTER = _make_splitter(CHUNK_SIZE, CHUNK_OVERLAP)

# Enlève le handler stderr par défaut et configure un format lisible
logger.remove()
logger.add(
    sys.stderr,
    level="INFO",
    format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan> - {message}",
)


def clean_markdown(md_text: str, min_repetitions: int = 3) -> str:
    """Supprime les artefacts PDF récurrents du Markdown extrait.

    - Numéros de page isolés (regex).
    - En-têtes/pieds de page : toute ligne quasi identique présente
      >= min_repetitions fois dans le document (typique des en-têtes
      de lois officielles marocaines) est supprimée.
    - Les lignes Markdown structurelles (#, |, -, *) sont préservées :
      ce sont des titres, tableaux ou listes légitimes.
    """
    lines = md_text.splitlines()
    counts = Counter(line.strip() for line in lines if line.strip())
    repetitive = {
        text
        for text, count in counts.items()
        if count >= min_repetitions
        and len(text) <= 120
        and not text.startswith(("#", "|", "-", "*"))
    }

    cleaned: List[str] = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            cleaned.append("")
            continue
        if _PAGE_NUMBER_RE.fullmatch(stripped):
            continue
        if _PAGE_RANGE_RE.fullmatch(stripped):
            continue
        if _PAGE_LABEL_RE.fullmatch(stripped):
            continue
        if stripped in repetitive:
            continue
        cleaned.append(line)

    # Compresse les accumulations de lignes vides
    return re.sub(r"\n{3,}", "\n\n", "\n".join(cleaned)).strip()


def extract_pages_markdown(pdf_path: Path) -> List[Tuple[int, str]]:
    """Extrait chaque page du PDF en Markdown.

    Retourne une liste de tuples (numéro_page_1_basé, markdown).
    Chaque page est traitée individuellement : si une page échoue,
    elle est loggée en warning et l'ingestion continue (robustesse).
    """
    import fitz  # backend PyMuPDF utilisé par pymupdf4llm

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


def read_md_pages(md_path: Path) -> List[Tuple[int, str]]:
    """Lit un .md produit par clean_pdf.py : découpe sur les marqueurs <!-- PAGE N -->.

    Le préambule (avant le premier marqueur, ex. couverture) est attribué à la
    page 0 ; un .md sans marqueurs est traité comme un unique bloc page 0.
    """
    text = md_path.read_text(encoding="utf-8", errors="replace")
    parts = _PAGE_MARKER_RE.split(text)
    pages: List[Tuple[int, str]] = []

    preamble = parts[0].strip()
    if preamble:
        pages.append((0, preamble))
    for i in range(1, len(parts) - 1, 2):
        try:
            page_no = int(parts[i])
        except ValueError:
            continue
        content = parts[i + 1].strip()
        if content:
            pages.append((page_no, content))

    if not pages and text.strip():
        pages.append((0, text.strip()))

    logger.info("{} page(s) lue(s) depuis '{}'.", len(pages), md_path.name)
    return pages


def _sector_for_path(path: Path, data_root: Path) -> str:
    """Déduit le secteur RAG du dossier de premier niveau sous `data/`.

    data/immobilier/x.pdf -> "immobilier" ; data/finance/x.md -> "finance" ;
    un fichier à la racine de `data/` (ex. loi_immo.pdf) -> SECTOR_LABEL (immobilier).
    """
    try:
        rel = path.resolve().relative_to(data_root.resolve())
    except ValueError:
        return SECTOR_LABEL
    if len(rel.parts) >= 2:
        return rel.parts[0]
    return SECTOR_LABEL


def chunk_page(page_no: int, md_text: str, source: str, sector: str,
               splitter: RecursiveCharacterTextSplitter | None = None) -> List[Tuple[str, dict]]:
    """Découpe le Markdown nettoyé d'une page en gros chunks cohérents."""
    splitter = splitter or _SPLITTER
    chunks = splitter.split_text(md_text)
    items: List[Tuple[str, dict]] = []
    for idx, chunk in enumerate(chunks):
        text = chunk.strip()
        if len(text) < MIN_CHUNK_CHARS:
            continue
        metadata = {
            "source": source,          # nom du fichier PDF (requis)
            "sector": sector,          # filtre de retrieval anti-hallucination
            "page": page_no,
            "chunk_index": idx,
        }
        items.append((text, metadata))
    return items


def ingest_document(path: Path, collection, embeddings: HuggingFaceEmbeddings,
                    sector: str = SECTOR_LABEL,
                    splitter: RecursiveCharacterTextSplitter | None = None,
                    skip_existing: bool = False) -> int:
    """Ingère un document (.pdf brut ou .md nettoyé) : extraction -> nettoyage -> chunks.

    Le nom de source retenu est le nom du fichier tel quel (ex. "loi_immo.pdf")
    pour les PDFs, et le nom sans extension pour les .md (ex. "IGOC 2024") :
    c'est ce nom qui s'affiche dans les citations côté frontend.
    """
    source = path.stem if path.suffix.lower() == ".md" else path.name
    if skip_existing:
        existing = collection.get(where={"source": source}, limit=1)
        if existing and existing.get("ids"):
            logger.info("Source déjà présente, fichier ignoré : '{}'.", path.name)
            return 0

    if path.suffix.lower() == ".md":
        logger.info("=== Ingestion (Markdown nettoyé) : {} ===", path.name)
        pages = read_md_pages(path)
    else:
        logger.info("=== Ingestion : {} ===", path.name)
        pages = extract_pages_markdown(path)

    if not pages:
        logger.error("Aucune page exploitable pour '{}' — fichier ignoré.", path.name)
        return 0

    items: List[Tuple[str, dict]] = []
    for page_no, md in pages:
        cleaned = clean_markdown(md)
        if not cleaned:
            logger.warning("Page {} de '{}' : contenu vide après nettoyage.", page_no, path.name)
            continue
        items.extend(chunk_page(page_no, cleaned, source, sector, splitter=splitter))

    if not items:
        logger.warning("Aucun chunk généré pour '{}' après nettoyage.", path.name)
        return 0

    # Idempotence : on supprime les chunks déjà présents pour ce fichier
    existing = collection.get(where={"source": source})
    if existing and existing.get("ids") and not skip_existing:
        collection.delete(ids=existing["ids"])
        logger.info("{} ancien(s) chunk(s) supprimé(s) pour '{}'.", len(existing["ids"]), source)

    # Migration PDF -> .md : si ce document était auparavant ingéré depuis son
    # PDF brut (source "<nom>.pdf"), on purge aussi ces anciens chunks pour
    # éviter la double ingestion de la même loi.
    if path.suffix.lower() == ".md" and not skip_existing:
        legacy_source = f"{source}.pdf"
        legacy = collection.get(where={"source": legacy_source})
        if legacy and legacy.get("ids"):
            collection.delete(ids=legacy["ids"])
            logger.info(
                "{} ancien(s) chunk(s) de '{}' (version PDF brut) supprimé(s).",
                len(legacy["ids"]),
                legacy_source,
            )

    inserted = 0
    for start in range(0, len(items), BATCH_SIZE):
        batch = items[start : start + BATCH_SIZE]
        docs = [text for text, _ in batch]
        metas = [meta for _, meta in batch]
        ids = [f"{source}__p{meta['page']}_{meta['chunk_index']}" for _, meta in batch]

        vectors = embeddings.embed_documents(docs)
        collection.add(ids=ids, documents=docs, metadatas=metas, embeddings=vectors)

        inserted += len(batch)
        logger.info("Lot {} -> {} : {} chunk(s) inséré(s) (cumul : {}).",
                    start + 1, start + len(batch), len(batch), inserted)

    logger.success("Terminé : {} chunk(s) pour '{}'.", inserted, path.name)
    return inserted


def collect_sources(paths: List[Path]) -> List[Path]:
    """Résout la liste des fichiers à ingérer : PDF bruts et .md nettoyés.

    Préférence aux versions propres : quand un .md du même nom existe à côté
    d'un .pdf dans le même dossier, seul le .md est ingéré (pas de doublon).
    """
    files: List[Path] = []
    for raw in paths:
        p = raw.expanduser().resolve()
        if p.is_dir():
            files.extend(sorted(f for f in p.rglob("*") if f.suffix.lower() in _SUPPORTED_EXTENSIONS))
        elif p.is_file() and p.suffix.lower() in _SUPPORTED_EXTENSIONS:
            files.append(p)
        else:
            logger.warning("Chemin ignoré (introuvable ou format non supporté) : {}", raw)

    selected: List[Path] = []
    by_dir: dict = {}
    for f in files:
        by_dir.setdefault(f.parent, []).append(f)

    for parent, group in by_dir.items():
        md_stems = {f.stem for f in group if f.suffix.lower() == ".md"}
        for f in sorted(group):
            if f.suffix.lower() == ".pdf" and f.stem in md_stems:
                logger.info("Version nettoyée disponible, PDF brut ignoré : {}", f.name)
                continue
            selected.append(f)

    # Déduplication des basenames entre dossiers (ex: data/loi.pdf et
    # data/immobilier/loi.pdf) : le nom de fichier sert de clé `source`
    # (et donc d'id ChromaDB) — deux copies produiraient des ids identiques.
    # On n'ingère donc qu'une seule copie de chaque basename.
    deduped: List[Path] = []
    seen_names: set = set()
    for f in sorted(selected, key=lambda x: x.name):
        if f.name in seen_names:
            logger.warning("Doublon ignoré (même nom de fichier déjà planifié) : {}", f)
            continue
        seen_names.add(f.name)
        deduped.append(f)
    return deduped


def ingest_path(path: Path, persist_dir: str = DEFAULT_PERSIST_DIR,
                collection_name: str = DEFAULT_COLLECTION,
                embedding_model: str = DEFAULT_EMBEDDING_MODEL,
                batch_size: int = BATCH_SIZE,
                prune_missing: bool = False) -> Tuple[int, int]:
    """Ingère un fichier ou dossier depuis une application hôte."""
    sources = collect_sources([path])
    if not sources:
        return 0, 0
    data_root = path.expanduser().resolve().parent
    embeddings = HuggingFaceEmbeddings(
        model_name=embedding_model,
        model_kwargs={"device": resolve_device()},
        encode_kwargs={"normalize_embeddings": True, "batch_size": batch_size},
    )
    client = PersistentClient(path=persist_dir)
    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )
    # Pour un dossier sectoriel explicite (ex. data/investissement), supprimer
    # les sources de ce secteur dont le fichier n'existe plus. Cette option
    # reste désactivée par défaut pour ne pas purger les autres secteurs lors
    # d'une ingestion globale de data/.
    if prune_missing and path.is_dir():
        sector_name = path.resolve().name
        current_sources = {
            source.stem if source.suffix.lower() == ".md" else source.name
            for source in sources
        }
        indexed = collection.get(where={"sector": sector_name}, include=["metadatas"])
        stale_ids = [
            record_id
            for record_id, metadata in zip(indexed.get("ids") or [], indexed.get("metadatas") or [])
            if str((metadata or {}).get("source") or "") not in current_sources
        ]
        if stale_ids:
            collection.delete(ids=stale_ids)
            logger.info("{} chunk(s) obsolète(s) supprimé(s) du secteur '{}'.", len(stale_ids), sector_name)
    splitter = _make_splitter(CHUNK_SIZE, CHUNK_OVERLAP)
    ingested_files = 0
    total_chunks = 0
    for source in sources:
        source_name = source.stem if source.suffix.lower() == ".md" else source.name
        sector = _sector_for_path(source, data_root)
        total_chunks += ingest_document(
            source, collection, embeddings, sector=sector, splitter=splitter,
            # Les fichiers existants sont remplacés afin de refléter les
            # corrections du contenu et de réinitialiser leur validation LLM.
            skip_existing=False,
        )
        ingested_files += 1
    return ingested_files, total_chunks


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ingestion de documents juridiques (droit immobilier marocain) dans ChromaDB.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "paths",
        nargs="*",
        type=Path,
        default=[DEFAULT_DATA_DIR],
        help="Fichiers PDF/.md ou dossiers à ingérer (défaut : data/)",
    )
    parser.add_argument("--persist-dir", default=DEFAULT_PERSIST_DIR, help="Dossier de persistance ChromaDB")
    parser.add_argument("--collection", default=DEFAULT_COLLECTION, help="Nom de la collection ChromaDB")
    parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL, help="Modèle d'embedding HF")
    parser.add_argument("--batch-size", type=int, default=64,
                        help="Taille de lot d'embedding (CPU : 64-128 accélère nettement)")
    parser.add_argument("--threads", type=int, default=0,
                        help="Nombre de threads torch (0 = tous les cœurs)")
    parser.add_argument("--chunk-size", type=int, default=CHUNK_SIZE,
                        help="Taille max des chunks (plus gros = moins d'embeddings = plus rapide)")
    parser.add_argument("--chunk-overlap", type=int, default=CHUNK_OVERLAP,
                        help="Chevauchement entre chunks")
    args = parser.parse_args()

    if args.threads > 0:
        import torch
        torch.set_num_threads(args.threads)
        logger.info("Threads torch fixés à {}.", args.threads)

    splitter = _make_splitter(args.chunk_size, args.chunk_overlap)
    logger.info("Splitter : chunk_size={}, overlap={} (moins de chunks = ingestion plus rapide).",
                args.chunk_size, args.chunk_overlap)

    sources = collect_sources(args.paths)
    if not sources:
        logger.error("Aucun document trouvé dans les chemins fournis. Abandon.")
        sys.exit(1)

    # Racine de données = premier chemin fourni (dossier, sinon le parent du fichier) :
    # sert à déduire le secteur de chaque document (data/<secteur>/...).
    base_roots: List[Path] = []
    for raw in args.paths:
        r = raw.expanduser().resolve()
        base_roots.append(r if r.is_dir() else r.parent)
    data_root = base_roots[0] if base_roots else DEFAULT_DATA_DIR.resolve()

    logger.info("Chargement du modèle d'embedding '{}' (FR/AR)...", args.embedding_model)
    embeddings = HuggingFaceEmbeddings(
        model_name=args.embedding_model,
        model_kwargs={"device": resolve_device()},
        encode_kwargs={"normalize_embeddings": True, "batch_size": args.batch_size},
    )

    logger.info("Initialisation ChromaDB persistante dans '{}'...", args.persist_dir)
    client = PersistentClient(path=args.persist_dir)
    collection = client.get_or_create_collection(
        name=args.collection,
        metadata={"hnsw:space": "cosine"},
    )
    logger.info("Collection '{}' prête ({} chunk(s) avant ingestion).",
                args.collection, collection.count())

    total_files = 0
    total_chunks = 0
    for doc in sources:
        try:
            sector = _sector_for_path(doc, data_root)
            total_chunks += ingest_document(doc, collection, embeddings, sector=sector, splitter=splitter)
            total_files += 1
        except Exception:
            logger.exception("Échec global de l'ingestion pour '{}'", doc)

    logger.info("=== RÉSUMÉ : {} fichier(s) ingéré(s), {} chunk(s) ajouté(s) à '{}' ===",
                total_files, total_chunks, args.collection)
    logger.info("La collection contient désormais {} chunk(s).", collection.count())


if __name__ == "__main__":
    main()
