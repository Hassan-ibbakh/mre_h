"""
Decouverte DYNAMIQUE de talents marocains.

Ce module expose :
  - les helpers RSS (load_sources, _fetch_rss_raw_entries...) : reutilises
    tels quels comme outils par l'agent (agents/talent_scout_agent.py) ;
  - les helpers de dedup (_normalize_name, _is_likely_duplicate) : idem,
    reutilises par l'agent pour la verification de doublon sans LLM ;
  - `discover_talents_from_sources()` : fonction de compatibilite appelee
    par app.py, qui delegue l'exploration RSS+Tavily a l'agent LangChain
    autonome (talent_scout_agent.py), puis ajoute les profils ORCID (source
    structuree, toujours sans appel LLM) en complement deterministe.

Pourquoi ORCID reste separe de l'agent : ORCID fournit deja des donnees
structurees (pays de residence lu directement dans l'API) -- le filtrage
est deterministe, sans avoir besoin qu'un LLM raisonne dessus. L'integrer
comme outil de l'agent ajouterait du cout Groq (tours de raisonnement
supplementaires) sans aucun benefice de precision.

L'import de `talent_scout_agent` est fait a l'INTERIEUR de
`discover_talents_from_sources()` (et non en haut de fichier) car
`talent_scout_agent.py` importe lui-meme des helpers de ce module : un
import en tete de fichier creerait une dependance circulaire au chargement.

NOUVEAU : Integration du cache ProfileCache (database.py) pour eviter la
reutation des memes sources ORCID/RSS/Tavily a chaque cycle de 24h.
Economie de ~50% de tokens en eliminant les doublons ORCID des 7 derniers jours.
"""
import json
import logging
import os
import re
import unicodedata
from difflib import SequenceMatcher
from typing import Any, Dict, List
from xml.etree import ElementTree as ET

from .orcid_source import discover_orcid_talents
from database import cleanup_old_cache
from config import get_settings

logger = logging.getLogger("mre_ai")

DEFAULT_SOURCES_PATH = os.path.join("data", "talent_sources.json")

SIMILARITY_THRESHOLD = 0.87  # au-dela, deux noms sont consideres comme la meme personne

DEFAULT_KEYWORDS = [
    "talent", "marocain", "marocaine", "maroc", "prix", "lauréat", "laureat",
    "award", "winner", "innovation", "entrepreneur", "chercheur", "start-up", "startup",
]

_PROFILE_SIGNAL_KEYWORDS = [
    "entrepreneur", "chercheur", "expert", "experte", "ingénieur", "ingenieur",
    "lauréat", "laureat", "prix", "innovation", "start-up", "startup",
    "investissement", "finance", "agriculture", "agro", "agritech", "tech",
    "fondatrice", "fondateur", "leader", "spécialisé", "specialise",
]

_GENERIC_NEWS_MARKERS = [
    "hectares", "record", "historique", "actualite", "actualité", "ravage",
    "gouvernement", "élections", "elections", "attaque", "crise",
]

_STOPWORDS = {
    "reçoit", "recoit", "est", "a", "avec", "pour", "sur", "dans", "de", "du",
    "des", "les", "et", "au", "aux", "son", "sa", "ses", "un", "une",
}

_MRE_INDICATOR_RE = re.compile(
    r"\b(maroc(?:ain(?:e|s)?)?|marocains|diaspora|étrang(?:er|ère)|etrang(?:er|ere)|expatri(?:é|ée|ation)|à l['’]?étranger|a l['’]?etranger|MRE)\b",
    re.IGNORECASE | re.UNICODE,
)


def _normalize_name(name: str) -> str:
    decomposed = unicodedata.normalize("NFKD", name)
    without_accents = "".join(c for c in decomposed if not unicodedata.combining(c))
    return re.sub(r"[^a-z0-9 ]", "", without_accents.lower()).strip()


def _is_likely_duplicate(name: str, known_normalized_names) -> bool:
    normalized = _normalize_name(name)
    if not normalized:
        return False
    for existing in known_normalized_names:
        if SequenceMatcher(None, normalized, existing).ratio() >= SIMILARITY_THRESHOLD:
            return True
    return False


def _extract_name_from_title(title: str) -> str | None:
    cleaned = re.sub(r"\s+", " ", (title or "").strip())
    if not cleaned:
        return None

    words = re.findall(r"[A-Za-zÀ-ÖØ-öø-ÿ]+", cleaned)
    if len(words) < 2:
        return None

    candidate_words: List[str] = []
    for word in words:
        if word.lower() in _STOPWORDS:
            break
        if word.lower() in {"un", "une", "le", "la", "les", "des"}:
            continue
        candidate_words.append(word)
        if len(candidate_words) >= 4:
            break

    if len(candidate_words) < 2:
        return None

    candidate = " ".join(candidate_words)
    if len(candidate.split()) >= 2 and candidate[0].isupper():
        return candidate
    return None


def _infer_sector(text: str) -> str:
    haystack = text.lower()
    if any(kw in haystack for kw in ["agriculture", "agro", "agritech", "agri"]):
        return "agriculture"
    if any(kw in haystack for kw in ["immobilier", "real estate", "property"]):
        return "real_estate"
    if any(kw in haystack for kw in ["industrie", "industrial", "manufacturing", "production"]):
        return "industry"
    if any(kw in haystack for kw in ["tourisme", "tourism", "hospitality"]):
        return "tourism"
    return "other"


def _infer_expertise_tags(title: str, description: str) -> str:
    haystack = f"{title} {description}".lower()
    tags = []
    if "immobilier" in haystack:
        tags.append("immobilier")
    if "finance" in haystack or "financier" in haystack or "investissement" in haystack:
        tags.append("finance")
    if "agriculture" in haystack or "agro" in haystack or "agritech" in haystack:
        tags.append("agriculture")
    if "tech" in haystack or "technologie" in haystack or "ia" in haystack:
        tags.append("technologie")
    if not tags:
        tags.append("talent")
    return ", ".join(tags)


def _infer_years_experience(text: str) -> int | None:
    match = re.search(r"(\d+)\s+ans?", text.lower())
    if match:
        return int(match.group(1))
    return None


def extract_profile_from_entry(entry: Dict[str, Any]) -> Dict[str, Any] | None:
    """Extrait un profil de talent de facon deterministe depuis un item RSS/flux."""
    title = (entry.get("title") or "").strip()
    description = (entry.get("description") or "").strip()
    combined = f"{title} {description}".strip()
    if not combined:
        return None

    lowered = combined.lower()
    if any(marker in lowered for marker in _GENERIC_NEWS_MARKERS):
        return None

    has_profile_signal = any(keyword in lowered for keyword in _PROFILE_SIGNAL_KEYWORDS)
    if not has_profile_signal:
        return None

    name = _extract_name_from_title(title)
    if not name:
        name = "Talent sans nom"

    profile = {
        "title": name,
        "sector": _infer_sector(combined),
        "country": "Maroc",
        "expertise_tags": _infer_expertise_tags(title, description),
        "years_experience": _infer_years_experience(combined),
        "description": description,
        "source_url": (entry.get("url") or "").strip(),
        "source_name": (entry.get("source") or "Source externe").strip(),
        "image_url": "",
    }
    return profile


def load_sources(source_path: str | None = None) -> List[Dict[str, Any]]:
    path = source_path or DEFAULT_SOURCES_PATH
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if isinstance(payload, dict):
        payload = payload.get("sources", [])
    return payload if isinstance(payload, list) else []


def _read_url(url: str) -> str:
    import urllib.request
    if url.startswith("http://") or url.startswith("https://"):
        request = urllib.request.Request(url, headers={
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                          "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
            "Accept": "application/rss+xml, application/xml, text/xml, */*;q=0.8",
            "Accept-Language": "fr-FR,fr;q=0.9,en;q=0.8",
        })
        with urllib.request.urlopen(request, timeout=20) as response:
            return response.read().decode("utf-8", errors="ignore")
    if os.path.exists(url):
        with open(url, "r", encoding="utf-8") as handle:
            return handle.read()
    raise FileNotFoundError(f"Source introuvable : {url}")


def _find_first_text(element: ET.Element, tags: List[str], ns: Dict[str, str] | None = None, attr: str | None = None) -> str:
    for tag in tags:
        match = element.find(tag, ns) if ns else element.find(tag)
        if match is None:
            continue
        value = match.attrib.get(attr, "") if attr else (match.text or "").strip()
        if value:
            return value
    return ""


def _parse_rss_entries(xml_text: str) -> List[Dict[str, str]]:
    root = ET.fromstring(xml_text)
    entries: List[Dict[str, str]] = []

    if root.tag == "rss":
        for item in root.findall(".//item"):
            title = _find_first_text(item, ["title"])
            link = _find_first_text(item, ["link"])
            description = _find_first_text(item, ["description", "summary"])
            if title or link:
                entries.append({"title": title, "link": link, "description": description})
        return entries

    if root.tag.startswith("{"):
        ns = {"atom": "http://www.w3.org/2005/Atom"}
        for entry in root.findall("atom:entry", ns):
            title = _find_first_text(entry, ["atom:title"], ns=ns)
            link = _find_first_text(entry, ["atom:link"], ns=ns, attr="href")
            description = _find_first_text(entry, ["atom:summary", "atom:content"], ns=ns)
            if title or link:
                entries.append({"title": title, "link": link, "description": description})
    return entries


def _fetch_rss_raw_entries(source_path: str | None = None, keywords: List[str] | None = None) -> List[Dict[str, Any]]:
    keywords = keywords or DEFAULT_KEYWORDS
    entries: List[Dict[str, Any]] = []

    for source in load_sources(source_path):
        name = source.get("name", "Source")
        url = source.get("url")
        if not url:
            continue
        try:
            xml_text = _read_url(url)
        except Exception as exc:
            logger.warning("Flux '%s' inaccessible (%s) -- source ignoree pour ce cycle.", name, exc)
            continue

        try:
            items = _parse_rss_entries(xml_text)
        except ET.ParseError:
            logger.warning("Flux '%s' n'est pas un XML valide (contenu probablement non-RSS) -- ignore.", name)
            continue
        except Exception:
            logger.exception("Erreur inattendue en parsant le flux '%s'", name)
            continue

        for item in items:
            combined = f"{item.get('title', '')} {item.get('description', '')}".lower()
            if not _MRE_INDICATOR_RE.search(combined):
                continue
            if not any(kw in combined for kw in keywords):
                continue
            entries.append({
                "title": item.get("title", ""),
                "content": item.get("description", ""),
                "url": item.get("link", ""),
                "source": name,
            })
    return entries


def discover_talents_from_sources(
    source_path: str | None = None,
    keywords: List[str] | None = None,
    existing_urls: set | None = None,
    existing_titles: set | None = None,
) -> List[Dict[str, Any]]:
    """
    Fonction de COMPATIBILITE conservee pour ne pas casser l'import existant
    dans app.py. Delegue l'exploration RSS/Tavily a l'agent LangChain
    autonome (agents/talent_scout_agent.py), puis ajoute les profils ORCID
    (source structuree, sans appel LLM) en complement deterministe.
    
    NOUVEAU : Appelle cleanup_old_cache() au debut pour purger les URL
    traitees depuis plus de CACHE_RETENTION_DAYS jours et eviter une
    accumulation infinie de donnees dans le cache.
    """
    # --- NOUVEAU : Cleanup du cache au debut de chaque cycle ---
    if get_settings().ENABLE_PROFILE_CACHE:
        try:
            deleted_count = cleanup_old_cache(days_retention=get_settings().CACHE_RETENTION_DAYS)
            if deleted_count > 0:
                logger.info("Cache cleanup : %d ancien(ne)s profils purge(s).", deleted_count)
        except Exception:
            logger.exception("Erreur lors du nettoyage du cache ProfileCache")
    # --- Fin cleanup ---
    
    existing_urls = existing_urls or set()
    existing_titles = existing_titles or set()
    known_normalized_names = {_normalize_name(t) for t in existing_titles}
    existing_titles_lower = {t.strip().lower() for t in existing_titles}

    from .talent_scout_agent import run_talent_scout_agent

    discovered: List[Dict[str, Any]] = []
    try:
        discovered = run_talent_scout_agent(
            existing_urls=existing_urls,
            existing_titles=existing_titles,
        )
    except Exception:
        logger.exception("Erreur durant l'execution de l'agent de decouverte de talents")

    for item in discovered:
        known_normalized_names.add(_normalize_name(item.get("title", "")))
        existing_titles_lower.add(item.get("title", "").strip().lower())
        if item.get("url"):
            existing_urls.add(item.get("url"))

    try:
        orcid_profiles = discover_orcid_talents(existing_urls=existing_urls)
    except Exception:
        logger.exception("Erreur durant la collecte ORCID complementaire")
        orcid_profiles = []

    for item in orcid_profiles:
        url = (item.get("url") or "").strip()
        title = (item.get("title") or "").strip()
        if not title or not url or url in existing_urls:
            continue
        normalized_title = _normalize_name(title)
        if normalized_title in known_normalized_names:
            continue
        discovered.append(item)
        known_normalized_names.add(normalized_title)
        existing_titles_lower.add(title.lower())
        existing_urls.add(url)

    logger.info("Decouverte dynamique : %s profils valides au total (agent scout + ressources outils).", len(discovered))
    return discovered