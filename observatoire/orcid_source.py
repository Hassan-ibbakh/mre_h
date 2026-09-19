"""
Source de decouverte de talents marocains via l'API publique ORCID.
(fichier de test -- copie exacte fournie par l'utilisateur)
"""
import logging
import time
import unicodedata
from typing import Any, Dict, List, Optional

import requests

from database import cache_profile, get_cached_profile_status
from config import get_settings

logger = logging.getLogger("mre_ai")

ORCID_API_BASE = "https://pub.orcid.org/v3.0"
MOROCCO_ISO = "MA"

DOMAIN_QUERIES: Dict[str, str] = {
    "health": '((text:Moroccan) AND (keyword:health))',
    "education": '((text:Moroccan) AND (keyword:education))',
    "tech": '((text:Moroccan) AND (keyword:tech))',
    "agriculture": '((text:Moroccan) AND (keyword:agriculture))',
    "industry": '((text:Moroccan) AND (keyword:industry))',
    "other": '((text:Moroccan) AND (keyword:other))',
}

MAX_RESULTS_PER_DOMAIN = 8

_ISO_TO_FR_COUNTRY = {
    "FR": "France", "CA": "Canada", "US": "Etats-Unis",
}


def _country_label(iso_code: Optional[str]) -> Optional[str]:
    if not iso_code:
        return None
    return _ISO_TO_FR_COUNTRY.get(iso_code.upper(), iso_code.upper())


def _headers() -> Dict[str, str]:
    return {"Accept": "application/json"}


def _search_orcid_ids(query: str, rows: int) -> List[str]:
    try:
        response = requests.get(
            f"{ORCID_API_BASE}/search/",
            params={"q": query, "rows": rows},
            headers=_headers(),
            timeout=15,
        )
        response.raise_for_status()
        data = response.json()
    except Exception:
        logger.exception("Erreur lors de la recherche ORCID pour la requete '%s'", query)
        return []

    return [
        item["orcid-identifier"]["path"]
        for item in data.get("result") or []
        if item.get("orcid-identifier", {}).get("path")
    ]


def _fetch_json(path: str) -> Optional[Dict[str, Any]]:
    try:
        response = requests.get(f"{ORCID_API_BASE}{path}", headers=_headers(), timeout=15)
        if response.status_code == 404:
            return None
        response.raise_for_status()
        return response.json()
    except Exception:
        logger.warning("Erreur lors de la recuperation ORCID sur %s", path)
        return None


def _extract_person_info(orcid_id: str) -> Dict[str, Any]:
    person = _fetch_json(f"/{orcid_id}/person") or {}

    name_block = person.get("name") or {}
    given = ((name_block.get("given-names") or {}).get("value") or "").strip()
    family = ((name_block.get("family-name") or {}).get("value") or "").strip()
    full_name = f"{given} {family}".strip()

    biography = ((person.get("biography") or {}).get("content") or "").strip()

    keywords = [
        (kw.get("content") or "").strip()
        for kw in (person.get("keywords") or {}).get("keyword", [])
        if kw.get("content")
    ]

    return {"name": full_name, "biography": biography, "keywords": keywords}


def _extract_affiliations(orcid_id: str, endpoint: str) -> List[Dict[str, Any]]:
    data = _fetch_json(f"/{orcid_id}/{endpoint}") or {}
    summary_key = f"{endpoint[:-1]}-summary"
    affiliations = []

    for group in data.get("affiliation-group") or []:
        for summary in group.get("summaries") or []:
            entry = summary.get(summary_key)
            if not entry:
                continue
            org = entry.get("organization") or {}
            address = org.get("address") or {}
            start_year = ((entry.get("start-date") or {}).get("year") or {}).get("value")
            end_year = ((entry.get("end-date") or {}).get("year") or {}).get("value")
            affiliations.append({
                "org_name": org.get("name"),
                "country_iso": address.get("country"),
                "start_year": int(start_year) if start_year else None,
                "ongoing": end_year is None,
            })

    affiliations.sort(key=lambda a: (a["ongoing"], a["start_year"] or 0), reverse=True)
    return affiliations


_MOROCCAN_NAME_HINTS = ("ayad", "hassani", "el", "ben", "ait")


def _normalize_text(value: str) -> str:
    decomposed = unicodedata.normalize("NFKD", value or "")
    return "".join(c for c in decomposed if not unicodedata.combining(c)).lower().strip()


def _looks_moroccan_name(name: str) -> bool:
    normalized = _normalize_text(name)
    if not normalized:
        return False
    return any(hint in normalized for hint in _MOROCCAN_NAME_HINTS)


def _looks_moroccan_origin(person_info: Dict[str, Any], educations: List[Dict[str, Any]]) -> bool:
    if any(edu.get("country_iso") == MOROCCO_ISO for edu in educations):
        return True
    haystack = " ".join([person_info.get("biography", ""), " ".join(person_info.get("keywords", []))])
    normalized = _normalize_text(haystack)
    if any(marker in normalized for marker in ("moroccan", "marocain", "maroc")):
        return True
    return _looks_moroccan_name(person_info.get("name", ""))


def _current_foreign_country(employments: List[Dict[str, Any]], educations: List[Dict[str, Any]]) -> Optional[str]:
    for job in employments:
        if job.get("country_iso") and job["country_iso"] != MOROCCO_ISO:
            return job["country_iso"]
    for edu in educations:
        if edu.get("country_iso") and edu["country_iso"] != MOROCCO_ISO:
            return edu["country_iso"]
    return None


def discover_orcid_talents(existing_urls: set | None = None) -> List[Dict[str, Any]]:
    existing_urls = existing_urls or set()
    discovered: List[Dict[str, Any]] = []
    seen_orcid_ids: set = set()

    for domain, query in DOMAIN_QUERIES.items():
        orcid_ids = _search_orcid_ids(query, MAX_RESULTS_PER_DOMAIN)
        for orcid_id in orcid_ids:
            if orcid_id in seen_orcid_ids:
                continue
            seen_orcid_ids.add(orcid_id)

            profile_url = f"https://orcid.org/{orcid_id}"
            if profile_url in existing_urls:
                continue

            if get_settings().ENABLE_PROFILE_CACHE and get_settings().SKIP_CACHED_PROFILES:
                cached_status = get_cached_profile_status(profile_url)
                if cached_status is not None:
                    logger.info("ORCID: profil deja en cache (%s) -- saute : '%s'", cached_status, profile_url)
                    continue

            time.sleep(0)  # pas de sleep reel en test

            person_info = _extract_person_info(orcid_id)
            if not person_info.get("name"):
                continue

            educations = _extract_affiliations(orcid_id, "educations")
            employments = _extract_affiliations(orcid_id, "employments")

            country_iso = _current_foreign_country(employments, educations)
            if not country_iso:
                logger.info("ORCID: profil ignore (pas de residence a l'etranger confirmee ni formation etrangere) : '%s'", person_info["name"])
                if get_settings().ENABLE_PROFILE_CACHE:
                    cache_profile(profile_url, source_name="orcid", status="rejected", source_identifier=orcid_id)
                continue

            if not _looks_moroccan_origin(person_info, educations):
                logger.info("ORCID: profil ignore (origine marocaine non confirmee) : '%s'", person_info["name"])
                if get_settings().ENABLE_PROFILE_CACHE:
                    cache_profile(profile_url, source_name="orcid", status="rejected", source_identifier=orcid_id)
                continue

            current_job = next((j for j in employments if j.get("country_iso") == country_iso), None)
            org_name = (current_job or {}).get("org_name", "")

            description = person_info.get("biography") or (
                f"Chercheur(se) actuellement affilie(e) a {org_name}." if org_name else ""
            )

            discovered.append({
                "title": person_info["name"],
                "description": description[:500],
                "source": "ORCID",
                "url": profile_url,
                "sector": domain,
                "country": _country_label(country_iso),
                "expertise_tags": ", ".join(person_info.get("keywords", [])[:5]),
                "years_experience": None,
                "image_url": "",
                "confidence_score": 100,
            })

            if get_settings().ENABLE_PROFILE_CACHE:
                cache_profile(profile_url, source_name="orcid", status="approved", source_identifier=orcid_id)

    logger.info(
        "ORCID : %s profils MRE decouverts (sur %s identifiants ORCID examines).",
        len(discovered), len(seen_orcid_ids),
    )
    return discovered