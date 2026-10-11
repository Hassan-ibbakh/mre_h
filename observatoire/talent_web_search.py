"""
Decouverte de talents marocains via recherche web (Tavily).

Contrairement au flux RSS (sources fixes), ici on interroge activement le web
avec des requetes ciblees par domaine, ce qui permet a l'observatoire de
"chercher lui-meme" de nouveaux profils sans liste de sources predefinie.

Recentrage MRE : toutes les requetes (par domaine et articles de presse)
incluent desormais explicitement une notion de diaspora/etranger, puisque
l'Observatoire ne cible plus que les Marocains Residant a l'Etranger (MRE).

CORRECTIF (portage Sanad AI) : la cle Tavily est desormais lue via
config.get_settings().TAVILY_API_KEY plutot que os.environ.get("TAVILY_API_KEY").
pydantic-settings lit bien .env et remplit Settings, mais n'exporte PAS ces
valeurs dans os.environ -- lire directement os.environ ici retournait donc
toujours None meme avec un .env correctement rempli.
"""
import logging
import re
from typing import Any, Dict, List

from config import get_settings

logger = logging.getLogger("mre_ai")

_MRE_ARTICLE_RE = re.compile(
    r"\b(maroc(?:ain(?:e|s)?)?|marocains|diaspora|étrang(?:er|ère)|etrang(?:er|ere)|expatri(?:é|ée|ation)|à l['’]?étranger|a l['’]?etranger|MRE)\b",
    re.IGNORECASE | re.UNICODE,
)


def _is_mre_article(text: str) -> bool:
    return bool(_MRE_ARTICLE_RE.search(text or ""))

try:
    from tavily import TavilyClient
except ImportError:  # pragma: no cover
    TavilyClient = None

# Une requete type par domaine de l'Observatoire (OBSERVATORY_DOMAINS dans app.py),
# reformulee pour inclure explicitement une notion de diaspora/etranger.
DOMAIN_QUERIES = {
    "tech": "interview marocain a l'etranger ingenieur startup fondateur marocain diaspora international",
    "health": "chercheur marocain a l'etranger prix international medecin innovation sante",
    "education": "professeur marocain a l'etranger universite recherche distinction diaspora",
    "agriculture": "entrepreneur marocain a l'etranger agritech agriculture innovation diaspora",
    "industry": "ingenieur marocain a l'etranger industrie manufacturing leadership international",
    "finance": "analyste marocain a l'etranger finance fintech investissement diaspora",
    "creative": "artiste marocain a l'etranger creatif prix international diaspora",
    "social": "entrepreneur social marocain a l'etranger impact ONG diaspora",
    "other": "talent marocain a l'etranger portrait laureat prix international entrepreneur",
}

MAX_RESULTS_PER_DOMAIN = 4

# Requetes dediees aux ARTICLES D'ACTUALITE parlant de talents marocains
# (distinctions, recompenses, succes entrepreneurial, publications...), par
# opposition aux requetes DOMAIN_QUERIES ci-dessus qui ciblent plutot des
# pages/profils generiques par secteur. Formulees comme des titres d'articles
# pour matcher le style journalistique.
NEWS_ARTICLE_QUERIES = [
    "marocain récompensé à l'étranger 2026",
    "talent marocain distinction internationale",
    "MRE réussite entrepreneuriale Maroc",
    "chercheur marocain découverte publication scientifique",
    "startup fondée par un marocain à l'étranger",
    "marocain classé parmi les meilleurs au monde",
    "diaspora marocaine success story",
    "jeune marocain primé innovation",
    "marocain expatrié réussite professionnelle",
    "ingénieur marocain installé en France Canada Allemagne",
    "chercheur marocain université étrangère distinction",
    "marocain entrepreneur silicon valley Dubai Londres",
]

MAX_RESULTS_PER_NEWS_QUERY = 3
NEWS_RECENCY_DAYS = 60  # ne remonte que les articles publies dans les N derniers jours


def _get_client():
    if TavilyClient is None:
        raise RuntimeError(
            "Le package 'tavily-python' n'est pas installe. Ajoutez-le a requirements.txt "
            "(pip install tavily-python)."
        )
    api_key = get_settings().TAVILY_API_KEY
    if not api_key:
        raise RuntimeError("TAVILY_API_KEY manquante dans l'environnement (.env).")
    return TavilyClient(api_key=api_key)


def search_talents_web(domains: List[str] | None = None) -> List[Dict[str, Any]]:
    """
    Interroge Tavily pour chaque domaine et retourne une liste d'entrees brutes
    (title, content, url, source) pretes a etre passees a l'extracteur LLM.
    """
    domains = domains or list(DOMAIN_QUERIES.keys())
    client = _get_client()
    results: List[Dict[str, Any]] = []

    for domain in domains:
        query = DOMAIN_QUERIES.get(domain)
        if not query:
            continue
        try:
            response = client.search(
                query=query,
                search_depth="basic",
                max_results=MAX_RESULTS_PER_DOMAIN,
                include_answer=False,
            )
        except Exception:
            logger.exception("Erreur Tavily pour le domaine '%s'", domain)
            continue

        for item in response.get("results", []):
            title = (item.get("title") or "").strip()
            content = (item.get("content") or "").strip()
            url = item.get("url") or ""
            if not title and not content:
                continue
            if not _is_mre_article(f"{title} {content}"):
                continue
            results.append({
                "title": title,
                "content": content,
                "url": url,
                "source": "Recherche web (Tavily)",
                "domain_hint": domain,
            })

    return results


def search_talent_articles() -> List[Dict[str, Any]]:
    """
    Recherche specifiquement des ARTICLES D'ACTUALITE parlant de talents
    marocains (recompenses, succes, publications...), via le mode "news" de
    Tavily -- plus adapte aux vrais articles de presse que la recherche
    generique de search_talents_web(), qui cible plutot des pages/profils.
    """
    client = _get_client()
    results: List[Dict[str, Any]] = []

    for query in NEWS_ARTICLE_QUERIES:
        try:
            response = client.search(
                query=query,
                topic="news",
                search_depth="basic",
                max_results=MAX_RESULTS_PER_NEWS_QUERY,
                days=NEWS_RECENCY_DAYS,
                include_answer=False,
            )
        except Exception:
            logger.exception("Erreur Tavily (recherche d'articles) pour la requete '%s'", query)
            continue

        for item in response.get("results", []):
            title = (item.get("title") or "").strip()
            content = (item.get("content") or "").strip()
            url = item.get("url") or ""
            if not title and not content:
                continue
            if not _is_mre_article(f"{title} {content}"):
                continue
            results.append({
                "title": title,
                "content": content,
                "url": url,
                "source": "Article de presse (Tavily News)",
                "domain_hint": None,
            })

    return results