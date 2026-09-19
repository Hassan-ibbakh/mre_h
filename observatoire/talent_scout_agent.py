"""
Agent LangChain (tool-calling) de decouverte de talents MRE.

C'est le LLM lui-meme qui choisit, a chaque etape, quelle source interroger
(Tavily par secteur, Tavily actualites, RSS, ORCID) et quand s'arreter --
l'agent reste pleinement autonome sur le CHOIX des outils.

NOUVEAU (garde-fou search_orcid) : ORCID est une source STRUCTUREE et
DETERMINISTE (contrairement a Tavily/RSS, ou rappeler avec des mots-cles
differents peut legitimement remonter du nouveau contenu). Un seul appel a
search_orcid suffit donc a couvrir toute la recherche ORCID d'un cycle ; un
2e appel ne trouverait rien de neuf mais consommerait quand meme un tour de
raisonnement (donc un appel OpenRouter). MAX_ORCID_CALLS plafonne ce nombre
a 1 -- l'outil reste dans la toolbox de l'agent (qui peut toujours choisir
de l'appeler ou non, et dans l'ordre qu'il veut), seul un 2e appel redondant
est intercepte avec un message explicite qui redirige l'agent vers les
autres sources. Ce n'est pas une restriction de l'agentivite : c'est le
meme principe que MAX_VALIDATIONS deja en place pour
validate_and_store_profile.

CORRECTIF (search_orcid ne passe plus par raw_cache/validate_and_store_profile) :
contrairement a Tavily/RSS, qui retournent du texte brut a faire juger par un
LLM (extract_talent_profile), ORCID a DEJA fait tout le travail de validation
en interne (discover_orcid_talents() filtre residence etrangere + origine
marocaine, et cache directement approved/rejected via cache_profile()). Les
entrees ORCID n'ont pas de champ "content" (elles ont "description"), donc les
faire transiter par raw_cache -> validate_and_store_profile echouait
silencieusement : extract_talent_profile recevait un texte vide, le LLM
renvoyait souvent une reponse vide, et le profil etait alors marque "rejected"
en cache -- alors qu'il avait deja ete "approved" par ORCID lui-meme. Comme
SKIP_CACHED_PROFILES saute ensuite toute URL en cache (quel que soit son
statut), ces profils devenaient injoignables a vie des le cycle suivant.
Le tool search_orcid ci-dessous ajoute donc desormais les resultats
DIRECTEMENT dans `discovered` (avec dedup par nom), sans passer par
validate_and_store_profile ni par une 2e validation LLM.

Economie de quota OpenRouter (pool Observatoire) :
  - la recherche de sources (Tavily, RSS, ORCID) NE CONSOMME AUCUN token
    OpenRouter ;
  - la verification de doublon est une comparaison de chaines locale ;
  - le SEUL outil qui consomme un appel OpenRouter "metier" est
    `validate_and_store_profile` (reutilise extract_talent_profile(), deja
    protege par son propre retry multi-cles via openrouter_key_pool.py) ;
  - le raisonnement de l'agent lui-meme (choix d'outil a chaque tour)
    consomme AUSSI un appel OpenRouter par tour -- c'est pourquoi
    RetryableChatOpenRouter est utilise ici, et pourquoi limiter les tours
    "inutiles" (comme un 2e appel ORCID redondant) reduit la charge globale.

Garde-fous :
  - RetryableChatOpenRouter (retry multi-cles) au lieu d'une seule cle figee ;
  - compteur CODE (pas juste une consigne texte) qui plafonne le nombre de
    validate_and_store_profile par cycle a MAX_VALIDATIONS ;
  - compteur CODE equivalent pour search_orcid, plafonne a MAX_ORCID_CALLS ;
  - delai de courtoisie apres chaque validation reussie ;
  - deduplication via ProfileCache (database.py) pour eviter la reutation
    des memes sources ORCID/RSS/Tavily a chaque cycle de 24h.

Les resultats bruts (title/content/url/source) recuperes par les outils de
recherche TEXTE (Tavily, RSS) ne sont PAS renvoyes en entier au LLM : ils
sont mis en cache localement (par URL) et seul un resume court est montre a
l'agent. Quand l'agent valide un candidat, il ne passe que l'URL ; le texte
complet est relu depuis le cache pour l'appel a l'extracteur LLM. ORCID suit
desormais un chemin different (voir plus haut) : ses resultats sont deja
des profils complets et valides, pas du texte a faire juger.
"""
import json
import logging
import os
import time
from typing import Any, Dict, List, Set

from .langchain_compat import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool

from .retryable_openrouter import RetryableChatOpenRouter
from .talent_extractor_llm import TalentExtractionUnavailable, extract_talent_profile
from .talent_web_search import DOMAIN_QUERIES, search_talents_web, search_talent_articles
from .talent_monitor import _fetch_rss_raw_entries, _is_likely_duplicate, _normalize_name
from .orcid_source import discover_orcid_talents
from database import cache_profile, classify_mre_review_status, is_profile_cached
from config import get_settings

logger = logging.getLogger("sanad.ai")

MAX_ITERATIONS = 18
MAX_VALIDATIONS = 12

# NOUVEAU : garde-fou pour search_orcid, meme principe que MAX_VALIDATIONS.
# ORCID est une source structuree/deterministe (contrairement a Tavily/RSS ou
# rappeler avec des mots-cles differents a du sens) : un seul appel par cycle
# suffit a couvrir la recherche, un 2e appel ne trouve rien de neuf mais
# consomme quand meme un tour de raisonnement (donc un appel OpenRouter).
MAX_ORCID_CALLS = 1

POST_VALIDATION_SLEEP_SECONDS = 1.0
MAX_SNIPPET_CHARS = 150
MAX_TOOL_RESULTS_SHOWN = 8

SYSTEM_PROMPT = """Tu es un agent de veille qui cherche des Marocains Residant a l'Etranger \
(MRE) talentueux (chercheurs, entrepreneurs, ingenieurs, experts...) pour l'Observatoire des \
Talents Marocains.

Outils disponibles :
- des outils de RECHERCHE (search_sector, search_news, search_rss) : gratuits, pas de cout, \
utilise-les pour explorer. Ils retournent du texte brut (titre/url/extrait) que tu dois faire \
valider ensuite via validate_and_store_profile.
- search_orcid : gratuit, source structuree et deterministe. Contrairement aux autres outils \
de recherche, les profils qu'il retourne sont DEJA VALIDES et ajoutes automatiquement aux \
resultats -- inutile d'appeler validate_and_store_profile dessus, et inutile de le rappeler \
plusieurs fois (un seul appel suffit a couvrir tout le cycle).
- check_duplicate : verification locale gratuite, a utiliser AVANT de valider un candidat issu \
de search_sector / search_news / search_rss.
- validate_and_store_profile : SEUL outil qui coute une requete IA. A n'utiliser que sur un \
candidat serieux issu de search_sector / search_news / search_rss (nom propre identifiable, \
indice clair de residence/activite a l'etranger). Ne JAMAIS l'appeler sur une url ORCID. \
Passe-lui l'URL exacte renvoyee par un outil de recherche. Limite a 12 validations par cycle --
l'outil te previendra si tu atteins la limite.

Consignes :
1. Varie les sources et les secteurs a chaque etape (essaie plusieurs secteurs differents avec \
search_sector avant de conclure qu'il n'y a rien), ne repete pas une recherche deja faite.
2. Verifie toujours check_duplicate avant validate_and_store_profile.
3. Arrete-toi des que tu as valide 8 a 12 nouveaux profils, ou des que plusieurs recherches \
d'affilee n'apportent plus rien de nouveau.
4. Sois direct et econome : pas de blabla, enchaine les appels d'outils utiles."""


def _entry_summary(entry: Dict[str, Any]) -> str:
    snippet = (entry.get("content") or "")[:MAX_SNIPPET_CHARS].strip()
    return f"- titre: {entry.get('title', '')}\n  url: {entry.get('url', '')}\n  extrait: {snippet}"


def _summarize_entries(entries: List[Dict[str, Any]]) -> str:
    if not entries:
        return "Aucun resultat."
    shown = entries[:MAX_TOOL_RESULTS_SHOWN]
    lines = [_entry_summary(e) for e in shown]
    remainder = len(entries) - len(shown)
    if remainder > 0:
        lines.append(f"(+ {remainder} autre(s) resultat(s) non affiches)")
    return "\n".join(lines)


def _build_tools(
    existing_urls: Set[str],
    known_normalized_names: Set[str],
    discovered: List[Dict[str, Any]],
    raw_cache: Dict[str, Dict[str, Any]],
):
    validation_count = {"n": 0}
    # NOUVEAU : compteur d'appels a search_orcid pour ce cycle.
    orcid_call_count = {"n": 0}

    def _cache_new_entries(entries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        fresh = []
        for entry in entries:
            url = entry.get("url") or ""
            if not url or url in existing_urls or url in raw_cache:
                continue
            raw_cache[url] = entry
            fresh.append(entry)
        return fresh

    @tool
    def search_sector(domain: str) -> str:
        """Recherche web (Tavily) de talents MRE dans UN secteur donne. Domaines
        valides : tech, health, education, agriculture, industry, finance,
        creative, social, other. Ne consomme aucun quota IA. Retourne une
        courte liste titre/url/extrait -- utilise l'url exacte pour valider
        un candidat ensuite."""
        domain = (domain or "").strip().lower()
        if domain not in DOMAIN_QUERIES:
            return f"Domaine inconnu '{domain}'. Domaines valides : {', '.join(DOMAIN_QUERIES.keys())}."
        try:
            entries = search_talents_web(domains=[domain])
        except Exception:
            logger.exception("Erreur outil search_sector pour le domaine '%s'", domain)
            return "Erreur lors de la recherche web pour ce secteur."
        fresh = _cache_new_entries(entries)
        logger.info(
            "Tavily Web [%s] : %s resultat(s), %s nouveau(x).",
            domain, len(entries), len(fresh),
        )
        return _summarize_entries(fresh)

    @tool
    def search_news(_dummy: str = "") -> str:
        """Recherche des articles de presse recents (60 derniers jours) parlant
        de talents marocains a l'etranger (distinctions, succes, publications).
        Ne consomme aucun quota IA. Retourne une courte liste titre/url/extrait."""
        try:
            entries = search_talent_articles()
        except Exception:
            logger.exception("Erreur outil search_news")
            return "Erreur lors de la recherche d'articles de presse."
        fresh = _cache_new_entries(entries)
        logger.info(
            "Tavily News : %s resultat(s), %s nouveau(x).",
            len(entries), len(fresh),
        )
        return _summarize_entries(fresh)

    @tool
    def search_rss(_dummy: str = "") -> str:
        """Parcourt les flux RSS/Atom configures (data/talent_sources.json) a la
        recherche d'articles mentionnant un talent marocain. Ne consomme
        aucun quota IA. Retourne une courte liste titre/url/extrait."""
        try:
            entries = _fetch_rss_raw_entries()
        except Exception:
            logger.exception("Erreur outil search_rss")
            return "Erreur lors de la lecture des flux RSS."
        fresh = _cache_new_entries(entries)
        logger.info(
            "RSS : %s resultat(s), %s nouveau(x).",
            len(entries), len(fresh),
        )
        return _summarize_entries(fresh)

    @tool
    def search_orcid(_dummy: str = "") -> str:
        """Interroge l'API ORCID pour des profils structures de chercheurs
        marocains actifs ou residant a l'etranger. Ne consomme aucun quota IA.
        Contrairement aux autres outils de recherche, les profils ORCID sont
        DEJA VALIDES (filtrage deterministe cote API, pas besoin d'une 2e
        validation LLM) : ils sont ajoutes AUTOMATIQUEMENT aux resultats de ce
        cycle. N'appelle PAS validate_and_store_profile sur une url ORCID --
        c'est deja fait. Source deterministe : UN SEUL appel par cycle est
        autorise, un 2e appel n'apporterait rien de nouveau (l'outil refusera
        et te le signalera)."""
        if orcid_call_count["n"] >= MAX_ORCID_CALLS:
            return (
                "search_orcid deja utilise ce cycle (source deterministe, un "
                "seul appel suffit) -- inutile de le rappeler, explore plutot "
                "search_sector / search_news / search_rss."
            )
        orcid_call_count["n"] += 1
        try:
            entries = discover_orcid_talents(existing_urls=existing_urls)
        except Exception:
            logger.exception("Erreur outil search_orcid")
            return "Erreur lors de la lecture des profils ORCID."

        added = 0
        skipped_duplicates = 0
        for entry in entries:
            url = (entry.get("url") or "").strip()
            title = (entry.get("title") or "").strip()
            if not title or not url or url in existing_urls:
                continue
            if _is_likely_duplicate(title, known_normalized_names):
                skipped_duplicates += 1
                continue

            known_normalized_names.add(_normalize_name(title))
            existing_urls.add(url)
            discovered.append({
                "title": title,
                "description": entry.get("description", ""),
                "source": entry.get("source", "ORCID"),
                "url": url,
                "sector": entry.get("sector", "other"),
                "country": entry.get("country"),
                "expertise_tags": entry.get("expertise_tags", ""),
                "years_experience": entry.get("years_experience"),
                "image_url": entry.get("image_url", ""),
            })
            added += 1

        if added == 0 and skipped_duplicates == 0:
            return "Aucun nouveau profil ORCID trouve (deja en cache ou aucun resultat)."
        if added == 0:
            return f"Aucun nouveau profil ORCID ajoute ({skipped_duplicates} doublon(s) ecarte(s))."
        return (
            f"{added} profil(s) ORCID valide(s) et ajoute(s) automatiquement "
            f"({skipped_duplicates} doublon(s) ecarte(s)). Aucune action supplementaire "
            f"requise sur ces profils -- ne pas appeler validate_and_store_profile dessus."
        )

    @tool
    def check_duplicate(name: str) -> str:
        """Verifie SANS appel IA si un nom ressemble a un profil deja connu
        (deja en base ou deja valide dans cette session). Utilise ceci avant
        validate_and_store_profile pour ne pas gaspiller de quota sur un
        doublon evident."""
        if _is_likely_duplicate(name or "", known_normalized_names):
            return f"Doublon probable pour '{name}' -- ne pas valider."
        return f"Pas de doublon connu pour '{name}'."

    @tool
    def validate_and_store_profile(url: str) -> str:
        """Valide DEFINITIVEMENT un candidat via l'agent d'extraction IA (SEUL
        outil qui consomme un appel OpenRouter) puis l'ajoute a la liste des
        profils decouverts si retenu. `url` doit etre une URL EXACTE renvoyee
        par search_sector / search_news / search_rss (PAS par search_orcid,
        deja traite automatiquement). Limite a 12 validations par cycle."""
        if validation_count["n"] >= MAX_VALIDATIONS:
            return (
                f"Limite de {MAX_VALIDATIONS} validations atteinte pour ce cycle -- "
                "arrete-toi maintenant, n'appelle plus cet outil."
            )

        url_clean = (url or "").strip()
        if get_settings().ENABLE_PROFILE_CACHE and get_settings().SKIP_CACHED_PROFILES:
            if is_profile_cached(url_clean):
                return "URL deja en cache (approved/rejected) -- saute pour economiser tokens."

        entry = raw_cache.get(url_clean)
        if not entry:
            return "URL inconnue -- utilise une url exacte renvoyee par search_sector / search_news / search_rss."

        try:
            profile = extract_talent_profile(
                title=entry.get("title", ""),
                text=entry.get("content", ""),
                source_name=entry.get("source", "Source externe"),
                source_url=entry.get("url", ""),
            )
        except TalentExtractionUnavailable:
            # Une panne OpenRouter ne constitue pas un rejet éditorial. Ne pas
            # écrire "rejected" dans ProfileCache, sinon SKIP_CACHED_PROFILES
            # empêcherait toute nouvelle tentative lors du prochain cycle.
            logger.warning("Validation différée pour '%s' : LLM indisponible.", entry.get("title"))
            return "Validation differee : le fournisseur IA est indisponible, l'URL sera reessayee au prochain cycle."
        except Exception:
            logger.exception("Erreur lors de la validation IA pour '%s'", entry.get("title"))
            return "Validation differee : erreur technique, l'URL sera reessayee au prochain cycle."

        if not profile:
            if get_settings().ENABLE_PROFILE_CACHE:
                cache_profile(url_clean, source_name=entry.get("source", "unknown"), status="rejected")
            return "Candidat rejete par la validation IA (pas un talent MRE confirme)."

        if _is_likely_duplicate(profile["title"], known_normalized_names):
            if get_settings().ENABLE_PROFILE_CACHE:
                cache_profile(url_clean, source_name=entry.get("source", "unknown"), status="rejected")
            return f"Rejete : '{profile['title']}' est un doublon probable."

        known_normalized_names.add(_normalize_name(profile["title"]))
        confidence_score = profile.get("confidence_score", 0)
        review_status = classify_mre_review_status(confidence_score, profile["country"])
        discovered.append({
            "title": profile["title"],
            "confidence_score": confidence_score,
            "description": profile["description"],
            "source": profile["source_name"],
            "url": profile["source_url"],
            "sector": profile["sector"],
            "country": profile["country"],
            "expertise_tags": profile["expertise_tags"],
            "years_experience": profile["years_experience"],
            "image_url": profile.get("image_url", ""),
        })

        if get_settings().ENABLE_PROFILE_CACHE:
            cache_profile(url_clean, source_name=entry.get("source", "unknown"), status=review_status)

        validation_count["n"] += 1
        time.sleep(POST_VALIDATION_SLEEP_SECONDS)

        remaining = MAX_VALIDATIONS - validation_count["n"]
        return (
            f"Profil enregistre : '{profile['title']}' ({profile['country']}, score {confidence_score}/100, statut {review_status}). "
            f"({remaining} validation(s) restante(s) pour ce cycle.)"
        )

    return [search_sector, search_news, search_rss, search_orcid, check_duplicate, validate_and_store_profile]


def _build_agent_executor(tools) -> AgentExecutor:
    llm = RetryableChatOpenRouter(model=get_settings().OPENROUTER_AGENT_MODEL, temperature=0.1)

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])

    agent = create_tool_calling_agent(llm, tools, prompt)
    return AgentExecutor(
        agent=agent,
        tools=tools,
        max_iterations=MAX_ITERATIONS,
        early_stopping_method="force",
        verbose=True,
        handle_parsing_errors=True,
    )


def _load_local_seed_profiles(existing_urls: Set[str]) -> List[Dict[str, Any]]:
    seed_path = os.path.join("data", "talents_seed.json")
    if not os.path.exists(seed_path):
        return []

    with open(seed_path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)

    profiles = []
    for item in payload or []:
        if not isinstance(item, dict):
            continue
        title = (item.get("title") or "").strip()
        url = (item.get("source_url") or "").strip()
        if not title:
            continue
        if url and url in existing_urls:
            continue
        profiles.append({
            "title": title,
            "description": item.get("description", ""),
            "source": "seed",
            "url": url or f"seed:{title}",
            "sector": item.get("sector", "other"),
            "country": item.get("country", "Maroc"),
            "expertise_tags": item.get("expertise_tags", ""),
            "years_experience": item.get("years_experience"),
            "image_url": "",
            "confidence_score": 100,
        })
    return profiles


def run_talent_scout_agent(
    existing_urls: Set[str] | None = None,
    existing_titles: Set[str] | None = None,
) -> List[Dict[str, Any]]:
    """
    Lance un cycle de decouverte pilote par l'agent : le LLM choisit lui-meme
    ses outils/sources et decide quand s'arreter (borne par MAX_ITERATIONS,
    MAX_VALIDATIONS, et desormais MAX_ORCID_CALLS pour search_orcid).

    Retourne une liste de profils avec les memes cles que l'ancien pipeline
    (title, description, source, url, sector, country, expertise_tags,
    years_experience, image_url), pour rester compatible avec
    `discover_talents_from_sources` / talent_sync.py.
    """
    existing_urls = existing_urls or set()
    known_normalized_names = {_normalize_name(t) for t in (existing_titles or set())}
    discovered: List[Dict[str, Any]] = []
    raw_cache: Dict[str, Dict[str, Any]] = {}

    tools = _build_tools(existing_urls, known_normalized_names, discovered, raw_cache)

    try:
        executor = _build_agent_executor(tools)
    except Exception:
        logger.exception("Impossible de construire l'agent de decouverte de talents.")
        return []

    task = (
        "Trouve de nouveaux talents MRE en explorant plusieurs secteurs et sources "
        "differentes, verifie les doublons, et valide les candidats serieux."
    )

    try:
        executor.invoke({"input": task})
    except Exception:
        logger.exception("Erreur durant l'execution de l'agent de decouverte de talents.")

    # Collecte deterministe complete a chaque cycle, meme si l'agent a deja
    # trouve des profils : tous les domaines Web, News, RSS et ORCID sont ainsi
    # couverts avant de terminer la synchronisation.
    search_tools = {getattr(tool, "name", ""): tool for tool in tools}
    try:
        sector_tool = search_tools.get("search_sector")
        if sector_tool:
            for domain in DOMAIN_QUERIES.keys():
                try:
                    sector_tool.invoke(domain)
                except Exception:
                    logger.exception("Collecte search_sector error pour le domaine '%s'", domain)
        for name in ("search_news", "search_rss", "search_orcid"):
            tool = search_tools.get(name)
            if tool:
                try:
                    tool.invoke("")
                except Exception:
                    logger.exception("Collecte %s error", name)
    except Exception:
        logger.exception("Collecte complete des sources impossible")

    validate_tool = search_tools.get("validate_and_store_profile")
    check_tool = search_tools.get("check_duplicate")
    if validate_tool and check_tool:
        for url, entry in list(raw_cache.items()):
            if len(discovered) >= MAX_VALIDATIONS:
                break
            title = entry.get("title", "")
            if not title:
                continue
            try:
                duplicate_message = check_tool.invoke(title)
            except Exception:
                logger.exception("Collecte check_duplicate error pour '%s'", title)
                duplicate_message = ""
            if "Doublon probable" in duplicate_message:
                continue
            try:
                validate_tool.invoke(url)
            except Exception:
                logger.exception("Collecte validate_and_store_profile error pour '%s'", url)
                continue

    if not discovered:
        seed_profiles = _load_local_seed_profiles(existing_urls)
        for profile in seed_profiles:
            title = profile.get("title", "")
            if not title:
                continue
            if _is_likely_duplicate(title, known_normalized_names):
                continue
            discovered.append(profile)

    logger.info(
        "Agent scout : %s profil(s) valide(s) sur %s piste(s) examinees.",
        len(discovered), len(raw_cache),
    )
    return discovered
