import logging
from typing import Any, Callable, Dict, List, Tuple

from .langchain_compat import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool

from config import get_settings
from .retryable_openrouter import RetryableChatOpenRouter
from .utils import tokenize_keywords

logger = logging.getLogger("sanad.ai")

MAX_ITERATIONS = 5
MAX_RESULTS_SHOWN = 10

SYSTEM_PROMPT = """Tu es l'assistant de l'Observatoire des Talents Marocains. \
Un utilisateur decrit en langage naturel le profil qu'il cherche (competences, \
secteur, pays, experience). Tu dois trouver les meilleurs profils correspondants \
dans la base et les selectionner, puis repondre brievement a l'utilisateur.

Outils disponibles :
- list_options : liste les secteurs et pays reellement presents dans la base \
(utile si tu n'es pas sur des valeurs exactes a utiliser). Gratuit.
- search_profiles : cherche des profils par mots-cles/secteur/pays/experience \
minimale. Gratuit, tu peux l'appeler plusieurs fois avec des criteres differents \
pour affiner (elargir si trop peu de resultats, restreindre si trop de bruit).
- select_matches : FINALISE ta reponse en donnant les ids des profils les plus \
pertinents parmi ceux vus via search_profiles (5 maximum), du meilleur au moins \
bon. A appeler une seule fois, en dernier.

Consignes : 2-3 appels a search_profiles maximum avant de finaliser. Si aucun \
profil pertinent n'est trouve meme apres avoir elargi la recherche, appelle \
select_matches avec une liste vide plutot que de forcer des resultats non \
pertinents. Termine toujours par une courte reponse en francais expliquant ce \
que tu as trouve (ou pas)."""


def _build_tools(search_profiles_fn, list_options_fn, selected, result_cache, raw_query):
    def _cache_and_format(results: List[Dict[str, Any]]) -> str:
        if not results:
            return "Aucun profil trouve pour ces criteres."
        lines = []
        for r in results[:MAX_RESULTS_SHOWN]:
            result_cache[r["id"]] = r
            years = r.get("years_experience")
            lines.append(
                f"- id:{r['id']} | {r['title']} | secteur:{r.get('sector_label', '?')} | "
                f"pays:{r.get('country') or '?'} | experience:{years if years is not None else '?'} ans | "
                f"score:{r.get('match_score', '?')} | tags:{', '.join(r.get('tags', [])[:5])}"
            )
        return "\n".join(lines)

    @tool
    def list_options() -> str:
        """Liste les secteurs et pays actuellement presents dans la base de
        l'Observatoire. Gratuit, utile pour caler tes criteres de recherche
        sur des valeurs qui existent reellement plutot que de deviner."""
        try:
            options = list_options_fn()
        except Exception:
            logger.exception("Erreur outil list_options")
            return "Erreur lors de la lecture des options disponibles."
        sectors = ", ".join(options.get("sectors", [])) or "aucun"
        countries = ", ".join(options.get("countries", [])) or "aucun"
        return f"Secteurs disponibles : {sectors}\nPays disponibles : {countries}"

    @tool
    def search_profiles(
        keywords: str = "", sector: str = "", country: str = "",
        min_years_experience: int = 0,
    ) -> str:
        """Cherche des profils dans la base de l'Observatoire. `keywords` :
        mots-cles separes par des virgules ou des espaces (competences,
        domaine...), vide si aucun mot-cle precis. `sector` : code exact
        d'un secteur (vide = tous les secteurs). `country` : nom de pays,
        meme partiel (vide = tous les pays). `min_years_experience` : 0 si
        pas de contrainte d'experience. Gratuit, appelable plusieurs fois
        pour affiner. Retourne une liste courte avec un id et un score par
        profil -- reutilise ces ids pour select_matches."""
        kw_list = tokenize_keywords(keywords)
        try:
            results = search_profiles_fn(
                query=raw_query, keywords=kw_list, sector=sector or None,
                country=country or None, min_years_experience=min_years_experience or None,
            )
        except Exception:
            logger.exception("Erreur outil search_profiles")
            return "Erreur lors de la recherche dans la base."
        return _cache_and_format(results)

    @tool
    def select_matches(post_ids: List[int]) -> str:
        """Finalise la selection : donne la liste des ids (5 maximum) des
        profils les plus pertinents parmi ceux vus via search_profiles, du
        meilleur au moins bon. Utilise une liste vide si rien de pertinent
        n'a ete trouve."""
        selected.clear()
        for post_id in (post_ids or [])[:5]:
            entry = result_cache.get(post_id)
            if entry:
                selected.append(entry)
        if not selected:
            return "Selection finalisee : aucun profil retenu."
        return f"Selection finalisee : {len(selected)} profil(s) retenu(s)."

    return [list_options, search_profiles, select_matches]


def build_observatory_agent(
    search_profiles_fn: Callable[..., List[Dict[str, Any]]],
    list_options_fn: Callable[[], Dict[str, List[str]]],
    raw_query: str = "",
) -> Tuple[AgentExecutor, List[Dict[str, Any]]]:
    """
    A appeler UNE FOIS PAR REQUETE HTTP (pas au demarrage) : `selected` est un
    etat local capture par closure pour cet appel precis. Lis `selected`
    APRES executor.invoke(...), pas avant.

    `raw_query` : la phrase brute de l'utilisateur, propagee jusqu'a
    search_profiles_fn pour permettre le scoring semantique (embeddings) en
    plus du scoring mots-cles.
    """
    selected: List[Dict[str, Any]] = []
    result_cache: Dict[int, Dict[str, Any]] = {}
    tools = _build_tools(search_profiles_fn, list_options_fn, selected, result_cache, raw_query)

    settings = get_settings()
    llm = RetryableChatOpenRouter(model=settings.OPENROUTER_AGENT_MODEL, temperature=0.0)

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])
    agent = create_tool_calling_agent(llm, tools, prompt)
    executor = AgentExecutor(
        agent=agent, tools=tools, max_iterations=MAX_ITERATIONS,
        early_stopping_method="force", verbose=False, handle_parsing_errors=True,
    )
    return executor, selected