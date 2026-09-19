import json
import logging
import re
import unicodedata
from typing import Any, Dict, Optional

import httpx

from config import get_settings
from .openrouter_key_pool import get_next_openrouter_key_with_name, pool_size

logger = logging.getLogger("sanad.ai")


class TalentExtractionUnavailable(Exception):
    """Le fournisseur LLM est indisponible : le candidat doit être réessayé."""

VALID_DOMAINS = [
    "tech", "health", "education", "agriculture", "industry",
    "finance", "creative", "social", "other",
]

MAX_CONTENT_CHARS = 2200
# Certains modeles gratuits reserves par le routeur imposent des jetons de
# raisonnement avant la reponse. 700 laisse une marge pour ce raisonnement et
# pour l'objet JSON final (court) sans tronquer la sortie.
MAX_COMPLETION_TOKENS = 700

SYSTEM_PROMPT = f"""Analyse cet extrait (article/bio) et determine s'il decrit UN SEUL talent \
marocain individuel (personne precise, marocaine de nationalite/origine : entrepreneur, \
chercheur, ingenieur, expert, laureat...) RESIDANT OU ACTIF A L'ETRANGER (MRE -- Marocain \
Residant a l'Etranger), pas une entreprise/institution/actualite generale.

IMPORTANT -- ne confonds pas deux choses distinctes :
- la NATIONALITE/ORIGINE de la personne (presque toujours "marocaine" dans ces textes -- ne va JAMAIS dans "country")
- le PAYS DE RESIDENCE/ACTIVITE ACTUEL (ce qui va dans "country") : c'est le pays ou la personne vit ou \
travaille MAINTENANT, meme si elle est nee/a grandi/a etudie au Maroc.

Exemple : "Yassine, ingenieur marocain ne a Fes, installe a Lyon depuis 2019, travaille chez..." \
-> country = "France" (PAS "Maroc", meme si "Fes" et "marocain" apparaissent dans le texte).

Reponds UNIQUEMENT en JSON, format exact :
{{"is_talent": bool, "confidence": 0-1, "name": "nom complet ou vide", "sector": un mot parmi {VALID_DOMAINS}, "country": "pays ETRANGER de residence/activite actuel (jamais Maroc, jamais vide si is_talent=true)", "expertise_tags": ["3-5 mots-cles fr"], "years_experience": int ou null, "short_bio": "1-2 phrases reformulees, jamais copiees"}}

is_talent=false si :
- pas une personne identifiable (entreprise, ville, evenement, actualite generale)
- personne non explicitement marocaine (mention de "Maroc" dans un contexte sans lien avec la nationalite de la personne ne suffit pas)
- le texte ne mentionne pas clairement un PAYS DE RESIDENCE/ACTIVITE ETRANGER ACTUEL pour la personne -- \
le simple fait qu'elle soit nee/originaire du Maroc ne suffit PAS et ne doit jamais donner country="Maroc". \
Si aucun pays etranger de residence actuelle n'est explicitement mentionne, is_talent doit etre false -- \
ne jamais mettre "Maroc" par defaut dans "country"
- plusieurs personnes citees ("X et Y", liste de noms) -> toujours false, meme si individuellement pertinentes
- nom generique/anonyme (pas de vrai nom propre donne)
- politicien/elu/responsable gouvernemental OU personnalite sportive (athlete, entraineur, joueur...) sauf si presente explicitement comme entrepreneur/expert dans un domaine non-sportif/non-politique. En cas de doute -> false.

N'invente rien (null/vide si absent du texte). confidence>0.6 requis pour publication auto."""


# Parseur JSON tolerant : certains modeles (notamment les tiers ":free"
# d'OpenRouter, qui ne respectent pas toujours strictement
# response_format={"type": "json_object"}) ajoutent parfois du texte
# explicatif APRES l'objet JSON (ex: "{...}\n\n**Explication:** ..."). Un
# json.loads() classique echoue alors sur toute la chaine. raw_decode()
# parse depuis le debut et s'arrete des que le premier objet JSON est
# complet, en ignorant ce qui suit.
_JSON_DECODER = json.JSONDecoder()


def _extract_first_json_object(raw: str) -> Dict[str, Any]:
    """Extrait le premier objet JSON complet en tete de `raw`, meme s'il est
    suivi de texte parasite. Leve json.JSONDecodeError si aucun objet JSON
    valide ne demarre la chaine (apres nettoyage des ``` eventuels)."""
    cleaned = re.sub(r"^```json|^```", "", raw.strip(), flags=re.MULTILINE).lstrip()
    obj, _end_index = _JSON_DECODER.raw_decode(cleaned)
    return obj


def _normalize_country(country: str) -> str:
    decomposed = unicodedata.normalize("NFKD", country or "")
    without_accents = "".join(c for c in decomposed if not unicodedata.combining(c))
    return without_accents.lower().strip()


def _split_sentences(text: str) -> list[str]:
    return re.split(r"(?<=[.!?])\s+", text)


_GENERIC_NAME_MARKERS = ["inconnu", "unknown", "anonyme", "sans nom", "non precise", "non identifie"]
_GENERIC_NAME_PREFIXES = ("un ", "une ", "le ", "la ", "les ", "des ", "ce ", "cette ")
_MULTI_PERSON_MARKERS = [" et ", " and ", ",", "&"]
_MOROCCO_COUNTRY_VALUES = {"maroc", "morocco"}

_FOREIGN_LOCATION_HINTS = {
    "france": "France", "paris": "France", "lyon": "France", "marseille": "France", "bordeaux": "France",
    "montreal": "Canada", "quebec": "Canada", "canada": "Canada",
    "etats-unis": "Etats-Unis", "etats unis": "Etats-Unis", "usa": "Etats-Unis", "new york": "Etats-Unis", "californie": "Etats-Unis",
    "royaume-uni": "Royaume-Uni", "angleterre": "Royaume-Uni", "londres": "Royaume-Uni",
    "suisse": "Suisse", "geneve": "Suisse",
    "belgique": "Belgique", "bruxelles": "Belgique",
    "allemagne": "Allemagne", "berlin": "Allemagne",
    "espagne": "Espagne", "madrid": "Espagne",
    "italie": "Italie", "rome": "Italie",
    "pays-bas": "Pays-Bas", "amsterdam": "Pays-Bas",
    "emirats": "Emirats Arabes Unis", "dubai": "Emirats Arabes Unis",
    "qatar": "Qatar", "australie": "Australie", "japon": "Japon", "chine": "Chine", "singapour": "Singapour",
}
_FOREIGN_RESIDENCE_TERMS = [
    "installe", "installee", "vit", "habite", "reside", "residant", "travaille",
    "bas", "base", "expatrie", "expatriee", "diaspora", "immigre", "immigree",
    "aujourd'hui", "depuis", "actuel", "actuellement",
]


def _looks_like_single_real_name(name: str) -> bool:
    lowered = name.lower().strip()
    if not lowered:
        return False
    if any(marker in lowered for marker in _GENERIC_NAME_MARKERS):
        return False
    if lowered.startswith(_GENERIC_NAME_PREFIXES):
        return False
    if any(marker in name for marker in _MULTI_PERSON_MARKERS):
        return False
    words = re.split(r"[\s\-]+", name.strip())
    significant_words = [w for w in words if len(w) >= 2 and w[0].isupper()]
    return len(significant_words) >= 2


def _call_openrouter_with_retry(combined_text: str) -> Optional[str]:
    """Retourne le contenu texte du message, ou None si le modele a repondu
    200 OK sans le moindre contenu exploitable (arrive occasionnellement
    avec les modeles ":free" -- finish_reason atteint avant tout token
    utile, reponse vide cote provider, etc.). None est une valeur de retour
    legitime ici, pas une erreur : c'est a l'appelant de decider quoi en
    faire (traiter comme un candidat rejete)."""
    settings = get_settings()
    model = settings.OPENROUTER_EXTRACTOR_MODEL
    last_error: Optional[Exception] = None

    for attempt in range(pool_size()):
        env_name, api_key = get_next_openrouter_key_with_name()
        try:
            response = httpx.post(
                f"{settings.OPENROUTER_BASE_URL}/chat/completions",
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                json={
                    "model": model,
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": combined_text[:MAX_CONTENT_CHARS]},
                    ],
                    "temperature": 0.1,
                    "max_tokens": MAX_COMPLETION_TOKENS,
                    "response_format": {"type": "json_object"},
                    # Certains modeles gratuits imposent le raisonnement et
                    # refusent ``effort: none``. On le laisse donc actif,
                    # mais on ne le retourne pas au client ; le budget de
                    # sortie ci-dessus reserve de la place au JSON final.
                    "reasoning": {"exclude": True},
                    # Le routeur gratuit peut sinon envoyer la requete a un
                    # fournisseur qui ignore ``response_format``. On exige
                    # donc explicitement le support de tous les parametres
                    # envoyes, en particulier le mode JSON.
                    "provider": {"require_parameters": True},
                },
                timeout=30.0,
            )
            # Ces erreurs désignent la requête (endpoint, modèle ou paramètres),
            # et non une clé particulière. Les répéter sur toutes les clés du
            # pool ne peut pas réussir et masque le message utile du fournisseur.
            if response.status_code in (400, 404, 422):
                logger.error(
                    "Requête OpenRouter non exécutable (HTTP %s, modèle '%s') : %s",
                    response.status_code,
                    model,
                    response.text[:500],
                )
                raise RuntimeError(f"OpenRouter HTTP {response.status_code}")
            if response.status_code in (401, 429):
                logger.warning("OpenRouter %s avec cle '%s' -- tentative %s/%s",
                                response.status_code, env_name, attempt + 1, pool_size())
                last_error = RuntimeError(f"{response.status_code} ({env_name})")
                continue
            response.raise_for_status()
            payload = response.json()
            choices = payload.get("choices") or []
            if not choices:
                logger.warning(
                    "Reponse OpenRouter sans 'choices' (cle '%s') : %s",
                    env_name, str(payload)[:200],
                )
                return None
            content = (choices[0].get("message") or {}).get("content")
            return content
        except httpx.HTTPError as exc:
            logger.warning("Erreur HTTP OpenRouter cle '%s': %s", env_name, exc)
            last_error = exc
            continue

    raise last_error or RuntimeError("Echec appel OpenRouter (pool epuise).")


def extract_talent_profile(
    title: str,
    text: str,
    source_name: str = "",
    source_url: str = "",
) -> Optional[Dict[str, Any]]:
    combined = f"Titre: {title}\n\nContenu: {text}".strip()
    if not combined or len(combined) < 20:
        return None

    try:
        raw = _call_openrouter_with_retry(combined)
    except Exception as exc:
        logger.warning("Extraction abandonnee pour '%s': toutes les cles OpenRouter ont echoue.", title[:80])
        raise TalentExtractionUnavailable from exc

    # Le modele a repondu 200 OK mais sans contenu exploitable (voir
    # docstring de _call_openrouter_with_retry) -- traite comme un rejet
    # normal plutot que de laisser une AttributeError remonter.
    if not raw or not raw.strip():
        logger.info("Extraction ignoree pour '%s' : reponse vide du modele.", title[:80])
        raise TalentExtractionUnavailable("Réponse vide du fournisseur LLM")

    try:
        data = _extract_first_json_object(raw)
    except json.JSONDecodeError:
        logger.warning("Reponse JSON invalide de l'extracteur LLM: %s", raw[:200])
        raise TalentExtractionUnavailable("Réponse JSON invalide du fournisseur LLM")

    logger.debug("Extraction LLM pour '%s' -> %s", title[:80], data)

    if not data.get("is_talent"):
        return None
    confidence = float(data.get("confidence") or 0)
    if confidence < 0.4:
        return None

    name = (data.get("name") or "").strip()
    if not name or len(name) > 120:
        return None
    if not _looks_like_single_real_name(name):
        logger.info("Profil rejete (nom generique/multi-personnes): '%s'", name)
        return None

    country = (data.get("country") or "").strip()
    normalized_country = _normalize_country(country)

    if not country or normalized_country in _MOROCCO_COUNTRY_VALUES:
        inferred_country = None
        for sent in _split_sentences(combined):
            sent_lower = sent.lower()
            if not any(term in sent_lower for term in _FOREIGN_RESIDENCE_TERMS):
                continue
            for hint, label in _FOREIGN_LOCATION_HINTS.items():
                if hint in sent_lower:
                    inferred_country = label
                    break
            if inferred_country:
                break

        if inferred_country:
            country = inferred_country
            logger.info("Profil conserve via indices de residence etrangere: '%s' -> '%s'", name, country)
        else:
            logger.info("Profil rejete (pas de residence etrangere confirmee): '%s' (pays LLM: '%s')", name, country)
            return None

    sector = data.get("sector") if data.get("sector") in VALID_DOMAINS else "other"
    tags = data.get("expertise_tags") or []
    if not isinstance(tags, list):
        tags = []

    return {
        "title": name,
        "confidence_score": round(confidence * 100),
        "sector": sector,
        "country": country,
        "expertise_tags": ", ".join(str(t).strip() for t in tags if str(t).strip()),
        "years_experience": data.get("years_experience") if isinstance(data.get("years_experience"), int) else None,
        "description": (data.get("short_bio") or "").strip(),
        "source_url": source_url,
        "source_name": source_name or "Source externe",
        "image_url": "",
    }
