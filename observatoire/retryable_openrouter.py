"""
ChatOpenAI (langchain_openai) pointe sur l'endpoint OpenRouter, avec retry
multi-cles sur 401/429, limite a MAX_RETRIES_PER_CALL tentatives PAR APPEL
(pas pool_size()) -- meme logique/raisonnement que l'ancien RetryableChatGroq,
transpose sur OpenRouter (qui expose une API OpenAI-compatible, d'ou l'usage
de ChatOpenAI plutot que d'un client dedie).

Design identique a l'original : l'instance parente (self) est construite
normalement avec une VRAIE cle du pool -- elle sert uniquement a satisfaire
l'interface BaseChatModel attendue par create_tool_calling_agent (bind_tools,
etc.), jamais utilisee pour le vrai appel reseau. A chaque generation,
_generate() cree une instance ChatOpenAI FRAICHE avec la cle courante du pool
et max_retries=0 (pour eviter le double-retry avec le mecanisme interne de
LangChain, qui reessaierait plusieurs fois sur la MEME cle avant que notre
rotation ne s'applique).

OpenRouter renvoie des erreurs au format openai (openai.AuthenticationError /
openai.RateLimitError leves par le SDK openai sous-jacent de ChatOpenAI),
d'ou l'usage de ces exceptions plutot que celles du SDK groq.
"""
import logging
import time
from typing import Any, List, Optional

from openai import AuthenticationError, RateLimitError
from langchain_core.messages import BaseMessage
from langchain_core.outputs import ChatResult
from langchain_openai import ChatOpenAI

from config import get_settings
from .openrouter_key_pool import get_next_openrouter_key_with_name

logger = logging.getLogger("sanad.ai")

# Plafond PAR APPEL individuel, pas pool_size() -- evite qu'une panne
# OpenRouter globale ne declenche des dizaines d'appels HTTP en cascade sur
# un agent a plusieurs tours.
MAX_RETRIES_PER_CALL = 3


class RetryableChatOpenRouter(ChatOpenAI):
    """ChatOpenAI (endpoint OpenRouter) qui bascule sur une autre cle du pool
    en cas de 401/429, limite a MAX_RETRIES_PER_CALL tentatives par appel de
    generation."""

    def __init__(self, **kwargs):
        settings = get_settings()
        _, first_key = get_next_openrouter_key_with_name()
        super().__init__(
            api_key=first_key,
            base_url=settings.OPENROUTER_BASE_URL,
            max_retries=0,
            **kwargs,
        )

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[Any] = None,
        **kwargs: Any,
    ) -> ChatResult:
        settings = get_settings()
        # self.model_name / self.temperature / self.max_tokens sont les vrais
        # champs pydantic, deja valides normalement par super().__init__()
        # ci-dessus.
        model = self.model_name
        temperature = self.temperature
        max_tokens = self.max_tokens

        last_exception: Optional[Exception] = None
        for attempt in range(1, MAX_RETRIES_PER_CALL + 1):
            env_name, api_key = get_next_openrouter_key_with_name()
            try:
                llm = ChatOpenAI(
                    model=model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    api_key=api_key,
                    base_url=settings.OPENROUTER_BASE_URL,
                    max_retries=0,  # notre rotation de cle remplace le retry interne de LangChain
                )
                return llm._generate(messages, stop=stop, run_manager=run_manager, **kwargs)
            except (AuthenticationError, RateLimitError) as exc:
                logger.warning(
                    "Erreur OpenRouter (%s) avec la cle '%s' -- tentative %s/%s, rotation vers la cle suivante.",
                    exc.__class__.__name__, env_name, attempt, MAX_RETRIES_PER_CALL,
                )
                last_exception = exc
                time.sleep(0.3 * attempt)  # backoff simple, croissant
                continue

        logger.error("RetryableChatOpenRouter : echec apres %s tentatives.", MAX_RETRIES_PER_CALL)
        raise last_exception or RuntimeError("Echec de generation apres toutes les tentatives.")