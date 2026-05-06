import logging
from functools import lru_cache

from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper

from app.core.config import get_settings

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _wiki_runner() -> WikipediaQueryRun:
    settings = get_settings()
    logger.info("Initializing Wikipedia tool wrapper")
    api_wrapper = WikipediaAPIWrapper(
        lang=settings.wikipedia_language,
        top_k_results=settings.wikipedia_results_limit,
        doc_content_chars_max=settings.wikipedia_content_limit,
    )
    return WikipediaQueryRun(api_wrapper=api_wrapper)


def get_wikipedia_info(query: str) -> str:
    """Search Russian Wikipedia and return the best matching summary."""
    logger.info("Wikipedia query started, query_length=%d", len(query))
    try:
        result = _wiki_runner().run(query)
        logger.info("Wikipedia query completed")
        return result
    except Exception:
        logger.exception("Wikipedia query failed")
        return f"Ничего не найдено в Википедии по запросу {query}."
