from functools import lru_cache

from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper

from app.core.config import get_settings


@lru_cache(maxsize=1)
def _wiki_runner() -> WikipediaQueryRun:
    settings = get_settings()
    api_wrapper = WikipediaAPIWrapper(
        lang=settings.wikipedia_language,
        top_k_results=settings.wikipedia_results_limit,
        doc_content_chars_max=settings.wikipedia_content_limit,
    )
    return WikipediaQueryRun(api_wrapper=api_wrapper)


def get_wikipedia_info(query: str) -> str:
    """Search Russian Wikipedia and return the best matching summary."""
    try:
        return _wiki_runner().run(query)
    except Exception:
        return f"Ничего не найдено в Википедии по запросу {query}."
