from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper


WIKIPEDIA_LANGUAGE = "ru"
WIKIPEDIA_RESULTS_LIMIT = 1
WIKIPEDIA_CONTENT_LIMIT = 2_000

_api_wrapper = WikipediaAPIWrapper(
    lang=WIKIPEDIA_LANGUAGE,
    top_k_results=WIKIPEDIA_RESULTS_LIMIT,
    doc_content_chars_max=WIKIPEDIA_CONTENT_LIMIT,
)
_wiki_tool = WikipediaQueryRun(api_wrapper=_api_wrapper)


def get_wikipedia_info(query: str) -> str:
    """Search Russian Wikipedia and return the best matching summary."""
    try:
        return _wiki_tool.run(query)
    except Exception:
        return f"Ничего не найдено в Википедии по запросу {query}."
