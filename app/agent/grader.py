import logging
import re
from functools import lru_cache

import pymorphy3

from app.agent.state import RelevanceCheck

logger = logging.getLogger(__name__)

_TOKEN_PATTERN = re.compile(r"\b[а-яёa-z0-9]{2,}\b")
_NUMBER_PATTERN = re.compile(r"\b\d+\b")
_EDUCATION_TERMS = frozenset({"лабораторный", "работа", "задание", "курс", "тема"})
_RELEVANCE_THRESHOLD = 0.3


@lru_cache(maxsize=1)
def _morph() -> pymorphy3.MorphAnalyzer:
    logger.info("Initializing pymorphy3 analyzer")
    return pymorphy3.MorphAnalyzer()


def _get_lemmas(text: str) -> list[str]:
    morph = _morph()
    return [morph.parse(token)[0].normal_form for token in _TOKEN_PATTERN.findall(text.lower())]


def advanced_keyword_check(question: str, context: str) -> RelevanceCheck:
    """Fast lexical check that rejects obviously irrelevant RAG context."""
    question_lemmas = _get_lemmas(question)
    context_lemmas = _get_lemmas(context)
    question_numbers = set(_NUMBER_PATTERN.findall(question))
    context_numbers = set(_NUMBER_PATTERN.findall(context))

    if question_numbers and not question_numbers.intersection(context_numbers):
        logger.debug("Relevance check rejected by numeric mismatch")
        return {"relevant": False, "reason": "Числовое несовпадение"}

    if any(term in question_lemmas for term in _EDUCATION_TERMS) and not any(
        term in context_lemmas for term in _EDUCATION_TERMS
    ):
        logger.debug("Relevance check rejected by education lexicon mismatch")
        return {"relevant": False, "reason": "Нет учебной лексики"}

    keywords = [lemma for lemma in question_lemmas if len(lemma) > 3 and not lemma.isdigit()]
    matches = [lemma for lemma in keywords if lemma in context_lemmas]
    score = len(matches) / len(keywords) if keywords else 1.0
    logger.debug("Relevance score computed: %.2f", score)
    return {"relevant": score >= _RELEVANCE_THRESHOLD, "reason": f"Score: {score:.2f}"}
