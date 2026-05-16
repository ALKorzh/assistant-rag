import logging
import time
from collections.abc import Callable
from typing import TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


def call_with_retry(
    operation: Callable[[], T],
    *,
    attempts: int,
    base_delay_seconds: float,
    operation_name: str,
) -> T:
    last_exc: BaseException | None = None
    for attempt in range(1, attempts + 1):
        try:
            return operation()
        except Exception as exc:
            last_exc = exc
            if attempt == attempts:
                logger.exception("%s failed after %d attempts", operation_name, attempts)
                raise
            delay = base_delay_seconds * (2 ** (attempt - 1))
            logger.warning(
                "%s attempt %d/%d failed (%s), retrying in %.2fs",
                operation_name,
                attempt,
                attempts,
                exc,
                delay,
            )
            time.sleep(delay)
    raise RuntimeError(f"{operation_name} failed") from last_exc
