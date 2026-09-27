"""structlog JSON logging with PII redaction for both structlog and stdlib (uvicorn) records."""

from __future__ import annotations

import logging
import sys
from typing import Any, TextIO, cast

import structlog
from structlog.typing import FilteringBoundLogger, Processor

from progressive_model_delivery.core.redaction import redact_processor

__all__ = ["configure_logging", "get_logger"]


def _shared_processors() -> list[Processor]:
    # Redaction runs after exception formatting so tracebacks are scrubbed too.
    return [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        redact_processor,
    ]


def configure_logging(
    level: str = "INFO", *, json_output: bool = True, stream: TextIO | None = None
) -> None:
    """Configure structlog and route stdlib logging through the same redacting pipeline."""
    level_no = logging.getLevelNamesMapping()[level.upper()]
    output: TextIO = stream if stream is not None else sys.stdout
    renderer: Processor = (
        structlog.processors.JSONRenderer(sort_keys=True)
        if json_output
        else structlog.dev.ConsoleRenderer(colors=False)
    )
    shared = _shared_processors()
    structlog.configure(
        processors=[*shared, renderer],
        wrapper_class=structlog.make_filtering_bound_logger(level_no),
        logger_factory=structlog.PrintLoggerFactory(file=output),
        cache_logger_on_first_use=False,
    )

    formatter = structlog.stdlib.ProcessorFormatter(
        foreign_pre_chain=shared,
        processors=[structlog.stdlib.ProcessorFormatter.remove_processors_meta, renderer],
    )
    handler = logging.StreamHandler(output)
    handler.setFormatter(formatter)
    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(level_no)
    for noisy in ("uvicorn", "uvicorn.error"):
        logging.getLogger(noisy).handlers = []
        logging.getLogger(noisy).propagate = True
    # Access logs would print raw query strings; RequestContextMiddleware logs paths instead.
    logging.getLogger("uvicorn.access").disabled = True


def get_logger(name: str | None = None, **initial_values: Any) -> FilteringBoundLogger:
    """Return a structlog logger that resolves the configuration when it logs.

    Deliberately *not* bound here. Every module creates its logger at import time, which happens
    before :func:`configure_logging` runs, and ``bind()`` freezes whatever configuration exists at
    that moment -- structlog's default: console renderer, stdout, INFO, and no redaction. Such a
    logger ignores ``APP_LOG_LEVEL`` and ``APP_LOG_JSON`` and prints fields in the clear that the
    redacting processor exists to remove. The lazy proxy keeps its initial values and picks up the
    configuration on the first call instead (``tests/unit/test_logging.py``).
    """
    logger = (
        structlog.get_logger(name, **initial_values)
        if name
        else structlog.get_logger(**initial_values)
    )
    return cast(FilteringBoundLogger, logger)
