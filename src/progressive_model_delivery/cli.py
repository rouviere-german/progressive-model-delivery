"""Command line entry point: ``progressive-model-delivery`` or ``python -m <package>``."""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import sys
from pathlib import Path

__all__ = ["main"]

_DEMOS = {
    "ml": ("progressive_model_delivery.ml.demo", "run_ml_demo"),
}


def _serve(host: str | None, port: int | None) -> int:
    import uvicorn  # noqa: PLC0415 - only needed when serving

    from progressive_model_delivery.core.app import create_app  # noqa: PLC0415
    from progressive_model_delivery.core.config import get_settings  # noqa: PLC0415
    from progressive_model_delivery.core.logging import configure_logging  # noqa: PLC0415
    from progressive_model_delivery.core.tracing import (  # noqa: PLC0415
        install_global_tracer_provider,
    )

    settings = get_settings()
    configure_logging(settings.log_level, json_output=settings.log_json)
    app = create_app(settings)
    install_global_tracer_provider(app.state.tracer_provider)
    uvicorn.run(
        app,
        host=host or settings.host,
        port=port or settings.port,
        log_config=None,
        access_log=False,
        server_header=False,
    )
    return 0


def _demo(kit: str, workdir: Path) -> int:
    from progressive_model_delivery.core.logging import configure_logging  # noqa: PLC0415

    configure_logging("WARNING", json_output=True, stream=sys.stderr)
    selected = list(_DEMOS) if kit == "all" else [kit]
    ran = 0
    for name in selected:
        module_name, function_name = _DEMOS[name]
        if importlib.util.find_spec(module_name) is None:
            continue  # kit removed from this repo
        function = getattr(importlib.import_module(module_name), function_name)
        summary = function(workdir / name)
        print(json.dumps({"demo": name, **summary}, indent=2, sort_keys=True))
        ran += 1
    return 0 if ran else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="progressive-model-delivery")
    sub = parser.add_subparsers(dest="command", required=True)

    serve = sub.add_parser("serve", help="run the HTTP service")
    serve.add_argument("--host", default=None)
    serve.add_argument("--port", type=int, default=None)

    sub.add_parser("generate-api-key", help="print a new API key and its SHA-256 digest")

    demo = sub.add_parser("demo", help="run the offline demos")
    demo.add_argument("--kit", choices=["all", *sorted(_DEMOS)], default="all")
    demo.add_argument("--workdir", type=Path, default=Path("var/demo"))

    args = parser.parse_args(argv)
    if args.command == "serve":
        return _serve(args.host, args.port)
    if args.command == "generate-api-key":
        from progressive_model_delivery.core.security import generate_api_key  # noqa: PLC0415

        raw, digest = generate_api_key()
        print(json.dumps({"api_key": raw, "api_key_sha256": digest}, indent=2))
        print("Store only api_key_sha256 in APP_API_KEY_SHA256.", file=sys.stderr)
        return 0
    return _demo(args.kit, args.workdir)


if __name__ == "__main__":
    raise SystemExit(main())
