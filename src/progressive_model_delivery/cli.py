"""Command line entry point: ``progressive-model-delivery`` or ``python -m <package>``.

Subcommands:

``serve``              run the router (``--role router``) or the controller and control API
                      (``--role controller``) or the shadow worker (``--role shadow``).
``demo``              the offline end-to-end demo (two rollouts, no network, no API key).
``generate-api-key``  print a new API key and its SHA-256 digest.
``role-binding``      print an ``APP_ROLE_BINDINGS`` entry for a principal and its roles.
``audit``             verify or export the hash-chained audit log of a control database.
``bootstrap``         train and publish the demo releases into a registry directory.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # imported lazily at runtime so that --help stays fast
    from fastapi import FastAPI

    from progressive_model_delivery.core.config import Settings

__all__ = ["main"]

ROLES = ("router", "controller", "shadow")


def _build_app(role: str) -> tuple[FastAPI, Settings]:
    from progressive_model_delivery.control.api import build_control_app  # noqa: PLC0415
    from progressive_model_delivery.core.config import get_settings  # noqa: PLC0415
    from progressive_model_delivery.core.logging import configure_logging  # noqa: PLC0415
    from progressive_model_delivery.serving.router import build_router_app  # noqa: PLC0415
    from progressive_model_delivery.serving.shadow_worker import (  # noqa: PLC0415
        build_shadow_app,
    )

    settings = get_settings()
    configure_logging(settings.log_level, json_output=settings.log_json)
    builders = {
        "router": build_router_app,
        "controller": build_control_app,
        "shadow": build_shadow_app,
    }
    return builders[role](settings), settings


def _serve(role: str, host: str | None, port: int | None) -> int:
    import uvicorn  # noqa: PLC0415 - only needed when serving

    from progressive_model_delivery.core.tracing import (  # noqa: PLC0415
        install_global_tracer_provider,
    )

    app, settings = _build_app(role)
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


def _demo(workdir: Path) -> int:
    import sys as _sys  # noqa: PLC0415

    from progressive_model_delivery.core.logging import configure_logging  # noqa: PLC0415
    from progressive_model_delivery.demo import run_demo  # noqa: PLC0415

    configure_logging("WARNING", json_output=True, stream=_sys.stderr)
    print(json.dumps(run_demo(workdir), indent=2, sort_keys=True))
    return 0


def _audit(action: str, database: Path, output: Path | None) -> int:
    from progressive_model_delivery.control.store import ControlStore  # noqa: PLC0415

    if not database.exists():
        print(f"no control database at {database}", file=sys.stderr)
        return 2
    store = ControlStore(database)
    if action == "export":
        target = output or database.with_suffix(".audit.jsonl")
        count = store.export_audit(target)
        print(json.dumps({"exported": count, "path": str(target)}, indent=2))
        return 0
    result = store.verify_audit()
    print(
        json.dumps(
            {
                "ok": result.ok,
                "entries": result.entries,
                "head": result.head,
                "error": result.error,
                "seq": result.seq,
            },
            indent=2,
        )
    )
    return 0 if result.ok else 1


def _bootstrap(root: Path, key_path: Path | None) -> int:
    from progressive_model_delivery.models.pipeline import build_demo_models  # noqa: PLC0415
    from progressive_model_delivery.registry.signing import SigningKey  # noqa: PLC0415

    key = SigningKey.generate()
    models = build_demo_models(root, key)
    if key_path is not None:
        key.save(key_path)
    print(
        json.dumps(
            {
                "registry_dir": str(root / "registry"),
                "public_key_b64": key.public().to_b64(),
                "private_key_file": None if key_path is None else str(key_path),
                "releases": {name: models.release_id(name) for name in sorted(models.releases)},
                "dataset_sha256": list(models.dataset_sha256),
                # Two views of quality. `pipeline_metrics` come from the validation frame the
                # training pipeline prepared itself (bugs included) and are what the signed
                # manifest records; `live_metrics` score the clean holdout, which is what serving
                # sees. They disagree exactly for the challengers with a pipeline bug -- which is
                # why offline metrics alone cannot gate a release.
                "pipeline_metrics": {
                    name: dict(sorted(models.metrics[name].items()))
                    for name in sorted(models.metrics)
                },
                "live_metrics": {
                    name: dict(sorted(models.live_metrics[name].items()))
                    for name in sorted(models.live_metrics)
                },
            },
            indent=2,
        )
    )
    if key_path is None:
        print("The signing key was not saved; pass --key-out to keep it.", file=sys.stderr)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="progressive-model-delivery")
    sub = parser.add_subparsers(dest="command", required=True)

    serve = sub.add_parser("serve", help="run one of the services")
    serve.add_argument("--role", choices=ROLES, default="router")
    serve.add_argument("--host", default=None)
    serve.add_argument("--port", type=int, default=None)

    sub.add_parser("generate-api-key", help="print a new API key and its SHA-256 digest")

    binding = sub.add_parser("role-binding", help="print an APP_ROLE_BINDINGS entry")
    binding.add_argument("principal")
    binding.add_argument("roles", help="one or more roles joined by '+' (deployer, ...)")

    demo = sub.add_parser("demo", help="run the offline end-to-end demo")
    demo.add_argument("--workdir", type=Path, default=Path("var/demo"))

    audit = sub.add_parser("audit", help="verify or export the audit chain")
    audit.add_argument("action", choices=["verify", "export"])
    audit.add_argument("--database", type=Path, default=Path("var/control.sqlite"))
    audit.add_argument("--output", type=Path, default=None)

    bootstrap = sub.add_parser("bootstrap", help="train and publish the demo releases")
    bootstrap.add_argument("--root", type=Path, default=Path("var/models"))
    bootstrap.add_argument("--key-out", type=Path, default=None)

    args = parser.parse_args(argv)
    if args.command == "serve":
        return _serve(args.role, args.host, args.port)
    if args.command == "generate-api-key":
        from progressive_model_delivery.core.security import generate_api_key  # noqa: PLC0415

        raw, digest = generate_api_key()
        print(json.dumps({"api_key": raw, "api_key_sha256": digest}, indent=2))
        print("Store only api_key_sha256 in APP_API_KEY_SHA256.", file=sys.stderr)
        return 0
    if args.command == "role-binding":
        from progressive_model_delivery.core.security import (  # noqa: PLC0415
            generate_api_key,
        )

        raw, digest = generate_api_key()
        print(json.dumps({"api_key": raw, "binding": f"{digest}:{args.principal}:{args.roles}"}))
        print("Give the api_key to the principal; keep only the binding.", file=sys.stderr)
        return 0
    if args.command == "audit":
        return _audit(args.action, args.database, args.output)
    if args.command == "bootstrap":
        return _bootstrap(args.root, args.key_out)
    return _demo(args.workdir)


if __name__ == "__main__":
    raise SystemExit(main())
