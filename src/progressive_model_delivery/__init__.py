"""Progressive model delivery: shadow, weighted canary and production stages for ML models,
governed by anytime-valid sequential tests with automatic rollback.

Sub-packages:
- ``core``: settings, logging with PII redaction, tracing, metrics, hardened FastAPI app factory.
- ``ml``: data contracts, run manifests, content-addressed store, drift monitors, safe I/O.
- ``stats``: e-values (beta-binomial mixtures), mixture SPRT, confidence sequences, power.
- ``models``: synthetic population with delayed labels, LightGBM demo models and challengers.
- ``registry``: Ed25519-signed releases, verify-before-load, guarded lifecycle state machine.
- ``serving``: router, consistent-hash assignment, shadow forwarder and worker, prediction log.
- ``control``: RolloutSpec, SQLite control plane (lease, audit chain, approvals), controller.
- ``simulation``: in-process traffic simulator and the vectorized A/A and power study.
"""

__version__ = "0.1.0"

__all__ = ["__version__"]
