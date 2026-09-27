# Security policy

## Supported versions

Only the latest commit on `main` is supported. This is a portfolio project, not a product with
release branches.

## Reporting a vulnerability

Please do not open a public issue for a security problem. Use GitHub's private vulnerability
reporting ("Report a vulnerability" under the repository's Security tab). Include:

- the affected commit and component;
- steps to reproduce, ideally as a failing test;
- the impact you expect.

You can expect an acknowledgement within 7 days and a fix or a written assessment within 30 days.
Credit is given in the fix commit unless you ask otherwise.

## Scope

In scope: the code in this repository, its container image and its CI configuration.
Out of scope: denial of service by traffic volume, findings that need a compromised host, and
issues in third-party dependencies that are already publicly tracked (please report those
upstream).

## What an attacker would want here

This is a release control plane: it decides whether a new model gets customer traffic. Two assets
follow from that, and they shape the controls below.

- **The decision.** Getting a bad model promoted, or a good one rolled back.
- **The evidence the decision rests on.** The prediction log, the delayed labels and the drift
  references. Poison the evidence and the decision follows honestly.

## How this repository is hardened

- **Model supply chain.** Releases are content-addressed and Ed25519-signed; the signature is
  verified against a trust store of public keys *before* an artifact is parsed, and the parsed
  booster's feature names must match the signed manifest. Artifacts are LightGBM text; there is no
  pickle path and a test fails the build if a pickle-family import appears.
- **Separation of duties.** A deployer proposes, a release manager who is not the proposer approves
  the final promotion, and forcing a stage past its statistical gate needs two distinct release
  managers. Automatic rollback needs nobody, always.
- **Tamper-evident decisions.** Every proposal, approval, rollback and override is an entry in a
  hash-chained log carrying the evidence it was based on and the hash of the policy it was judged
  under. `progressive-model-delivery audit verify` checks the chain; `audit export` writes a head
  you can keep off-box so truncation is detectable.
- **Untrusted challenger code.** The shadow worker runs in its own container on a network with no
  gateway, holds no credentials with side effects, and is reached only through a bounded
  drop-on-overflow queue with a hard timeout. A crashing or hanging challenger costs dropped shadow
  jobs and nothing else.
- **Metric-poisoning defence.** A label is accepted only for a request id the router issued, only
  inside the maturation window, only once, and under its own rate limit.
- **Request surface.** API keys stored as SHA-256 digests and compared with no early exit; pandera
  and pydantic contracts; per-client rate limits; body-size limits; security headers; strict CORS;
  errors that carry no internals.
- **Privacy.** Subject identifiers are HMAC-hashed before assignment and before anything is logged.
  Logs pass through a redacting processor. Secrets are `SecretStr` from the environment, and only
  `.env.example` is committed.
- **Threats, controls and the tests that prove them:**
  [docs/THREAT_MODEL.md](docs/THREAT_MODEL.md). Every security claim is linked to evidence in
  [CLAIMS.md](CLAIMS.md), and what the statistical guarantees do and do not cover is in
  [docs/ROLLOUT_POLICY.md](docs/ROLLOUT_POLICY.md).
- **CI** runs ruff, mypy --strict, pytest with a coverage floor, bandit, semgrep (registry packs
  plus `.semgrep/rules.yml`), pip-audit, gitleaks, a trivy image scan, a syft CycloneDX SBOM, and an
  end-to-end compose job that asserts the shadow worker cannot open an outbound connection.

## Known limitations

These are deliberate and are not vulnerabilities to report:

- The signing private key is a file and the trust store is an environment variable. There is no KMS,
  HSM or key revocation list.
- Role bindings are API keys. One person holding two keys defeats the two-person rules; that is an
  operational control, not a technical one.
- The audit chain is an unkeyed SHA-256 chain, so it detects edits by anyone who cannot recompute
  it. Write access to `control.sqlite` is enough to forge a consistent replacement chain, whether
  shorter or different; the head that `audit --export` records off-box is the only defence against
  both. Signing entries with the release key would make the chain authenticated; that is not done.
- Rate limiting is per process and in memory. Behind a proxy, all clients share one address unless
  proxy headers are trusted.
- Only the subgroups in the data contract are monitored. The README Non-goals section lists the
  rest.
