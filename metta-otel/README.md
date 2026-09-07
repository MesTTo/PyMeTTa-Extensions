# metta-otel

A MeTTa reduction trace as OpenTelemetry spans, and its counters as metrics.

`pymetta` names no third-party library. This package is registers nothing; it CALLS the seat's `observe` service, which holds the engine's one trace session, and the seat
finds it exactly as it finds a package written by somebody else: through the
`metta.extensions` entry point, loaded at the first dispatch that has no answer
without it, or through `import metta_otel`, whose module body registers the same
rows.

```sh
pip install metta-otel          # or: pip install 'pymetta[otel]'
```

Its tests are in `tests/`, and they run in the workspace suite:

```sh
sh extensions/python/test.sh ext/metta-otel
```
