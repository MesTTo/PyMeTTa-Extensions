# metta-websocket

websocket-client's own timeouts, so an absent MeTTa backend reads as absent.

`pymetta` names no third-party library. This package is one row against the `transport-error` point, and the seat
finds it exactly as it finds a package written by somebody else: through the
`metta.extensions` entry point, loaded at the first dispatch that has no answer
without it, or through `import metta_websocket`, whose module body registers the same
rows.

```sh
pip install metta-websocket          # or: pip install 'pymetta[das]'
```

Its tests are in `tests/`, and they run in the workspace suite:

```sh
sh extensions/python/test.sh ext/metta-websocket
```
