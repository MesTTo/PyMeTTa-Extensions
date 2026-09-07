# metta-pyarrow

pyarrow as the writer and reader of MeTTa's Arrow IPC streams.

`pymetta` names no third-party library. This package is one row against the `ipc` point, and the seat
finds it exactly as it finds a package written by somebody else: through the
`metta.extensions` entry point, loaded at the first dispatch that has no answer
without it, or through `import metta_pyarrow`, whose module body registers the same
rows.

```sh
pip install metta-pyarrow          # or: pip install 'pymetta[arrow]'
```

Its tests are in `tests/`, and they run in the workspace suite:

```sh
sh extensions/python/test.sh ext/metta-pyarrow
```
