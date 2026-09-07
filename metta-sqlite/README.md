# metta-sqlite

sqlite3 as a SQL engine a MeTTa head can be registered into.

`pymetta` names no third-party library. This package is one row against the `sql` point, and the seat
finds it exactly as it finds a package written by somebody else: through the
`metta.extensions` entry point, loaded at the first dispatch that has no answer
without it, or through `import metta_sqlite`, whose module body registers the same
rows.

```sh
pip install metta-sqlite          # or: pip install 'pymetta[sql]'
```

Its tests are in `tests/`, and they run in the workspace suite:

```sh
sh extensions/python/test.sh ext/metta-sqlite
```
