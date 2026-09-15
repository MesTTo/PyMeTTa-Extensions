# metta-polars

polars as a frame library of the MeTTa Python seat.

`pymetta` names no third-party library. This package is one row against the `frame` point, and the seat
finds it exactly as it finds a package written by somebody else: through the
`metta.extensions` entry point, loaded at the first dispatch that has no answer
without it, or through `import metta_polars`, whose module body registers the same
rows.

```sh
pip install metta-polars          # or: pip install 'pymetta[dataframes]'
```

The frame row declares polars' native `iter_rows()` extraction.
`frame.metta.into(space, "row")` reads and writes the whole frame in one
transaction. Foreign stores require transactional writes; nested ingestion
also requires provider savepoints. Input and cleanup failures roll back the
load before its transaction commits.

Its tests are in `tests/`, and they run in the workspace suite:

```sh
sh extensions/python/test.sh ext/metta-polars
```
