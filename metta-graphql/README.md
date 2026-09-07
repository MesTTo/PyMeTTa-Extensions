# metta-graphql

Executing a GraphQL query against a served MeTTa space, through graphql-core.

`pymetta` names no third-party library. This package is one row against the `graphql` point, and the seat
finds it exactly as it finds a package written by somebody else: through the
`metta.extensions` entry point, loaded at the first dispatch that has no answer
without it, or through `import metta_graphql`, whose module body registers the same
rows.

```sh
pip install metta-graphql          # or: pip install 'pymetta[graphql]'
```

Its tests are in `tests/`, and they run in the workspace suite:

```sh
sh extensions/python/test.sh ext/metta-graphql
```
