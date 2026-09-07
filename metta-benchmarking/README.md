# metta-benchmarking

Benchmark plumbing for MeTTa: counter baselines, instruction pins, slopes.

`pymetta` names no third-party library. This package is registers nothing; it measures a workload from outside through the public space surface, and the seat
finds it exactly as it finds a package written by somebody else: through the
`metta.extensions` entry point, loaded at the first dispatch that has no answer
without it, or through `import metta_benchmarking`, whose module body registers the same
rows.

```sh
pip install metta-benchmarking          # or: pip install 'pymetta[test]'
```

Its tests are in `tests/`, and they run in the workspace suite:

```sh
sh extensions/python/test.sh ext/metta-benchmarking
```
