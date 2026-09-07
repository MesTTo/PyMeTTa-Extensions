# metta-numpy

NumPy as the default array library of the MeTTa Python seat.

`pymetta` names no third-party library. This package is one row against the `array` point, and the seat
finds it exactly as it finds a package written by somebody else: through the
`metta.extensions` entry point, loaded at the first dispatch that has no answer
without it, or through `import metta_numpy`, whose module body registers the same
rows.

```sh
pip install metta-numpy          # or: pip install 'pymetta[arrays]'
```

Its tests are in `tests/`, and they run in the workspace suite:

```sh
sh extensions/python/test.sh ext/metta-numpy
```
