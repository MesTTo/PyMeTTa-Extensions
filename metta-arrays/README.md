# metta-arrays

Arrays as atoms for every library speaking DLPack and the Python array API.

`pymetta` names no third-party library. This package is declares nothing; it CONSUMES the seat's `array` and `index` points and registers one fallback index backend of its own, the Array API path, and the seat
finds it exactly as it finds a package written by somebody else: through the
`metta.extensions` entry point, loaded at the first dispatch that has no answer
without it, or through `import metta_arrays`, whose module body registers the same
rows.

```sh
pip install metta-arrays          # or: pip install 'pymetta[arrays]'
```

Its tests are in `tests/`, and they run in the workspace suite:

```sh
sh extensions/python/test.sh ext/metta-arrays
```
