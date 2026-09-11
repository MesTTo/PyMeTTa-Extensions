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

The wheel also advertises `lib_arrays` through `metta.libraries`. Its face
derives `arrays-is-array` and its arrow and documentation from `is_array`:

```python
import importlib
from metta import MeTTa, importing

with MeTTa() as m:
    with importing.install(m.self):
        arrays = importlib.import_module("lib_arrays")
        assert m.eval(arrays.arrays_is_array((1, 2, 3))) == [False]
```

The `face-sync` check includes faces shipped under `extensions/python/ext/`.
Regenerate the face with `python extensions/python/tools/facegen.py --write
extensions/python/ext/metta-arrays/metta_arrays_library/lib_arrays.metta`.
The wheel test builds this distribution and executes the installed face.
