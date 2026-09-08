# metta-tables

This package declares the `m.tables` accessor through `metta.seam.door`.
Its immutable `DOORS` records carry the signatures, effects, return shapes,
and evidence. Discovery imports those records. Calling an accessor loads the
existing implementation, whose exceptions and lifetime protocol apply.

Install beside the matching version of `pymetta`:

```sh
pip install metta-tables
```
