"""Purpose: nanoarrow as the builder of the Arrow C structs, in one row.

`rows.__arrow_c_schema__()` and `rows.__arrow_c_stream__()` are the PyCapsule
interface every Arrow consumer already reads; somebody has to BUILD the two C
structs from buffers, and this package is one answer. One row against the
seat's `arrow` point, which is an ownership point: this row claims when
nanoarrow is importable and declines otherwise.

A CONSUMER of the capsules needs no row and no nanoarrow: it reads whatever
Arrow support it already has. This is only who makes them, which is why
nanoarrow rather than pyarrow is the light dependency for a producer.

Assumes:
  - a projection whose `kinds` are the seam's own five words, mapped here to
    nanoarrow types through `seam.ARROW_KINDS`
Guarantees:
  - fields are built one at a time rather than from a name-to-type mapping, so
    a bridge declaration naming one table column twice keeps both, which a
    dict would silently drop [tested: tests/test_nanoarrow.py; commit=94057a0f073c0fab0a35c42beff2c324d8a0addd]
  - a requested schema this producer cannot satisfy exactly is IGNORED rather
    than refused, which is the interface's own best-effort rule
    [source: https://arrow.apache.org/docs/format/CDataInterface/PyCapsuleInterface.html;
    tested: tests/test_nanoarrow.py::test_an_unsatisfiable_request_is_ignored;
    commit=94057a0f073c0fab0a35c42beff2c324d8a0addd]
Open Obligations:
  To Do: None
  Hacks: None
  Future Enhancements: None
"""

from __future__ import annotations

from typing import Any, Final

from metta import seam

_MISSING: Final = (
    "the Arrow doors build the C structs with nanoarrow, which is not "
    "installed; install pymetta[arrow]. A consumer needs no pyarrow, only "
    "its own Arrow support"
)

require_module = seam.at("module").call()
optional_module = seam.at("optional-module").call()
_batch_bounds = seam.at("batch-bounds").call()

INT64, FLOAT64, BOOL, UTF8, TEXT = seam.ARROW_KINDS


def _claims() -> Any:
    """nanoarrow, when it is installed; nothing otherwise.

    nanoarrow builds both C structs from buffers, so a producer needs no
    pyarrow and a consumer needs only whatever Arrow support it already has.
    """
    return optional_module("nanoarrow")


def _types(na: Any) -> dict[str, Any]:
    """Each of the seam's five column kinds as a nanoarrow type."""
    return {
        INT64: na.int64(),
        FLOAT64: na.float64(),
        BOOL: na.bool_(),
        UTF8: na.string(),
        TEXT: na.string(),
    }


def _schema_of(na: Any, projection: Any) -> Any:
    """The struct CSchema for a projection.

    Fields are built one at a time rather than from a name-to-type mapping,
    because a bridge declaration may name one table column twice and Arrow
    allows the duplicate where a dict would silently drop it.
    """
    types = _types(na)
    fields = [
        na.Schema(types[kind], name=name)
        for name, kind in zip(projection.names, projection.kinds, strict=True)
    ]
    return na.c_schema(na.struct(fields))


def _schema(projection: Any) -> Any:
    """The "arrow_schema" PyCapsule for a projection."""
    na = require_module("nanoarrow", _MISSING)
    return _schema_of(na, projection).__arrow_c_schema__()


def _honour(na: Any, projection: Any, requested_schema: Any) -> Any:
    """The projection a requested schema asks for, or the derived one.

    Best-effort by the interface's own rule: a request this producer cannot
    satisfy exactly is ignored rather than refused, and a consumer that cares
    reads the schema it actually got.
    """
    if requested_schema is None:
        return projection
    try:
        wanted = na.c_schema(requested_schema)
        children = list(wanted.children)
    except Exception:  # noqa: BLE001  -- an unreadable request is a request this producer ignores
        return projection
    if wanted.format != "+s" or len(children) != len(projection.names):
        return projection
    if [child.name for child in children] != list(projection.names):
        return projection
    kinds = [seam.ARROW_FORMAT.get(child.format, "") for child in children]
    return projection.retyped(kinds) or projection


def _stream(projection: Any, requested_schema: Any = None) -> Any:
    """The "arrow_array_stream" PyCapsule for a projection."""
    from nanoarrow.c_array_stream import CArrayStream  # noqa: PLC0415  -- its own

    na = require_module("nanoarrow", _MISSING)
    projection = _honour(na, projection, requested_schema)
    schema = _schema_of(na, projection)
    types = _types(na)
    batches = [
        na.c_array_from_buffers(
            schema,
            stop - start,
            [None],
            children=[
                na.c_array(projection.values(index, start, stop), types[kind])
                for index, kind in enumerate(projection.kinds)
            ],
        )
        for start, stop in _batch_bounds(projection.length)
    ]
    # Every batch was built from `schema` itself, so type equality holds by
    # construction and the per-batch re-check would only re-derive it.
    return CArrayStream.from_c_arrays(batches, schema, validate=False).__arrow_c_stream__()


def _batches(source: Any) -> tuple[tuple[str, ...], Any]:
    """The stream's column names, and an iterator of its record batches."""
    na = require_module("nanoarrow", _MISSING)
    stream = na.ArrayStream(source)
    schema = stream.schema
    if schema.type != na.Type.STRUCT:
        stream.close()
        msg = (
            f"an Arrow stream of rows is a stream of struct arrays; this one "
            f"carries {schema.type}, which has no columns to become an atom's "
            f"arguments"
        )
        raise TypeError(msg)
    names = tuple(field.name for field in schema.fields)

    def batches():
        with stream:
            for chunk in stream.iter_chunks():
                yield list(chunk.iter_tuples())

    return names, batches()


def register() -> None:
    """This package's one row, against the seat's `arrow` point."""
    seam.arrow.register(
        "nanoarrow",
        claims=_claims,
        schema=_schema,
        stream=_stream,
        batches=_batches,
        missing=_MISSING,
    )


register()
