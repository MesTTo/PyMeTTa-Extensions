"""Purpose: pyarrow as the writer and reader of the Arrow IPC stream, in one row.

A gateway response body carrying answers is an Arrow IPC STREAM, which is a
FlatBuffers envelope: the C-struct builder behind the local capsule doors does
not write it, so this is a second point beside `arrow` and a second package
beside `metta-nanoarrow`. One row against the seat's `ipc` point, an ownership
point: this row claims when pyarrow is importable and declines otherwise.

pyarrow is the IPC streaming format's own reference implementation, which is
why it is the row here; a CONSUMER that reads the capsule needs no pyarrow at
all.

Assumes:
  - a projection's kinds are the seam's own five words, mapped here to pyarrow
    types through `seam.ARROW_KINDS`
Guarantees:
  - one chunk is one COMPLETE stream -- schema message, one batch, end marker
    -- because a response body is what `pyarrow.ipc.open_stream` is handed and
    a fragment is not readable on its own
    [source: https://arrow.apache.org/docs/format/Columnar.html#ipc-streaming-format;
    tested: tests/test_pyarrow.py; commit=WORKTREE]
  - every field carries its MeTTa type in `metta.type` metadata, and a column
    that is text because nothing declared it also carries `metta.kind=mixed`
    [tested: tests/test_pyarrow.py::test_a_field_carries_its_metta_type;
    commit=WORKTREE]
Open Obligations:
  To Do: None
  Hacks: None
  Future Enhancements: None
"""

from __future__ import annotations

from typing import Any, Final

from metta import seam

_MISSING: Final = (
    "the Arrow IPC stream is encoded with pyarrow, which is not installed; "
    "install pymetta[arrow]. The local capsule doors need only a C-struct "
    "builder; a stream of BYTES needs something that writes the format"
)

optional_module = seam.at("optional-module").call()

INT64, FLOAT64, BOOL, UTF8, TEXT = seam.ARROW_KINDS


def _claims() -> Any:
    """pyarrow, when it is installed; nothing otherwise."""
    return optional_module("pyarrow")


def _types(pa: Any) -> dict[str, Any]:
    """Each of the seam's five column kinds as a pyarrow type."""
    # TEXT and UTF8 are both utf8 and differ only in what a cell renders as,
    # which the projection's values already decide.
    return {
        INT64: pa.int64(),
        FLOAT64: pa.float64(),
        BOOL: pa.bool_(),
        UTF8: pa.string(),
        TEXT: pa.string(),
    }


def _schema(names: Any, kinds: Any, declared: Any) -> Any:
    """The schema for a set of columns, each field carrying its MeTTa type.

    `declared` is one MeTTa type name per column and rides in the field's
    metadata under `metta.type`; a column whose kind is text because nothing
    declared it also carries `metta.kind=mixed`, the word for a column that
    holds whatever its cells hold.
    """
    pa = _claims()
    types = _types(pa)
    fields = []
    for name, kind, kind_name in zip(names, kinds, declared, strict=True):
        metadata = {"metta.type": kind_name}
        if kind == TEXT:
            metadata["metta.kind"] = "mixed"
        fields.append(pa.field(name, types[kind], metadata=metadata))
    return pa.schema(fields)


def _stream(schema: Any, columns: Any) -> bytes:
    """One complete IPC stream: the schema message, one batch, the end marker.

    Complete rather than a fragment, because a response BODY is what
    `pyarrow.ipc.open_stream` is handed and a fragment is not readable on its
    own; a cursor's chunks are therefore one stream each with the same schema,
    which is the shape Arrow Flight's DoGet already has
    [source: https://arrow.apache.org/docs/format/Columnar.html#ipc-streaming-format].
    An empty chunk is still a stream, so a consumer reads a schema either way.
    """
    pa = _claims()
    sink = pa.BufferOutputStream()
    with pa.ipc.new_stream(sink, schema) as writer:
        if columns and columns[0]:
            writer.write_batch(
                pa.record_batch(
                    [
                        pa.array(values, type=field.type)
                        for values, field in zip(columns, schema, strict=True)
                    ],
                    schema=schema,
                )
            )
    return sink.getvalue().to_pybytes()


def _read(raw: bytes) -> Any:
    """One IPC stream's record batches, as a pyarrow Table."""
    pa = _claims()
    with pa.ipc.open_stream(pa.BufferReader(raw)) as reader:
        return reader.read_all()


def _concat(tables: Any) -> Any:
    """The drained chunks of one cursor as one table."""
    return _claims().concat_tables(list(tables))


def register() -> None:
    """This package's one row, against the seat's `ipc` point."""
    seam.ipc.register(
        "pyarrow",
        claims=_claims,
        schema=_schema,
        stream=_stream,
        read=_read,
        concat=_concat,
        missing=_MISSING,
    )


register()
