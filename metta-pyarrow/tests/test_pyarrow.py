"""Purpose: prove the pyarrow row writes and reads the Arrow IPC stream.

The IPC stream is what crosses a gateway as BYTES, and it is a different
question from the local capsules: this row writes the FlatBuffers envelope the
capsule builder does not.
Open Obligations:
  To Do: None
  Hacks: None
  Future Enhancements: None
"""

from __future__ import annotations

import metta_pyarrow  # noqa: F401  -- imported for the ipc row it registers
import pytest

from metta import seam

pa = pytest.importorskip("pyarrow")


@pytest.fixture()
def row():
    """This package's own row, which is what the core reaches through."""
    claim = seam.ipc.claim()
    assert claim is not None, "the ipc point has no claimant"
    return claim.row


def test_the_row_claims_the_ipc_point():
    """An ownership point: this row claims because its library is importable."""
    claim = seam.ipc.claim()
    assert claim is not None
    assert claim.name == "pyarrow"


def test_a_field_carries_its_metta_type(row):
    """One MeTTa type name per column rides in the field's own metadata."""
    schema = row.schema(("n", "who"), ("int64", "text"), ("Number", "Atom"))
    assert schema.field("n").metadata[b"metta.type"] == b"Number"
    assert schema.field("who").metadata[b"metta.kind"] == b"mixed"


def test_a_stream_round_trips_through_the_codec(row):
    """Schema message, one batch, end marker: a complete, readable stream."""
    schema = row.schema(("n",), ("int64",), ("Number",))
    table = row.read(row.stream(schema, ([1, 2, 3],)))
    assert table.column("n").to_pylist() == [1, 2, 3]


def test_an_empty_chunk_is_still_a_stream(row):
    """A consumer reads a schema either way, which a fragment would not give."""
    schema = row.schema(("n",), ("int64",), ("Number",))
    table = row.read(row.stream(schema, ([],)))
    assert table.num_rows == 0
    assert table.schema.names == ["n"]


def test_the_drained_chunks_concatenate(row):
    """A cursor's chunks are one stream each; concat is how they become one."""
    schema = row.schema(("n",), ("int64",), ("Number",))
    chunks = [row.read(row.stream(schema, ([1, 2],))), row.read(row.stream(schema, ([3],)))]
    assert row.concat(chunks).column("n").to_pylist() == [1, 2, 3]
