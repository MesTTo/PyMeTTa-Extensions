"""Purpose: prove the nanoarrow row builds the two Arrow C structs.

Blackbox through the PyCapsule interface a consumer reads: a schema capsule
and a stream capsule off a set of answers. What builds them is this row, and a
consumer needs nothing of it.
Open Obligations:
  To Do: None
  Hacks: None
  Future Enhancements: None
"""

from __future__ import annotations

import metta_nanoarrow  # noqa: F401  -- imported for the arrow row it registers
import pytest

from metta import G, S, seam
from metta.results import Rows

nanoarrow = pytest.importorskip("nanoarrow")


@pytest.fixture()
def people():
    """Answers over one symbol column and one number column."""
    return Rows(("who", "n"), [(S.Ada, G(1)), (S.Bob, G(2))])


def test_the_row_claims_the_arrow_point():
    """An ownership point: this row claims because its library is importable."""
    claim = seam.arrow.claim()
    assert claim is not None
    assert claim.name == "nanoarrow"


def test_the_schema_capsule_names_the_projected_columns(people):
    """The struct schema carries one field per column, in order."""
    schema = nanoarrow.c_schema(people.__arrow_c_schema__())
    assert [field.name for field in schema.children] == ["who", "n"]


def test_the_stream_capsule_carries_the_values(people):
    """The stream reads back as the rows it was built from."""
    stream = nanoarrow.ArrayStream(people)
    with stream:
        read = [row for chunk in stream.iter_chunks() for row in chunk.iter_tuples()]
    assert read == [("Ada", 1), ("Bob", 2)]


def test_a_duplicate_column_name_survives_the_schema():
    """Fields are built one at a time, where a name-to-type dict would drop one.

    Answers cannot carry a repeated column name -- `Rows` refuses one -- but a
    bridge declaration can name one table column twice, and the projection it
    builds reaches this row directly. So the projection is built through the
    seam's own service, which is the door a registrant is handed.
    """
    projection = seam.at("projection").call()(("a", "a"), [(1, 2)])
    schema = nanoarrow.c_schema(seam.arrow.claim().row.schema(projection))
    assert [field.name for field in schema.children] == ["a", "a"]


def test_an_unsatisfiable_request_is_ignored(people):
    """The interface's own best-effort rule: ignore, never refuse.

    A request for one column where the projection has two cannot be satisfied,
    so the producer answers its own schema and a consumer that cares reads
    what it actually got.
    """
    wanted = nanoarrow.c_schema(nanoarrow.struct([nanoarrow.Schema(nanoarrow.int64(), name="who")]))
    stream = nanoarrow.ArrayStream(
        people.__arrow_c_stream__(wanted.__arrow_c_schema__())
    )
    with stream:
        assert [field.name for field in stream.schema.fields] == ["who", "n"]
