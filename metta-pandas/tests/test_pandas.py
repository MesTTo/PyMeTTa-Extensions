"""Purpose: prove the pandas row builds a frame and installs its accessor.

Blackbox through the doors a user takes: `rows.to_df()`, `rows.to(pandas)` and
`frame.metta`. Importing `metta_pandas` is one of the two sanctioned ways a
package's rows arrive, and the one a checkout can take; the other, the
`metta.extensions` entry point, is proved by the shell proof that installs a
distribution for real.
Open Obligations:
  To Do: None
  Hacks: None
  Future Enhancements: None
"""

from __future__ import annotations

import metta_pandas  # noqa: F401  -- imported for the frame row it registers
import pytest

from metta import G, S, seam
from metta.results import Rows

pandas = pytest.importorskip("pandas")


@pytest.fixture()
def people():
    """Two answers with a symbol column and a number column."""
    return Rows(("who", "n"), [(S.Ada, G(1)), (S.Bob, G(2))])


def test_the_row_is_registered_against_the_frame_point():
    """One row, named for the library, carrying the sugar it asked for."""
    row = seam.frame.find("pandas")
    assert row is not None
    assert row.module == "pandas"
    assert row.sugar == "to_df"
    assert not row.fallback


def test_to_df_builds_a_pandas_frame(people):
    """The declared sugar of this row: `rows.to_df()`."""
    frame = people.to_df()
    assert isinstance(frame, pandas.DataFrame)
    assert list(frame.columns) == ["who", "n"]
    assert list(frame["who"]) == ["Ada", "Bob"]
    assert list(frame["n"]) == [1, 2]


def test_the_general_door_reaches_the_same_frame(people):
    """`rows.to(<module>)` is the spelling every registrant gets."""
    assert people.to(pandas).equals(people.to_df())


def test_the_accessor_installs_for_an_imported_pandas(scratch_space):
    """`frame.metta` is this library's own accessor, put there by the row."""
    from metta import tables

    assert "pandas" in tables.accessors()
    frame = pandas.DataFrame({"a": [1], "b": ["x"]})
    assert frame.metta.into(scratch_space, "prow") == 1
    assert len(scratch_space.match(scratch_space.parse("(prow $a $b)"))) == 1
