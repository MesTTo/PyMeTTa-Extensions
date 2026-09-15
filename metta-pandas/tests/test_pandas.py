"""Purpose: prove the pandas row builds a frame and installs its accessor.

Guarantees: conversion sugar is declared by a door row beside its frame
  provider [tested: test_the_row_is_registered_against_the_frame_point;
  commit=b615b5a33b43252ef9826e5387da7c9bd7f6b543].

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

from metta import G, Rows, S, seam

pandas = pytest.importorskip("pandas")


@pytest.fixture()
def people():
    """Two answers with a symbol column and a number column."""
    return Rows(("who", "n"), [(S.Ada, G(1)), (S.Bob, G(2))])


def test_the_row_is_registered_against_the_frame_point():
    """The frame provider and receiver sugar have separate declared roles."""
    row = seam.frame.find("pandas")
    assert row is not None
    assert row.module == "pandas"
    contract = next(door for door in seam.door.find("metta-pandas").doors if door.key == "rows:to-df")
    assert contract.sugar_of.base == "rows:to"
    assert contract.sugar_of.fixed == (("library", "pandas"),)
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


@pytest.mark.parametrize("values", [[], [1, 1, 2]])
def test_frame_rows_use_the_declared_native_extractor(scratch_space, monkeypatch, values):
    """The provider keeps native extraction reachable for empty and duplicate rows."""
    from metta import tables
    from metta._catalog import arrow

    def refuse_arrow(_source):
        msg = "native frame extraction must not open the Arrow reader"
        raise AssertionError(msg)

    monkeypatch.setattr(arrow, "read_batches", refuse_arrow)
    row = seam.frame.find("pandas")
    assert row.rows(object()) is None
    frame = pandas.DataFrame({"value": values})
    assert list(row.rows(frame)) == [(value,) for value in values]
    assert tables.add(scratch_space, "native", frame) == len(values)
    assert len(scratch_space.atoms()) == len(values)
