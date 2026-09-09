"""Purpose: prove the polars row builds a frame through the Arrow view.

Guarantees: conversion sugar is declared by a door row beside its frame
  provider [tested: test_the_row_is_registered_against_the_frame_point;
  commit=b615b5a33b43252ef9826e5387da7c9bd7f6b543].

Blackbox: `rows.to_pl()`, `rows.to(polars)` and `frame.metta`. The Arrow path
is the one that matters here, because polars' constructor tests for a sequence
before it looks for the capsule and the view is what reaches the capsule.
Open Obligations:
  To Do: None
  Hacks: None
  Future Enhancements: None
"""

from __future__ import annotations

import metta_polars  # noqa: F401  -- imported for the frame row it registers
import pytest

from metta import G, Rows, S, seam

polars = pytest.importorskip("polars")


@pytest.fixture()
def people():
    """Two answers with a symbol column and a number column."""
    return Rows(("who", "n"), [(S.Ada, G(1)), (S.Bob, G(2))])


def test_the_row_is_registered_against_the_frame_point():
    """The frame provider and receiver sugar have separate declared roles."""
    row = seam.frame.find("polars")
    assert row is not None
    assert row.module == "polars"
    contract = next(door for door in seam.door.find("metta-polars").doors if door.key == "rows:to-pl")
    assert contract.sugar_of.base == "rows:to"
    assert contract.sugar_of.fixed == (("library", "polars"),)


def test_to_pl_builds_a_polars_frame(people):
    """The declared sugar of this row: `rows.to_pl()`."""
    frame = people.to_pl()
    assert isinstance(frame, polars.DataFrame)
    assert frame.columns == ["who", "n"]
    assert frame["who"].to_list() == ["Ada", "Bob"]
    assert frame["n"].to_list() == [1, 2]


def test_the_general_door_reaches_the_same_frame(people):
    """`rows.to(<module>)` is the spelling every registrant gets."""
    assert people.to(polars).equals(people.to_pl())


def test_the_accessor_installs_for_an_imported_polars(scratch_space):
    """`frame.metta` is this library's own namespace, put there by the row."""
    from metta import tables

    assert "polars" in tables.accessors()
    frame = polars.DataFrame({"a": [1], "b": ["x"]})
    assert frame.metta.into(scratch_space, "prow") == 1
    assert len(scratch_space.match(scratch_space.parse("(prow $a $b)"))) == 1
