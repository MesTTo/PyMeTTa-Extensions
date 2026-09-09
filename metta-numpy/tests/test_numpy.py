"""Purpose: prove the NumPy row is the array point's default and its scalars.

The row says three things: NumPy may be the no-argument default, what its
absence should say, and how to generate its own scalar values. Each is checked
here through the door that reads it.
Open Obligations:
  To Do: None
  Hacks: None
  Future Enhancements: None
"""

from __future__ import annotations

import metta_numpy  # noqa: F401  -- imported for the array row it registers
import pytest
from hypothesis import given

import metta.testing as pt
from metta import seam

numpy = pytest.importorskip("numpy")


def test_the_row_is_registered_against_the_array_point():
    """One row, named for the library, marked as the no-argument default."""
    row = seam.array.find("numpy")
    assert row is not None
    assert row.module == "numpy"
    assert row.default is True
    assert "pymetta[arrays]" in row.missing


@given(pt.library_scalars("numpy"))
def test_the_scalars_row_generates_numpy_scalars(scalar):
    """`library_scalars(<name>)` reads this row's own strategy."""
    assert isinstance(scalar, numpy.generic)


def test_the_column_door_builds_a_numpy_array(scratch_space):
    """`Column.__array__` reaches whichever row is the default."""
    scratch_space.add(scratch_space.parse("(n 1)"))
    scratch_space.add(scratch_space.parse("(n 2)"))
    rows = scratch_space.match(scratch_space.parse("(n $v)"))
    assert list(numpy.asarray(rows.v)) == [1, 2]
