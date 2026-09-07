"""Purpose: prove the DuckDB row registers a typed MeTTa head, and refuses one.

DuckDB wants the parameter and return types, and reads them from the head's
DECLARED arrow; a head with none is refused naming the arrow it needs, which is
the difference from the sqlite3 row and the reason both exist.
Open Obligations:
  To Do: None
  Hacks: None
  Future Enhancements: None
"""

from __future__ import annotations

import metta_duckdb  # noqa: F401  -- imported for the sql row it registers
import pytest

from metta import seam, tables

duckdb = pytest.importorskip("duckdb")


def test_the_row_is_registered_against_the_sql_point():
    """One row, named for the engine, carrying its own undeclared sentence."""
    row = seam.sql.find("duckdb")
    assert row is not None
    assert "declare the head's arrow" in row.undeclared("f")


def test_a_declared_head_becomes_a_duckdb_scalar_function(scratch_space):
    """The arrow the program declared is the arrow the SQL function carries."""
    scratch_space.run("(: dbl (-> Number Number))\n(= (dbl $x) (* 2 $x))")
    connection = duckdb.connect(":memory:")
    try:
        assert tables.sql_function(connection, scratch_space.fn.dbl) == "dbl"
        assert connection.sql("select dbl(21) as d").fetchone() == (42.0,)
    finally:
        connection.close()


def test_an_undeclared_head_is_refused_with_its_arrow(scratch_space):
    """A proposal is not a promise: the inferred arrow does not register."""
    scratch_space.run("(= (trip $x) (* 3 $x))")
    connection = duckdb.connect(":memory:")
    try:
        with pytest.raises(TypeError, match=r"\(: trip \(-> Number Number\)\)"):
            tables.sql_function(connection, scratch_space.fn.trip)
    finally:
        connection.close()
