"""Purpose: prove the sqlite3 row registers a MeTTa head as a SQL function.

Blackbox through `metta.tables.sql_function`, which is the door a program
takes; this package's row is what makes a sqlite3 connection one the seat
recognises.
Open Obligations:
  To Do: None
  Hacks: None
  Future Enhancements: None
"""

from __future__ import annotations

import sqlite3

import metta_sqlite  # noqa: F401  -- imported for the sql row it registers

from metta import seam, tables


def test_the_row_is_registered_against_the_sql_point():
    """One row, named for the engine, carrying claims and define."""
    row = seam.sql.find("sqlite3")
    assert row is not None
    assert not row.fallback


def test_an_undeclared_head_registers_by_its_arity(scratch_space):
    """sqlite3 wants the arity and no types, so no arrow is needed."""
    scratch_space.run("(= (dbl $x) (* 2 $x))")
    connection = sqlite3.connect(":memory:")
    try:
        assert tables.sql_function(connection, scratch_space.fn.dbl) == "dbl"
        assert connection.execute("select dbl(21)").fetchone() == (42,)
    finally:
        connection.close()


def test_a_foreign_connection_is_declined(scratch_space):
    """A connection this engine does not own leaves the next row to answer."""
    scratch_space.run("(= (dbl $x) (* 2 $x))")

    class SqliteProbeConnection:
        """Not a connection any registered engine claims."""

    assert seam.sql.claim(SqliteProbeConnection()) is None
