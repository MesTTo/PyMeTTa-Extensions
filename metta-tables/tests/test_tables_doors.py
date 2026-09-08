"""Purpose: verify each registered tables namespace door at its public boundary.

Guarantees: ingestion, declarations, accessors, and SQL calls reach their bodies
  [tested: test_table_namespace_preserves_ingestion_and_conversions; commit=WORKTREE].
Owns resources: the test closes its database connection and engine context.
"""

import sqlite3
from contextlib import closing

import metta_sqlite
import metta_tables
import pytest

from metta import MeTTa, S, V


def test_table_namespace_preserves_ingestion_and_conversions():
    """Table namespace preserves ingestion and conversions."""
    metta_tables.register()
    metta_sqlite.register()
    with MeTTa() as context, closing(sqlite3.connect(":memory:")) as connection:
        assert context.tables.add(S.person, [("Ada", 2), ("Bob", 3)]) == 2
        assert len(context.self[S.person(V.name, V.n)]) == 2
        declaration = context.tables.declare("people", "(bridge (person $name $n) (row people (name $name) (n $n)))")
        assert declaration in context.space("&metta")
        assert isinstance(context.tables.accessors(), tuple)
        context.run("(: twice (-> Number Number)) (= (twice $x) (* 2 $x))")
        assert context.self.tables.sql_function(connection, context.fn.twice, "twice") == "twice"
        assert connection.execute("select twice(21)").fetchall() == [(42,)]


def test_tables_add_refuses_an_unsupported_source():
    """Tables add refuses an unsupported source."""
    metta_tables.register()
    with MeTTa() as context:
        with pytest.raises(TypeError, match="offers none"):
            context.tables.add(S.person, object())


def test_tables_sql_function_refuses_noncallable_heads():
    """Tables sql function refuses noncallable heads."""
    metta_tables.register()
    with MeTTa() as context:
        with pytest.raises(TypeError, match="not callable"):
            context.tables.sql_function(object(), None)
