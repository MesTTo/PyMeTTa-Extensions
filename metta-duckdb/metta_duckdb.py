"""Purpose: DuckDB as a SQL engine of the MeTTa Python seat, in one row.

A MeTTa head registered into a DuckDB connection becomes a scalar SQL
function, typed from the head's DECLARED arrow because DuckDB wants types. One
row against the seat's `sql` point; pymetta names no engine.

Assumes:
  - a connection can only be DuckDB's if DuckDB is imported, so recognising one
    reads sys.modules and imports nothing
Guarantees:
  - a head with no declared arrow is refused with the arrow it needs, rather
    than registered with types inferred from what the stored atoms happen to
    justify [tested: tests/test_duckdb.py::test_an_undeclared_head_is_refused_with_its_arrow;
    commit=WORKTREE]
  - a head that answers nothing gives SQL NULL rather than an error, which is
    what null_handling="special" buys [tested: tests/test_duckdb.py;
    commit=WORKTREE]
Open Obligations:
  To Do: None
  Hacks: None
  Future Enhancements: None
"""

from __future__ import annotations

import sys
from typing import Any

from metta import seam


def _claims(connection: Any) -> Any:
    """A DuckDB connection, recognised without importing DuckDB.

    A connection can only be DuckDB's if DuckDB is imported, so reading
    sys.modules first keeps this free for every program that has no DuckDB.
    """
    duckdb = sys.modules.get("duckdb")
    if duckdb is None:
        return None
    return connection if isinstance(connection, duckdb.DuckDBPyConnection) else None


def _undeclared(name: str) -> str:
    """The refusal DuckDB gets for a head with no declared arrow."""
    return (
        f"DuckDB needs the types of {name} and cannot infer them; declare "
        f"the head's arrow, `(: {name} (-> Number Number))`, and register "
        f"again. An engine that needs only the arity takes it undeclared"
    )


def _define(connection: Any, name: str, call: Any, head: Any, signature: Any) -> None:
    """DuckDB wants the types, and reads them from the head's DECLARED arrow.

    `inspect.signature` shows the arrow the stored atoms JUSTIFY when nothing
    is declared (`Space.infer_types`'s proposal), which is the right thing to
    show a reader and the wrong thing to build SQL types from: a proposal is
    not a promise.
    """
    parameters, returns = seam.sql_types(head, name, signature, _undeclared)
    # SPECIAL, so a head may answer nothing and get SQL NULL: under DuckDB's
    # DEFAULT a returned NULL is an error, and NULL arguments never reach the
    # function at all.
    connection.create_function(name, call, parameters, returns, null_handling="special")


def register() -> None:
    """This package's one row, against the seat's `sql` point."""
    seam.sql.register("duckdb", claims=_claims, define=_define, undeclared=_undeclared)


register()
