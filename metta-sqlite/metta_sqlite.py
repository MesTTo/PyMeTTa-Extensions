"""Purpose: sqlite3 as a SQL engine of the MeTTa Python seat, in one row.

A MeTTa head registered into a sqlite3 connection becomes a scalar SQL
function, so a query can call a definition the space holds. One row against
the seat's `sql` point; pymetta names no engine.

sqlite3 is the standard library, so this distribution depends on nothing but
pymetta. It is still a distribution and not a shipped row, because "in the
standard library" is not a reason for the core to know a SQL engine: the second
engine of the class has to reach the same door the first did, and a row that
lived in the core would be the branch a fork starts from.

Assumes:
  - sqlite3 wants an ARITY and no types, which is why an undeclared head
    registers here where DuckDB refuses one
Guarantees:
  - a head with no declared arrow still registers, taking its argument count
    from the signature [tested: tests/test_sqlite.py; commit=94057a0f073c0fab0a35c42beff2c324d8a0addd]
  - a connection of any other engine is declined, so the next row is consulted
    [tested: tests/test_sqlite.py::test_a_foreign_connection_is_declined;
    commit=94057a0f073c0fab0a35c42beff2c324d8a0addd]
Open Obligations:
  To Do: None
  Hacks: None
  Future Enhancements: None
"""

from __future__ import annotations

import sqlite3
from typing import Any

from metta import seam


def _claims(connection: Any) -> Any:
    """sqlite3's own connection type, which is in the standard library."""
    return connection if isinstance(connection, sqlite3.Connection) else None


def _define(connection: Any, name: str, call: Any, head: Any, signature: Any) -> None:
    """sqlite3 wants the arity and no types, so an undeclared head registers."""
    del head
    connection.create_function(name, seam.sql_arity(signature), call)


def register() -> None:
    """This package's one row, against the seat's `sql` point."""
    seam.sql.register("sqlite3", claims=_claims, define=_define)


register()
