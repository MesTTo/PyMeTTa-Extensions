"""Purpose: register the tables accessor using deferred implementation references.

Guarantees: registration imports no implementation and accessor calls preserve
  its behavior [tested: test_table_namespace_preserves_ingestion_and_conversions; commit=b615b5a33b43252ef9826e5387da7c9bd7f6b543].
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

import metta.doors as _doors
from metta import seam

if TYPE_CHECKING:
    from metta import Atom, SpaceLike



@_doors.door(
    kind=_doors.Kind.write,
    answers=_doors.AnswersAs.integer,
    effect=_doors.EffectClass.oracleIO,
    determinism=_doors.Determinism.det,
    tiers=(_doors.Tier.sync, _doors.Tier.context),
    refuses=(_doors.Refusal(_doors.RefusalKind.type, 'extensions/python/ext/metta-tables/tests/test_tables_doors.py::test_tables_add_refuses_an_unsupported_source'),),
    provider=_doors.Provider('metta-tables', 'tables'),
    evidence=('extensions/python/ext/metta-tables/tests/test_tables_doors.py::test_table_namespace_preserves_ingestion_and_conversions',),
)
def add(space: SpaceLike, head: Any, data: Any) -> int:
    """Add a tabular source to a space as ``(head column...)`` facts.

    space may be a context or a space.

    The source may offer rows its own way (``iter_rows()`` for polars,
    ``itertuples()`` for pandas, a mapping of columns, any iterable of rows)
    or speak the Arrow PyCapsule Interface, which is how a DuckDB relation, a
    pyarrow Table, a Parquet reader or an Ibis expression hands over rows
    without a row-at-a-time Python door. A source with both keeps its own:
    the two produce identical atoms, and the row door is the faster of them
    [measured 2026-09-06: 10,000 rows, polars 14.07 ms through iter_rows
    against 14.36 ms through the stream, pandas 20.39 ms against 25.11 ms].

    An Arrow source is written one record batch at a time, so a reader larger
    than memory loads, and the writes are one transaction each; wrap the call
    in ``m.transaction(...)`` to make the whole load one.
    """
    return import_module('metta.tables').add(space, head, data)

@_doors.door(
    kind=_doors.Kind.write,
    answers=_doors.AnswersAs.atom,
    effect=_doors.EffectClass.oracleIO,
    determinism=_doors.Determinism.det,
    tiers=(_doors.Tier.sync, _doors.Tier.context),
    provider=_doors.Provider('metta-tables', 'tables'),
    evidence=('extensions/python/ext/metta-tables/tests/test_tables_doors.py::test_table_namespace_preserves_ingestion_and_conversions',),
)
def declare(m: SpaceLike, name: str, declaration: Atom | str) -> Atom:
    """Write one ctx-scoped bridge declaration into &metta, where explain
    and any program can read the schema, and from_context will.

    m may be a context or a space.
    """  # noqa: D205 -- preserve the declared documentation
    return import_module('metta.tables').declare(m, name, declaration)

@_doors.door(
    kind=_doors.Kind.provider,
    answers=_doors.AnswersAs.tuple,
    effect=_doors.EffectClass.oracleIO,
    determinism=_doors.Determinism.det,
    tiers=(_doors.Tier.sync, _doors.Tier.context),
    provider=_doors.Provider('metta-tables', 'tables'),
    evidence=('extensions/python/ext/metta-tables/tests/test_tables_doors.py::test_table_namespace_preserves_ingestion_and_conversions',),
)
def accessors() -> tuple[str, ...]:
    """Install the metta accessor for every registered frame library already imported.

    Answers the libraries that now carry it, so a program can ask.

    Registration never imports a frame library. `import metta.tables` costs
    15 ms and `import pandas` costs 531 ms [measured 2026-09-06,
    time.perf_counter around each import in a fresh interpreter], so a module
    that registered by importing would charge every tables user for a library
    the program may never touch. It installs for whichever registered module
    is in `sys.modules`, every door in this module calls it first, and a
    program that imports a frame library afterwards and touches nothing else
    here calls this by name. Idempotent, because a library warns when an
    accessor name is replaced.

    Which libraries these are is the `frame` point's rows, not a list here:
    each row says which module it is and how that library spells an accessor,
    so a third one installs `df.metta` by registering
    (`metta.seam.frame.register(...)`, or the `metta.extensions` entry point).
    """
    return import_module('metta.tables').accessors()

@_doors.door(
    kind=_doors.Kind.provider,
    answers=_doors.AnswersAs.text,
    effect=_doors.EffectClass.oracleIO,
    determinism=_doors.Determinism.det,
    tiers=(_doors.Tier.sync, _doors.Tier.context),
    refuses=(_doors.Refusal(_doors.RefusalKind.type, 'extensions/python/ext/metta-tables/tests/test_tables_doors.py::test_tables_sql_function_refuses_noncallable_heads'),),
    provider=_doors.Provider('metta-tables', 'tables'),
    evidence=('extensions/python/ext/metta-tables/tests/test_tables_doors.py::test_table_namespace_preserves_ingestion_and_conversions',),
)
def sql_function(connection: Any, head: Any, name: str | None=None) -> str:
    """Register a MeTTa head as a scalar SQL function, and answer its SQL name.

        m.run("(: dbl (-> Number Number))  (= (dbl $x) (* 2 $x))")
        tables.sql_function(connection, m.fn.dbl)
        connection.sql("select dbl(age) from people")

    The head is the callable from a space's `fn` namespace, which already
    carries its own name, its arity and its arrow, so nothing about the
    function is restated here; `name=` is the escape for a SQL identifier the
    head's own name cannot be.

    WHICH engines are known is the `sql` point's rows, and the first row that
    claims the connection declares the function its own way: sqlite3 wants the
    arity and no types, DuckDB wants the types and reads them from the head's
    DECLARED arrow, refusing by name when there is none (an arrow
    `inspect.signature` merely infers is a proposal, not a promise). A third
    engine registers rather than being added here. A row that produces no
    answer is SQL NULL and one that produces several refuses, because a scalar
    function has one result per row; a SQL NULL argument reaches the head as
    `Grounded(None)` and MeTTa decides what it means.
    """
    return import_module('metta.tables').sql_function(connection, head, name)


def register() -> None:
    """Publish this package's complete accessor declaration atomically."""
    seam.door.register('metta-tables', doors=_doors.declarations(__name__))


register()
