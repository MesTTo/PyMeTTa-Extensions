"""Purpose: register the tables accessor using deferred implementation references.

Guarantees: registration imports no implementation and accessor calls preserve
  its behavior [tested: test_table_namespace_preserves_ingestion_and_conversions; commit=b615b5a33b43252ef9826e5387da7c9bd7f6b543].
"""

from __future__ import annotations

from metta import seam
from metta.doors import (
    AnswersAs,
    Body,
    Door,
    Kind,
    Owner,
    Provider,
    Receiver,
    Refusal,
    Signature,
    Tier,
)
from metta.vocabularies import Determinism, EffectClass, RefusalKind

# closed-set: decides; policy=this package owns these accessor contracts; reads=the named implementation signatures checked by tools/doorgen.py
DOORS: tuple[Door, ...] = (
    Door(
        owner=Owner.namespace, name='add', kind=Kind.write,
        signatures=(Signature('space: SpaceLike, head: Any, data: Any', returns='int'),), answers=AnswersAs.integer,
        effect=EffectClass.oracleIO, determinism=Determinism.det,
        tiers=(Tier.sync, Tier.context), body=Body('metta.tables', 'add', Receiver.space),
        refuses=(Refusal(RefusalKind.type, 'extensions/python/ext/metta-tables/tests/test_tables_doors.py::test_tables_add_refuses_an_unsupported_source'),),
        provider=Provider('metta-tables', 'tables'),
        docs='Add a tabular source to a space as ``(head column...)`` facts.\n\nspace may be a context or a space.\n\nThe source may offer rows its own way (``iter_rows()`` for polars,\n``itertuples()`` for pandas, a mapping of columns, any iterable of rows)\nor speak the Arrow PyCapsule Interface, which is how a DuckDB relation, a\npyarrow Table, a Parquet reader or an Ibis expression hands over rows\nwithout a row-at-a-time Python door. A source with both keeps its own:\nthe two produce identical atoms, and the row door is the faster of them\n[measured 2026-09-06: 10,000 rows, polars 14.07 ms through iter_rows\nagainst 14.36 ms through the stream, pandas 20.39 ms against 25.11 ms].\n\nAn Arrow source is written one record batch at a time, so a reader larger\nthan memory loads, and the writes are one transaction each; wrap the call\nin ``m.transaction(...)`` to make the whole load one.',
        evidence=('extensions/python/ext/metta-tables/tests/test_tables_doors.py::test_table_namespace_preserves_ingestion_and_conversions',),
    ),
    Door(
        owner=Owner.namespace, name='declare', kind=Kind.write,
        signatures=(Signature('m: SpaceLike, name: str, declaration: Atom | str', returns='Atom'),), answers=AnswersAs.atom,
        effect=EffectClass.oracleIO, determinism=Determinism.det,
        tiers=(Tier.sync, Tier.context), body=Body('metta.tables', 'declare', Receiver.space),
        provider=Provider('metta-tables', 'tables'),
        docs='Write one ctx-scoped bridge declaration into &metta, where explain\nand any program can read the schema, and from_context will.\n\nm may be a context or a space.',
        evidence=('extensions/python/ext/metta-tables/tests/test_tables_doors.py::test_table_namespace_preserves_ingestion_and_conversions',),
    ),
    Door(
        owner=Owner.namespace, name='accessors', kind=Kind.provider,
        signatures=(Signature('', returns='tuple[str, ...]'),), answers=AnswersAs.tuple,
        effect=EffectClass.oracleIO, determinism=Determinism.det,
        tiers=(Tier.sync, Tier.context), body=Body('metta.tables', 'accessors', Receiver.none),
        provider=Provider('metta-tables', 'tables'),
        docs="Install the metta accessor for every registered frame library already imported.\n\nAnswers the libraries that now carry it, so a program can ask.\n\nRegistration never imports a frame library. `import metta.tables` costs\n15 ms and `import pandas` costs 531 ms [measured 2026-09-06,\ntime.perf_counter around each import in a fresh interpreter], so a module\nthat registered by importing would charge every tables user for a library\nthe program may never touch. It installs for whichever registered module\nis in `sys.modules`, every door in this module calls it first, and a\nprogram that imports a frame library afterwards and touches nothing else\nhere calls this by name. Idempotent, because a library warns when an\naccessor name is replaced.\n\nWhich libraries these are is the `frame` point's rows, not a list here:\neach row says which module it is and how that library spells an accessor,\nso a third one installs `df.metta` by registering\n(`metta.seam.frame.register(...)`, or the `metta.extensions` entry point).",
        evidence=('extensions/python/ext/metta-tables/tests/test_tables_doors.py::test_table_namespace_preserves_ingestion_and_conversions',),
    ),
    Door(
        owner=Owner.namespace, name='sql-function', kind=Kind.provider,
        signatures=(Signature('connection: Any, head: Any, name: str | None=None', returns='str'),), answers=AnswersAs.text,
        effect=EffectClass.oracleIO, determinism=Determinism.det,
        tiers=(Tier.sync, Tier.context), body=Body('metta.tables', 'sql_function', Receiver.none),
        refuses=(Refusal(RefusalKind.type, 'extensions/python/ext/metta-tables/tests/test_tables_doors.py::test_tables_sql_function_refuses_noncallable_heads'),),
        provider=Provider('metta-tables', 'tables'),
        docs='Register a MeTTa head as a scalar SQL function, and answer its SQL name.\n\n    m.run("(: dbl (-> Number Number))  (= (dbl $x) (* 2 $x))")\n    tables.sql_function(connection, m.fn.dbl)\n    connection.sql("select dbl(age) from people")\n\nThe head is the callable from a space\'s `fn` namespace, which already\ncarries its own name, its arity and its arrow, so nothing about the\nfunction is restated here; `name=` is the escape for a SQL identifier the\nhead\'s own name cannot be.\n\nWHICH engines are known is the `sql` point\'s rows, and the first row that\nclaims the connection declares the function its own way: sqlite3 wants the\narity and no types, DuckDB wants the types and reads them from the head\'s\nDECLARED arrow, refusing by name when there is none (an arrow\n`inspect.signature` merely infers is a proposal, not a promise). A third\nengine registers rather than being added here. A row that produces no\nanswer is SQL NULL and one that produces several refuses, because a scalar\nfunction has one result per row; a SQL NULL argument reaches the head as\n`Grounded(None)` and MeTTa decides what it means.',
        evidence=('extensions/python/ext/metta-tables/tests/test_tables_doors.py::test_table_namespace_preserves_ingestion_and_conversions',),
    ),
)


def register() -> None:
    """Publish this package's complete accessor declaration atomically."""
    seam.door.register('metta-tables', doors=DOORS)


register()
