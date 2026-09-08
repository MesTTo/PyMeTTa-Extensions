"""Purpose: polars as a frame library of the MeTTa Python seat, in one row.

`rows.to(polars)` and its declared sugar `rows.to_pl()` build a DataFrame from
a set of answers, and `frame.metta` becomes polars' own registered namespace so
a frame can put its rows into a space. The builder and accessor share a frame
row; the namespace and short methods are contracts on the door point.

Installed beside pymetta, or through `pip install 'pymetta[dataframes]'`, and
found either by import or by the `metta.extensions` entry point this
distribution advertises.

Assumes:
  - polars is importable when a frame is actually BUILT; the row holds the
    module NAME and imports nothing
Guarantees:
  - namespace and short conversion methods come from door contracts; the
    frame row owns the builder and library accessor [tested:
    tests/test_polars.py::test_the_row_is_registered_against_the_frame_point,
    tests/test_polars_doors.py; commit=WORKTREE]
  - a frame is built through the Arrow view where one exists, which is what
    reaches polars' capsule path: its constructor tests for a sequence before
    it looks for the capsule, and the view is the same data with nothing else
    on it [tested: tests/test_polars.py; commit=94057a0f073c0fab0a35c42beff2c324d8a0addd]
Open Obligations:
  To Do: None
  Hacks: None
  Future Enhancements: None
"""

from __future__ import annotations

from typing import Any, Final

from metta import seam
from metta.doors import (
    AnswersAs,
    Body,
    Door,
    Kind,
    Owner,
    Provider,
    Receiver,
    Signature,
    State,
    Sugar,
    Tier,
)
from metta.vocabularies import Determinism, EffectClass

_MISSING: Final = (
    "to_pl() builds a polars DataFrame and polars is not installed; "
    "rows.table() is the plain dict any frame constructor takes"
)

require_module = seam.at("module").call()


def _install_accessor(polars: Any, name: str, door: type) -> None:
    """`df.<name>` on a polars frame: its registered namespace."""
    polars.api.register_dataframe_namespace(name)(door)


def _build(source: Any, projection: Any, view: Any) -> Any:
    """A polars DataFrame of these rows, through the Arrow view where there is one.

    polars' constructor tests for a sequence before it looks for the capsule,
    so the view, which is the same data with nothing else on it, is what
    reaches the Arrow path.
    """
    polars = require_module("polars", _MISSING)
    if view is not None:
        return polars.DataFrame(view)
    if len(source) and not projection.names:
        return polars.DataFrame([{} for _ in source])
    return polars.DataFrame(projection.table())



def to_pl(rows: Any) -> Any:
    """These rows as a polars DataFrame; the declared point rows.to('polars')."""
    return rows.to('polars')


# closed-set: decides; policy=this frame library declares its accessor and short receiver sugars; reads=Rows.to and Answers.to
DOORS: tuple[Door, ...] = (
    Door(
        owner=Owner.namespace, name='to-pl', kind=Kind.query,
        signatures=(Signature("rows: Any", returns="Any"),), answers=AnswersAs.value,
        effect=EffectClass.oracleIO, determinism=Determinism.det, state=State.any,
        tiers=(Tier.sync, Tier.context), body=Body('metta_polars', 'to_pl', Receiver.none),
        provider=Provider('metta-polars', "tables"),
        docs="These rows as a polars DataFrame; the declared point rows.to('polars').",
        evidence=('extensions/python/ext/metta-polars/tests/test_polars_doors.py::test_polars_namespace_and_short_sugars_share_the_row',),
        sugar_of=Sugar("rows:to", (("library", 'polars'),)),
    ),
    Door(
        owner=Owner.rows, name='to-pl', kind=Kind.query,
        signatures=(Signature("self", returns="Any"),), answers=AnswersAs.value,
        effect=EffectClass.oracleIO, determinism=Determinism.det, state=State.any,
        tiers=(Tier.sync,), body=None,
        provider=Provider('metta-polars', "tables"),
        docs="These rows as a polars DataFrame; the declared point rows.to('polars').",
        evidence=('extensions/python/ext/metta-polars/tests/test_polars_doors.py::test_polars_namespace_and_short_sugars_share_the_row',),
        sugar_of=Sugar("rows:to", (("library", 'polars'),)),
    ),
    Door(
        owner=Owner.answers, name='to-pl', kind=Kind.query,
        signatures=(Signature("self", returns="Any"),), answers=AnswersAs.value,
        effect=EffectClass.oracleIO, determinism=Determinism.det, state=State.any,
        tiers=(Tier.sync,), body=None,
        provider=Provider('metta-polars', "tables"),
        docs="These rows as a polars DataFrame; the declared point rows.to('polars').",
        evidence=('extensions/python/ext/metta-polars/tests/test_polars_doors.py::test_polars_namespace_and_short_sugars_share_the_row',),
        sugar_of=Sugar("answers:to", (("library", 'polars'),)),
    ),
)


def register() -> None:
    """Publish the frame provider and the contracts that reach it."""
    seam.door.register('metta-polars', doors=DOORS)
    seam.frame.register(
        "polars",
        module="polars",
        accessor=_install_accessor,
        build=_build,
    )


register()
