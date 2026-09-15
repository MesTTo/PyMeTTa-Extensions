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
  - frame ingestion declares polars' native row iterator and leaves unrelated
    inputs unclaimed [tested: tests/test_polars.py::test_frame_rows_use_the_declared_native_extractor;
    commit=01b2a9b3dfb721804cd8378610566e1985502289]
  - namespace and short conversion methods come from door contracts; the
    frame row owns the builder and library accessor [tested:
    tests/test_polars.py::test_the_row_is_registered_against_the_frame_point,
    tests/test_polars_doors.py; commit=b615b5a33b43252ef9826e5387da7c9bd7f6b543]
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

import sys
from collections.abc import Iterator
from typing import TYPE_CHECKING, Any, Final

import metta.doors as _doors
from metta import seam

if TYPE_CHECKING:
    from metta import Answers, Rows

_MISSING: Final = (
    "to_pl() builds a polars DataFrame and polars is not installed; "
    "rows.table() is the plain dict any frame constructor takes"
)

require_module = seam.at("module").call()


def _install_accessor(polars: Any, name: str, door: type) -> None:
    """`df.<name>` on a polars frame: its registered namespace."""
    polars.api.register_dataframe_namespace(name)(door)


def _rows(source: Any) -> Iterator[Any] | None:
    """Read an already imported polars frame through its native row iterator."""
    polars = sys.modules.get("polars")
    if polars is not None and isinstance(source, polars.DataFrame):
        return source.iter_rows()
    return None


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



@_doors.door(
    kind=_doors.Kind.query,
    answers=_doors.AnswersAs.value,
    effect=_doors.EffectClass.oracleIO,
    determinism=_doors.Determinism.det,
    state=_doors.State.any,
    tiers=(_doors.Tier.sync, _doors.Tier.context),
    provider=_doors.Provider('metta-polars', 'tables'),
    evidence=('extensions/python/ext/metta-polars/tests/test_polars_doors.py::test_polars_namespace_and_short_sugars_share_the_row',),
    sugar_of=_doors.Sugar('rows:to', (('library', 'polars'),)),
)
def to_pl(rows: Any) -> Any:
    """These rows as a polars DataFrame; the declared point rows.to('polars')."""
    return rows.to('polars')




class _RowsSugar:
    @staticmethod
    @_doors.door(
        kind=_doors.Kind.query,
        answers=_doors.AnswersAs.value,
        effect=_doors.EffectClass.oracleIO,
        determinism=_doors.Determinism.det,
        state=_doors.State.any,
        tiers=(_doors.Tier.sync,),
        provider=_doors.Provider('metta-polars', 'tables'),
        evidence=('extensions/python/ext/metta-polars/tests/test_polars_doors.py::test_polars_namespace_and_short_sugars_share_the_row',),
        sugar_of=_doors.Sugar('rows:to', (('library', 'polars'),)),
    )
    def to_pl(rows: Rows) -> Any:
        """These rows as a polars DataFrame; the declared point rows.to('polars')."""
        return rows.to(library='polars')

class _AnswersSugar:
    @staticmethod
    @_doors.door(
        kind=_doors.Kind.query,
        answers=_doors.AnswersAs.value,
        effect=_doors.EffectClass.oracleIO,
        determinism=_doors.Determinism.det,
        state=_doors.State.any,
        tiers=(_doors.Tier.sync,),
        provider=_doors.Provider('metta-polars', 'tables'),
        evidence=('extensions/python/ext/metta-polars/tests/test_polars_doors.py::test_polars_namespace_and_short_sugars_share_the_row',),
        sugar_of=_doors.Sugar('answers:to', (('library', 'polars'),)),
    )
    def to_pl(rows: Answers) -> Any:
        """These rows as a polars DataFrame; the declared point rows.to('polars')."""
        return rows.to(library='polars')


def register() -> None:
    """Publish the frame provider and the contracts that reach it."""
    seam.door.register('metta-polars', doors=_doors.declarations(__name__))
    seam.frame.register(
        "polars",
        module="polars",
        accessor=_install_accessor,
        build=_build,
        rows=_rows,
    )


register()
