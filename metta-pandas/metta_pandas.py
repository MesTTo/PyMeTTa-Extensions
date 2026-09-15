"""Purpose: pandas as a frame library of the MeTTa Python seat, in one row.

`rows.to(pandas)` and its declared sugar `rows.to_df()` build a DataFrame from
a set of answers, and `frame.metta` becomes pandas' own registered accessor so
a frame can put its rows into a space. The builder and accessor share a frame
row; the namespace and short methods are contracts on the door point.

Installed beside pymetta, or through `pip install 'pymetta[dataframes]'`, and
found either by import (the row registers from this module's body) or by the
`metta.extensions` entry point this distribution advertises, which the seat
loads at the first frame dispatch that has no answer without it. That is
Pygments' shape for a plugin lexer and Airflow's for a provider.

Assumes:
  - pandas is importable when a frame is actually BUILT; the row itself holds
    the module NAME and imports nothing, so `import metta._spaces.results` stays free
Guarantees:
  - frame ingestion declares pandas' native row iterator, `tables.add` calls it
    for a frame, and unrelated inputs stay unclaimed [tested:
    tests/test_pandas.py::test_frame_rows_use_the_declared_native_extractor;
    commit=179bcf460e69f3f7e05683027983a063df0b482e]
  - namespace and short conversion methods come from door contracts; the
    frame row owns the builder and library accessor [tested:
    tests/test_pandas.py::test_the_row_is_registered_against_the_frame_point,
    tests/test_pandas_doors.py; commit=b615b5a33b43252ef9826e5387da7c9bd7f6b543]
  - a frame built from a pandas 3 install is typed BY the projection through
    DataFrame.from_arrow, and falls back to the projected columns below that
    or without an Arrow builder [tested: tests/test_pandas.py; commit=b615b5a33b43252ef9826e5387da7c9bd7f6b543]
  - the accessor installs on a pandas already imported and imports nothing to
    find out [tested: tests/test_pandas.py::test_the_accessor_installs_for_an_imported_pandas;
    commit=b615b5a33b43252ef9826e5387da7c9bd7f6b543]
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
    "to_df() builds a pandas DataFrame and pandas is not installed; "
    "rows.table() is the plain dict any frame constructor takes"
)

require_module = seam.at("module").call()


def _install_accessor(pandas: Any, name: str, door: type) -> None:
    """`df.<name>` on a pandas frame: its registered accessor."""
    pandas.api.extensions.register_dataframe_accessor(name)(door)


def _rows(source: Any) -> Iterator[Any] | None:
    """Read an already imported pandas frame through its native row iterator."""
    pandas = sys.modules.get("pandas")
    if pandas is not None and isinstance(source, pandas.DataFrame):
        return source.itertuples(index=False)
    return None


def _build(source: Any, projection: Any, view: Any) -> Any:
    """A pandas DataFrame of these rows.

    `DataFrame.from_arrow` is pandas 3's reader for the PyCapsule stream, so
    the columns are TYPED by the projection rather than inferred from Python
    objects. Without pandas 3, or without a builder for the capsules, the same
    projected columns go through the frame constructor and answer the same
    values.
    """
    pandas = require_module("pandas", _MISSING)
    from_arrow = getattr(pandas.DataFrame, "from_arrow", None)
    if from_arrow is not None and view is not None:
        return from_arrow(source)
    if len(source) and not projection.names:
        return pandas.DataFrame([{} for _ in source])
    return pandas.DataFrame(projection.table())



@_doors.door(
    kind=_doors.Kind.query,
    answers=_doors.AnswersAs.value,
    effect=_doors.EffectClass.oracleIO,
    determinism=_doors.Determinism.det,
    state=_doors.State.any,
    tiers=(_doors.Tier.sync, _doors.Tier.context),
    provider=_doors.Provider('metta-pandas', 'tables'),
    evidence=('extensions/python/ext/metta-pandas/tests/test_pandas_doors.py::test_pandas_namespace_and_short_sugars_share_the_row',),
    sugar_of=_doors.Sugar('rows:to', (('library', 'pandas'),)),
)
def to_df(rows: Any) -> Any:
    """These rows as a pandas DataFrame; the declared point rows.to('pandas')."""
    return rows.to('pandas')




class _RowsSugar:
    @staticmethod
    @_doors.door(
        kind=_doors.Kind.query,
        answers=_doors.AnswersAs.value,
        effect=_doors.EffectClass.oracleIO,
        determinism=_doors.Determinism.det,
        state=_doors.State.any,
        tiers=(_doors.Tier.sync,),
        provider=_doors.Provider('metta-pandas', 'tables'),
        evidence=('extensions/python/ext/metta-pandas/tests/test_pandas_doors.py::test_pandas_namespace_and_short_sugars_share_the_row',),
        sugar_of=_doors.Sugar('rows:to', (('library', 'pandas'),)),
    )
    def to_df(rows: Rows) -> Any:
        """These rows as a pandas DataFrame; the declared point rows.to('pandas')."""
        return rows.to(library='pandas')

class _AnswersSugar:
    @staticmethod
    @_doors.door(
        kind=_doors.Kind.query,
        answers=_doors.AnswersAs.value,
        effect=_doors.EffectClass.oracleIO,
        determinism=_doors.Determinism.det,
        state=_doors.State.any,
        tiers=(_doors.Tier.sync,),
        provider=_doors.Provider('metta-pandas', 'tables'),
        evidence=('extensions/python/ext/metta-pandas/tests/test_pandas_doors.py::test_pandas_namespace_and_short_sugars_share_the_row',),
        sugar_of=_doors.Sugar('answers:to', (('library', 'pandas'),)),
    )
    def to_df(rows: Answers) -> Any:
        """These rows as a pandas DataFrame; the declared point rows.to('pandas')."""
        return rows.to(library='pandas')


def register() -> None:
    """Publish the frame provider and the contracts that reach it."""
    seam.door.register('metta-pandas', doors=_doors.declarations(__name__))
    seam.frame.register(
        "pandas",
        module="pandas",
        accessor=_install_accessor,
        build=_build,
        rows=_rows,
    )


register()
