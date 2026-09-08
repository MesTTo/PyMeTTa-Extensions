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
    the module NAME and imports nothing, so `import metta.results` stays free
Guarantees:
  - namespace and short conversion methods come from door contracts; the
    frame row owns the builder and library accessor [tested:
    tests/test_pandas.py::test_the_row_is_registered_against_the_frame_point,
    tests/test_pandas_doors.py; commit=WORKTREE]
  - a frame built from a pandas 3 install is typed BY the projection through
    DataFrame.from_arrow, and falls back to the projected columns below that
    or without an Arrow builder [tested: tests/test_pandas.py; commit=WORKTREE]
  - the accessor installs on a pandas already imported and imports nothing to
    find out [tested: tests/test_pandas.py::test_the_accessor_installs_for_an_imported_pandas;
    commit=WORKTREE]
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
    "to_df() builds a pandas DataFrame and pandas is not installed; "
    "rows.table() is the plain dict any frame constructor takes"
)

require_module = seam.at("module").call()


def _install_accessor(pandas: Any, name: str, door: type) -> None:
    """`df.<name>` on a pandas frame: its registered accessor."""
    pandas.api.extensions.register_dataframe_accessor(name)(door)


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



def to_df(rows: Any) -> Any:
    """These rows as a pandas DataFrame; the declared point rows.to('pandas')."""
    return rows.to('pandas')


# closed-set: decides; policy=this frame library declares its accessor and short receiver sugars; reads=Rows.to and Answers.to
DOORS: tuple[Door, ...] = (
    Door(
        owner=Owner.namespace, name='to-df', kind=Kind.query,
        signatures=(Signature("rows: Any", returns="Any"),), answers=AnswersAs.value,
        effect=EffectClass.oracleIO, determinism=Determinism.det, state=State.any,
        tiers=(Tier.sync, Tier.context), body=Body('metta_pandas', 'to_df', Receiver.none),
        provider=Provider('metta-pandas', "tables"),
        docs="These rows as a pandas DataFrame; the declared point rows.to('pandas').",
        evidence=('extensions/python/ext/metta-pandas/tests/test_pandas_doors.py::test_pandas_namespace_and_short_sugars_share_the_row',),
        sugar_of=Sugar("rows:to", (("library", 'pandas'),)),
    ),
    Door(
        owner=Owner.rows, name='to-df', kind=Kind.query,
        signatures=(Signature("self", returns="Any"),), answers=AnswersAs.value,
        effect=EffectClass.oracleIO, determinism=Determinism.det, state=State.any,
        tiers=(Tier.sync,), body=None,
        provider=Provider('metta-pandas', "tables"),
        docs="These rows as a pandas DataFrame; the declared point rows.to('pandas').",
        evidence=('extensions/python/ext/metta-pandas/tests/test_pandas_doors.py::test_pandas_namespace_and_short_sugars_share_the_row',),
        sugar_of=Sugar("rows:to", (("library", 'pandas'),)),
    ),
    Door(
        owner=Owner.answers, name='to-df', kind=Kind.query,
        signatures=(Signature("self", returns="Any"),), answers=AnswersAs.value,
        effect=EffectClass.oracleIO, determinism=Determinism.det, state=State.any,
        tiers=(Tier.sync,), body=None,
        provider=Provider('metta-pandas', "tables"),
        docs="These rows as a pandas DataFrame; the declared point rows.to('pandas').",
        evidence=('extensions/python/ext/metta-pandas/tests/test_pandas_doors.py::test_pandas_namespace_and_short_sugars_share_the_row',),
        sugar_of=Sugar("answers:to", (("library", 'pandas'),)),
    ),
)


def register() -> None:
    """Publish the frame provider and the contracts that reach it."""
    seam.door.register('metta-pandas', doors=DOORS)
    seam.frame.register(
        "pandas",
        module="pandas",
        accessor=_install_accessor,
        build=_build,
    )


register()
