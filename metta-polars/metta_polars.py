"""Purpose: polars as a frame library of the MeTTa Python seat, in one row.

`rows.to(polars)` and its declared sugar `rows.to_pl()` build a DataFrame from
a set of answers, and `frame.metta` becomes polars' own registered namespace so
a frame can put its rows into a space. Both are one row against the seat's
`frame` point; pymetta names no library.

Installed beside pymetta, or through `pip install 'pymetta[dataframes]'`, and
found either by import or by the `metta.extensions` entry point this
distribution advertises.

Assumes:
  - polars is importable when a frame is actually BUILT; the row holds the
    module NAME and imports nothing
Guarantees:
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


def register() -> None:
    """This package's one row, against the seat's `frame` point."""
    seam.frame.register(
        "polars",
        module="polars",
        accessor=_install_accessor,
        build=_build,
        sugar="to_pl",
    )


register()
