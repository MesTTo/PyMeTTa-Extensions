"""Purpose: NumPy as the default array library of the MeTTa Python seat.

One row against the seat's `array` point: it says NumPy may be the no-argument
default where an installer is given no `default=`, what its absence should say,
and how to generate its own scalar values for property tests. Nothing else here
knows NumPy exists, and an array library that only wants to be USABLE needs no
row at all, because the array layer reaches any module the Array API standard
covers.

Installed beside pymetta, or through `pip install 'pymetta[arrays]'`, which
installs the generic array layer with it.

Assumes:
  - hypothesis is present when `scalars()` is called; it is the test extra and
    the row's other three fields do not need it
Guarantees:
  - the row makes NumPy the default with no default= given, and taking it away
    leaves the array doors refusing by name rather than reaching NumPy anyway
    [tested: tests/test_numpy.py; commit=94057a0f073c0fab0a35c42beff2c324d8a0addd]
  - the scalars strategy generates values MeTTa accepts as Number operands
    while they keep their NumPy identity [tested:
    tests/test_numpy.py::test_the_scalars_row_generates_numpy_scalars;
    commit=94057a0f073c0fab0a35c42beff2c324d8a0addd]
Open Obligations:
  To Do: None
  Hacks: None
  Future Enhancements: None
"""

from __future__ import annotations

from typing import Any, Final

from metta import seam

_MISSING: Final = (
    "the array layer needs NumPy for default arrays and embedding storage; "
    "install pymetta[arrays]"
)

require_module = seam.at("module").call()


def _scalars() -> Any:
    """NumPy integer and real scalars, as a Hypothesis strategy.

    These retain identity while MeTTa accepts them as Number operands and
    dispatches through Python operators.
    """
    st = require_module(
        "hypothesis.strategies",
        "generating this library's scalars requires hypothesis; install "
        "pymetta[arrays,test]",
    )
    numpy = require_module("numpy", _MISSING)
    return st.one_of(
        st.integers(-(2**31), 2**31 - 1).map(numpy.int32),
        st.integers(-(2**62), 2**62).map(numpy.int64),
        st.floats(allow_nan=False, allow_infinity=False, width=32).map(numpy.float32),
        st.floats(allow_nan=False, allow_infinity=False, width=64).map(numpy.float64),
    )


def register() -> None:
    """This package's one row, against the seat's `array` point."""
    seam.array.register(
        "numpy",
        module="numpy",
        default=True,
        missing=_MISSING,
        scalars=_scalars,
    )


register()
