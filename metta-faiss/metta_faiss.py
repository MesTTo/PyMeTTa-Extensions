"""Purpose: faiss as a nearest-neighbour backend of the MeTTa Python seat.

One row against the seat's `index` point, so an embedding store searches
through faiss when this package is installed. The row is NOT a fallback, which
is what makes `backend="auto"` prefer it over the generic Array API path
whatever order the two distributions loaded in; entry-point order is not
something a package can arrange.

NumPy is a dependency here and is not an integration: faiss' own interface
takes contiguous float32 NumPy arrays for `add` and `search`, so staging one is
part of calling faiss at all, and faiss-cpu depends on NumPy itself.

Assumes:
  - a normalized matrix, which is the `index` point's contract: the backend
    searches inner products and the store normalizes before it gets here
Guarantees:
  - `available()` answers False rather than raising when faiss is absent, so
    an embedding store falls through to the next backend instead of failing
    [tested: tests/test_faiss.py::test_an_absent_faiss_declines_rather_than_raising;
    commit=WORKTREE]
  - faiss wins `backend="auto"` over the Array API fallback in either load
    order [tested: tests/test_faiss.py::test_faiss_wins_auto_over_the_array_api_fallback;
    commit=WORKTREE]
Open Obligations:
  To Do: None
  Hacks: None
  Future Enhancements: None
"""

from __future__ import annotations

from typing import Any, Final

from metta import seam

_MISSING: Final = (
    "the faiss embedding backend needs faiss-cpu; install pymetta[arrays]"
)
_NUMPY_MISSING: Final = (
    "the faiss embedding backend stages its matrix as a contiguous float32 "
    "NumPy array, which is faiss' own interface, and NumPy is not installed; "
    "install pymetta[arrays]"
)

require_module = seam.at("module").call()
optional_module = seam.at("optional-module").call()


def _available() -> bool:
    """Whether faiss can run here."""
    return optional_module("faiss") is not None


def _build(matrix: Any) -> Any:
    """An exact inner-product index over the normalized matrix."""
    faiss = require_module("faiss", _MISSING)
    numpy = require_module("numpy", _NUMPY_MISSING)
    staged = numpy.ascontiguousarray(numpy.asarray(matrix, dtype=numpy.float32))
    index = faiss.IndexFlatIP(staged.shape[1])
    index.add(staged)
    return index


def _search(index: Any, query: Any, count: int) -> list[tuple[int, float]]:
    """(row, score) pairs best first, from the built index."""
    numpy = require_module("numpy", _NUMPY_MISSING)
    probe = numpy.ascontiguousarray(numpy.asarray(query, dtype=numpy.float32).reshape(1, -1))
    scores, indexes = index.search(probe, count)
    return [
        (int(index), float(score))
        for score, index in zip(scores[0], indexes[0], strict=True)
    ]


def register() -> None:
    """This package's one row, against the seat's `index` point."""
    seam.index.register(
        "faiss",
        available=_available,
        build=_build,
        search=_search,
        missing=_MISSING,
    )


register()
