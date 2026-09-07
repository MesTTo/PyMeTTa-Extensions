"""Purpose: prove the faiss row searches, and wins `backend="auto"`.

The second assertion is the load-bearing one. The Array API backend the array
layer registers is always available, so with both rows present the ORDER
decides `auto`, and order across distributions is not something a package can
arrange: the fallback flag is what makes the specific row win either way.
Open Obligations:
  To Do: None
  Hacks: None
  Future Enhancements: None
"""

from __future__ import annotations

import metta_arrays
import metta_faiss
import metta_numpy  # noqa: F401  -- the default array library the store needs
import pytest

from metta import S, seam

pytest.importorskip("faiss")


def test_the_row_is_registered_against_the_index_point():
    """One row, named for the library, and NOT a fallback."""
    row = seam.index.find("faiss")
    assert row is not None
    assert row.available() is True
    assert not row.fallback


def test_faiss_wins_auto_over_the_array_api_fallback():
    """Every non-fallback row is consulted before every fallback one."""
    names = [row.name for row in seam.index.rows()]
    assert names.index("faiss") < names.index("argsort")
    assert seam.index.find("argsort").fallback is True


def test_an_embedding_store_searches_through_faiss(scratch_space):
    """The store takes the backend by name and answers what faiss ranked."""
    store = metta_arrays.EmbeddingStore(scratch_space, name="faiss_emb", backend="faiss")
    store.add(S.dog, [1.0, 0.0])
    store.add(S.cat, [0.0, 1.0])
    assert [str(key) for key, _score in store.ranked([1.0, 0.0], 1)] == ["dog"]


def test_an_absent_faiss_declines_rather_than_raising(monkeypatch):
    """available() answers False, so the store falls through to the next row."""
    monkeypatch.setattr(metta_faiss, "optional_module", lambda _name: None)
    assert seam.index.find("faiss").available() is False
