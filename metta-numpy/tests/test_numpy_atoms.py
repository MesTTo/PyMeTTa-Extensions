"""Purpose: verify optional NumPy scalars keep identity and remain numeric.
Open Obligations:
  To Do: None
  Hacks: None
  Future Enhancements: None.
"""  # noqa: D205  -- the scenario narrative is one continuous invariant, not summary-and-body prose

import metta_numpy  # noqa: F401  -- imported for the array row it registers
import pytest

import metta.testing as pt
from metta import Grounded, convert

pytest.importorskip("numpy")
hypothesis = pytest.importorskip("hypothesis")
given = hypothesis.given


@given(pt.library_scalars("numpy"))
def test_numpy_scalar_strategy_round_trips_through_the_engine(metta, scalar):  # noqa: D103  -- pytest discovers or injects this callable; its descriptive name states the contract
    atom = Grounded(scalar)
    assert atom.value is scalar
    assert atom.to_wire()[0] == "o"
    row = metta.runtime.once(
        "metta_py_decode_shared(W, _T, _), metta_py_encode(_T, W2)",
        W=atom.to_wire(),
    )
    restored = convert.from_wire(row["W2"])
    assert restored.value is scalar
