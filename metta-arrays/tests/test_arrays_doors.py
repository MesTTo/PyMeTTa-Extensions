"""Purpose: verify deferred array namespace installation and its inverse.

Guarantees: each accessor reads or retires the installation owned by its receiver
  [tested: test_array_namespace_preserves_installation_and_withdrawal; commit=b615b5a33b43252ef9826e5387da7c9bd7f6b543].
Owns resources: uninstall releases process-global operations before Space closes.
"""

import metta_arrays_doors
import metta_numpy
import pytest

from metta import Expression, MeTTa, MettaError

numpy = pytest.importorskip("numpy")


def test_array_namespace_preserves_installation_and_withdrawal():
    """Array namespace preserves installation and withdrawal."""
    metta_arrays_doors.register()
    metta_numpy.register()
    with MeTTa() as context:
        names = context.arrays.install(default=numpy)
        try:
            assert context.self.arrays.ops() == names
            assert context.arrays.backend() == "numpy"
            assert context.run("!(t-shape (zeros 2 3))") == [[Expression(2, 3)]]
        finally:
            removed = context.arrays.uninstall()
        assert set(removed) <= set(names)
        with pytest.raises(MettaError, match="install"):
            context.arrays.ops()
