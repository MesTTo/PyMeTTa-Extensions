"""Purpose: locate this distribution's generated MeTTa library.

Guarantees: the wheel includes the face advertised by metta.libraries
[tested: test_built_wheel_loads_its_generated_library; commit=WORKTREE].
"""

from pathlib import Path


def sources() -> Path:
    """Return the installed directory containing lib_arrays.metta."""
    return Path(__file__).parent
