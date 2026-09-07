"""Purpose: give every extension package's tests the fixtures the seat ships.

`metta.pytest_plugin` is a `pytest11` entry point, so an INSTALLED pymetta
carries `metta` and `scratch_space` into any suite; a checkout does not, and
these tests run from a checkout. `tests/conftest.py` registers it for the core
suite for the same reason and with the same guard.

Assumes: a `metta` importable from the seat's own directory, which the runner's
  `pythonpath` setting provides.
Guarantees:
  - a member's test asks for `metta` or `scratch_space` and gets the shipped
    fixture [tested: ext/metta-numpy/tests/test_numpy.py; commit=WORKTREE]
Open Obligations:
  To Do: None
  Hacks: None
  Future Enhancements: None
"""

from __future__ import annotations

import pytest

from metta import pytest_plugin as metta_pytest_plugin


def pytest_configure(config: pytest.Config) -> None:
    """Register the shipped fixtures only when entry-point discovery did not."""
    if not config.pluginmanager.is_registered(metta_pytest_plugin):
        config.pluginmanager.register(metta_pytest_plugin, "metta-source")
