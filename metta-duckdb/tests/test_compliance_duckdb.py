"""Purpose: the space compliance suite pointed at a real SQL backend.

The list-backed providers in test_compliance_suite.py prove the suite runs.
They do not prove it says anything, because a Python list satisfies the seam
trivially: it cannot over-approximate by accident, it cannot fail to filter,
and its atoms are already atoms. DuckDB is the shape the suite exists for, a
backend with its own query language, its own type system, and a translation on
both sides of every crossing.

The provider under test is the shipped example, imported rather than copied, so
this fails if the example stops satisfying the engine.
Guarantees:
  - the engine's space expectations hold of a provider backed by SQL, not only
    of one backed by a list
    [tested test_a_write_round_trip_leaves_the_provider_as_it_was]
Open Obligations:
  To Do: None
  Hacks: None
  Future Enhancements: None
"""

import importlib.util
import sys
from pathlib import Path

import metta_duckdb  # noqa: F401  -- imported for the sql row it registers
import pytest

from metta.testing import SpaceComplianceSuite

duckdb = pytest.importorskip("duckdb")

_EXAMPLES = Path(__file__).resolve().parents[3] / "examples"


def _duckdb_space_module():
    """The example, imported by path: it lives under examples/ rather than in
    the package, and its _common import needs examples/ on the path.
    """  # noqa: D205  -- the scenario narrative is one continuous invariant, not summary-and-body prose
    sys.path.insert(0, str(_EXAMPLES))
    try:
        spec = importlib.util.spec_from_file_location(
            "metta_example_duckdb_space", _EXAMPLES / "integration" / "duckdb_space.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.remove(str(_EXAMPLES))


class TestDuckDBSpaceComplies(SpaceComplianceSuite):  # noqa: D101  -- the local test double is documented by the scenario that constructs it
    # Its own in-memory database, so clearing it destroys nothing.
    destructive = True

    @pytest.fixture()
    def provider(self):  # noqa: D102  -- the test double method is documented by its containing scenario and protocol
        module = _duckdb_space_module()
        connection = duckdb.connect(":memory:")
        connection.execute("create table users (id integer, name text)")
        connection.execute(
            "insert into users values (1, 'Ada'), (2, 'Bob'), (3, 'Cy'), (1, 'Dee')"
        )
        return module.DuckDBSpace(connection)
