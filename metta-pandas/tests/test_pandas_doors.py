"""Purpose: exercise the frame namespace and both declared receiver sugars.

Guarantees: the three doors preserve columns and duplicate rows
  [tested: test_pandas_namespace_and_short_sugars_share_the_row; commit=b615b5a33b43252ef9826e5387da7c9bd7f6b543].
Owns resources: the Answers context closes its source and the engine context.
"""

import metta_pandas
import pytest

from metta import G, MeTTa, S
from metta.results import Answers, Rows

pandas = pytest.importorskip("pandas")


def test_pandas_namespace_and_short_sugars_share_the_row():
    """Pandas namespace and short sugars share the row."""
    metta_pandas.register()
    rows = Rows(("value",), [(G(3),), (G(3),)])
    with MeTTa() as context:
        frame = context.tables.to_df(rows)
        assert frame.equals(rows.to_df())
        with Answers(iter(rows), columns=rows.columns) as answers:
            assert answers.to_df().equals(answers.to("pandas"))
        assert "to_df" in dir(rows)
        with Answers(iter([S.row(3)])) as terms:
            with pytest.raises(TypeError, match="table face needs caller bindings"):
                terms.to_df()
