"""Purpose: exercise the frame namespace and both declared receiver sugars.

Guarantees: the three doors preserve columns and duplicate rows
  [tested: test_polars_namespace_and_short_sugars_share_the_row; commit=b615b5a33b43252ef9826e5387da7c9bd7f6b543].
Owns resources: the Answers context closes its source and the engine context.
"""

import metta_polars
import pytest

from metta import Answers, G, MeTTa, Rows, S

polars = pytest.importorskip("polars")


def test_polars_namespace_and_short_sugars_share_the_row():
    """Polars namespace and short sugars share the row."""
    metta_polars.register()
    rows = Rows(("value",), [(G(3),), (G(3),)])
    with MeTTa() as context:
        frame = context.tables.to_pl(rows)
        assert frame.equals(rows.to_pl())
        with Answers(iter(rows), columns=rows.columns) as answers:
            assert answers.to_pl().equals(answers.to("polars"))
        assert "to_pl" in dir(rows)
        with Answers(iter([S.row(3)])) as terms:
            with pytest.raises(TypeError, match="table face needs caller bindings"):
                terms.to_pl()
