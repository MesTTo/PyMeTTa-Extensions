"""Purpose: prove a pydantic model crosses as a constructor expression.

And that this row is asked BEFORE the four structural readings the seat ships,
which is what the fallback flag buys: a model is also a class with
`__match_args__` under some declarations, and the specific reading has to win
however the two distributions loaded.
Open Obligations:
  To Do: None
  Hacks: None
  Future Enhancements: None
"""

from __future__ import annotations

import metta_pydantic  # noqa: F401  -- imported for the image row it registers
import pytest

from metta import seam
from metta.convert import project

pydantic = pytest.importorskip("pydantic")


class PydanticProbe(pydantic.BaseModel):
    """A plain model: two declared fields and nothing else.

    The name is this file's own, because a projected class CLAIMS its type
    name in a PROCESS-WIDE registry: a second class of the same name anywhere
    in the suite meets the first one's claim, whatever module or function it
    was defined in, and pytest-randomly decides which of the two gets there
    first [measured 2026-09-08: `Reading` collided with a local class of that
    name in tests/ch03_atoms_and_expressions/test_convert.py].
    """

    x: int
    y: int


class PydanticProbeLoose(pydantic.BaseModel):
    """A model that admits extra fields, which cannot cross."""

    model_config = pydantic.ConfigDict(extra="allow")
    x: int


def test_the_row_is_registered_against_the_image_point():
    """One row, and NOT a fallback: it is asked before the structural four."""
    row = seam.image.find("pydantic")
    assert row is not None
    assert not row.fallback


def test_a_model_projects_as_its_own_constructor():
    """Fields become arguments of an expression named for the class."""
    assert str(project(PydanticProbe(x=1, y=2)).atom) == "(PydanticProbe 1 2)"


def test_a_model_is_not_read_as_a_match_args_class():
    """The claim comes from THIS row, not from one of the structural fallbacks."""
    claim = seam.image.claim(PydanticProbe)
    assert claim is not None
    assert claim.name == "pydantic"
    assert [row.name for row in seam.image.rows()].index("pydantic") < [
        row.name for row in seam.image.rows()
    ].index("match-args")


def test_extra_fields_refuse_rather_than_vanish():
    """A field the model did not declare would be lost, so the projection stops."""
    with pytest.raises(TypeError, match="extra fields"):
        project(PydanticProbeLoose(x=1, y=2))
