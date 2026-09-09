"""Purpose: prove the graphql-core row executes a query against a served space.

The SDL is the seat's own and needs no library; running a query against it is
what this row does. `POST /graphql` therefore works with this package installed
and refuses naming the door without it.
Open Obligations:
  To Do: None
  Hacks: None
  Future Enhancements: None
"""

from __future__ import annotations

import metta_graphql  # noqa: F401  -- imported for the graphql row it registers
import pytest

import metta.remote._gateway as _moved_metta_remote__gateway
from metta import seam

pytest.importorskip("graphql")


@pytest.fixture()
def served(scratch_space):
    """A space with two rows a schema can be projected from."""
    scratch_space.run("(: users (-> Number String Bool))")
    scratch_space.add(scratch_space.parse('(users 1 "Ada")'))
    scratch_space.add(scratch_space.parse('(users 2 "Bob")'))
    return scratch_space


def test_the_row_claims_the_graphql_point():
    """An ownership point: this row claims because its library is importable."""
    claim = seam.graphql.claim()
    assert claim is not None
    assert claim.name == "graphql-core"


def test_a_query_answers_the_spaces_own_rows(served):
    """The whole door, end to end: SDL from the seat, execution from here."""
    with _moved_metta_remote__gateway.Gateway(served) as gateway:
        answer = gateway("graphql", {"query": "{ users { x1 x2 } }"})
    assert answer["data"]["users"] == [
        {"x1": 1, "x2": "Ada"},
        {"x1": 2, "x2": "Bob"},
    ]


def test_a_malformed_request_is_refused(served):
    """GraphQL over HTTP's own shape: a request needs a `query` string."""
    from metta import MettaError

    with _moved_metta_remote__gateway.Gateway(served) as gateway:
        with pytest.raises(MettaError, match="needs a `query` field"):
            gateway("graphql", {})
        with pytest.raises(MettaError, match="variables must be an object"):
            gateway("graphql", {"query": "{ users { x1 } }", "variables": [1]})


def test_the_seats_scalars_are_what_serialize(served):
    """This row attaches serializers it did not write.

    The `schema` field is handed `{name: (serialize, parse_value)}` built by
    the seat, so what a Number column renders as is the seat's decision and
    not graphql-core's identity serializer, which would hand an atom object to
    a JSON encoder.
    """
    attached = {}
    row = seam.graphql.find("graphql-core")
    real = row.fields["schema"]

    def watched(sdl, scalars):
        attached.update(scalars)
        return real(sdl, scalars)

    row.fields["schema"] = watched
    try:
        with _moved_metta_remote__gateway.Gateway(served) as gateway:
            answer = gateway("graphql", {"query": "{ users { x1 } }"})
    finally:
        row.fields["schema"] = real
    assert sorted(attached) == ["Atom", "Number"]
    assert answer["data"]["users"][0]["x1"] == 1
