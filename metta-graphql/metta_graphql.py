"""Purpose: graphql-core as the executor of a served space's GraphQL schema.

A gateway publishes its schema as SDL TEXT, which this seat writes itself and
which needs no library at all; `GET /graphql` therefore answers with nothing
installed. RUNNING a query against that schema needs an implementation of the
language, and this package is one: one row against the seat's `graphql` point,
an ownership point like `arrow` and `ipc`, so `POST /graphql` works when it is
installed and refuses naming the door when it is not.

graphql-core is the reference port of GraphQL.js and has no dependency of its
own, which is why it is the row here; pymetta names it nowhere.

Assumes:
  - `scalars` maps a scalar NAME to `(serialize, parse_value)`, both built by
    the seat, so nothing here knows how an atom renders
Guarantees:
  - the two custom scalars get the seat's serializers under whichever name
    this graphql-core release reads them by: 3.2 calls `serialize` and 3.3
    renames it `coerce_output_value` and keeps both on the class
    [source: https://github.com/graphql-python/graphql-core, GraphQLScalarType's
    constructor assigning serialize and parse_value as plain attributes;
    tested: tests/test_graphql.py; commit=94057a0f073c0fab0a35c42beff2c324d8a0addd]
  - a request with no `query` field, and one whose `variables` is not an
    object, are refused by name rather than executed
    [source: https://graphql.github.io/graphql-over-http/draft/, the POST
    request body; tested: tests/test_graphql.py::test_a_malformed_request_is_refused;
    commit=94057a0f073c0fab0a35c42beff2c324d8a0addd]
Open Obligations:
  To Do: None
  Hacks: None
  Future Enhancements: None
"""

from __future__ import annotations

from typing import Any, Final

from metta import MettaError, seam

_MISSING: Final = (
    "executing a GraphQL query needs graphql-core, which is not installed; "
    "install pymetta[graphql]. GET /graphql still answers the schema without it"
)

optional_module = seam.at("optional-module").call()
require_module = seam.at("module").call()


def _claims() -> Any:
    """graphql-core, when it is installed; nothing otherwise."""
    return optional_module("graphql")


def _schema(sdl: str, scalars: Any) -> Any:
    """The executable schema for one SDL text, with the seat's scalars on it.

    `build_schema` gives a custom scalar the identity serializer, which would
    hand an atom object to a JSON encoder; each scalar the seat declared is
    given its own here, under whichever attribute this release reads.
    """
    graphql = require_module("graphql", _MISSING)
    schema = graphql.build_schema(sdl)
    for name, (serialize, parse_value) in scalars.items():
        scalar = schema.type_map.get(name)
        if scalar is None:
            continue
        # graphql-core 3.2 calls `serialize`; 3.3 renames it
        # `coerce_output_value` and keeps both on the class, so whichever
        # exists is set.
        for attribute in ("serialize", "coerce_output_value"):
            if hasattr(scalar, attribute):
                setattr(scalar, attribute, serialize)
        if parse_value is not None:
            scalar.parse_value = parse_value
    return schema


def _execute(schema: Any, root: Any, request: Any) -> dict[str, Any]:
    """Run one GraphQL request against a schema and answer the response body.

    The request is GraphQL over HTTP's own shape, `query`, `variables` and
    `operationName`, and the answer is its `data` and `errors`.
    """
    graphql = require_module("graphql", _MISSING)
    query = request.get("query")
    if not isinstance(query, str) or not query.strip():
        msg = "a GraphQL request needs a `query` field holding the document text"
        raise MettaError(msg)
    variables = request.get("variables")
    if variables is not None and not isinstance(variables, dict):
        msg = f"GraphQL variables must be an object, got {type(variables).__name__}"
        raise MettaError(msg)
    result = graphql.graphql_sync(
        schema,
        query,
        root_value=dict(root),
        variable_values=variables,
        operation_name=request.get("operationName"),
    )
    answer: dict[str, Any] = {"data": result.data}
    if result.errors:
        answer["errors"] = [error.formatted for error in result.errors]
    return answer


def register() -> None:
    """This package's one row, against the seat's `graphql` point."""
    seam.graphql.register(
        "graphql-core",
        claims=_claims,
        schema=_schema,
        execute=_execute,
        missing=_MISSING,
    )


register()
