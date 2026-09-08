"""Purpose: register the live-query accessor without loading its implementation.

Guarantees: the live accessor uses the existing query maintenance and lifetime
  protocol [tested: test_live_namespace_preserves_view_lifecycle; commit=b615b5a33b43252ef9826e5387da7c9bd7f6b543].
Owns resources: registration holds immutable metadata only. A returned Live
  owns its subscriptions until close or abandonment [tested:
  test_live_namespace_preserves_view_lifecycle; commit=b615b5a33b43252ef9826e5387da7c9bd7f6b543].
"""

from __future__ import annotations

from typing import Any

from metta import seam
from metta.atoms import parse
from metta.doors import (
    AnswersAs,
    Body,
    Door,
    Kind,
    Owner,
    Provider,
    Receiver,
    Signature,
    Tier,
)
from metta.vocabularies import Determinism, EffectClass, SubscriptionEdge


def view(space, *query: Any, on: SubscriptionEdge = SubscriptionEdge.both,
         strategy: str | None = None) -> Any:
    """Maintain a query's multiset through this space's committed writes.

    The returned Live owns its subscriptions. close() releases them, and
    changes() reads its progress and deltas. strategy selects pattern, heads,
    or tabled maintenance; omitting it selects from the query's shape.
    """
    from metta.live import Live  # noqa: PLC0415  -- only calling the accessor loads live queries

    return Live(space, *(parse(part) if isinstance(part, str) else part for part in query),
                on=on, strategy=strategy)


# closed-set: decides; policy=this package owns these accessor contracts; reads=the named implementation signatures checked by tools/doorgen.py
DOORS: tuple[Door, ...] = (
    Door(
        owner=Owner.namespace, name='view', kind=Kind.query,
        signatures=(Signature('space, *query: Any, on: SubscriptionEdge=SubscriptionEdge.both, strategy: str | None=None', returns='Any'),), answers=AnswersAs.value,
        effect=EffectClass.oracleIO, determinism=Determinism.det,
        tiers=(Tier.sync, Tier.context, Tier.async_), body=Body('metta_live', 'view', Receiver.space),
        provider=Provider('metta-live', 'live', callable=True),
        docs="Maintain a query's multiset through this space's committed writes.\n\nThe returned Live owns its subscriptions. close() releases them, and\nchanges() reads its progress and deltas. strategy selects pattern, heads,\nor tabled maintenance; omitting it selects from the query's shape.",
        evidence=('extensions/python/ext/metta-live/tests/test_live_doors.py::test_live_namespace_preserves_view_lifecycle',),
        alias="live",
    ),
)


def register() -> None:
    """Publish this package's complete accessor declaration atomically."""
    seam.door.register('metta-live', doors=DOORS)


register()
