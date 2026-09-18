"""Purpose: register the live-query accessor without loading its implementation.

Guarantees: the live accessor uses the existing query maintenance and lifetime
  protocol [tested: test_live_namespace_preserves_view_lifecycle; commit=b615b5a33b43252ef9826e5387da7c9bd7f6b543].
Owns resources: registration holds immutable metadata only. A returned Live
  owns its subscriptions until close or abandonment [tested:
  test_live_namespace_preserves_view_lifecycle; commit=b615b5a33b43252ef9826e5387da7c9bd7f6b543].
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import metta.doors as _doors
from metta import parse, seam

if TYPE_CHECKING:
    from metta import SpaceLike
from metta.vocabularies import SubscriptionEdge


@_doors.door(
    kind=_doors.Kind.query,
    answers=_doors.AnswersAs.value,
    effect=_doors.EffectClass.oracleIO,
    determinism=_doors.Determinism.det,
    tiers=(_doors.Tier.sync, _doors.Tier.context, _doors.Tier.async_),
    provider=_doors.Provider('metta-live', 'live', callable=True),
    evidence=('ext/metta-live/tests/test_live_doors.py::test_live_namespace_preserves_view_lifecycle',),
    alias='live',
)
def view(space: SpaceLike, *query: Any, on: SubscriptionEdge = SubscriptionEdge.both,
         strategy: str | None = None) -> Any:
    """Maintain a query's multiset through this space's committed writes.

    The returned Live owns its subscriptions. close() releases them, and
    changes() reads its progress and deltas. strategy selects pattern, heads,
    or tabled maintenance; omitting it selects from the query's shape.
    """
    from metta.live import Live  # noqa: PLC0415  -- only calling the accessor loads live queries

    return Live(space, *(parse(part) if isinstance(part, str) else part for part in query),
                on=on, strategy=strategy)






def register() -> None:
    """Publish this package's complete accessor declaration atomically."""
    seam.door.register('metta-live', doors=_doors.declarations(__name__))


register()
