"""Purpose: verify the registered live namespace owns and releases its views.

Guarantees: both accessor spellings preserve committed multiplicities
  [tested: test_live_namespace_preserves_view_lifecycle; commit=WORKTREE].
Owns resources: the context managers close both Live subscriptions and Space.
"""

import metta_live
import pytest

from metta import MeTTa, S, V


def test_live_namespace_preserves_view_lifecycle():
    """Live namespace preserves view lifecycle."""
    metta_live.register()
    with MeTTa() as context:
        with context.live.view(S.alert(V.level)) as named, context.self.live("(alert $level)") as called:
            context.add(S.alert(S.red), S.alert(S.red))
            assert named.count(S.alert(S.red)) == called.count(S.alert(S.red)) == 2
            context.remove(S.alert(S.red))
            assert named.rows == called.rows
            assert len(named) == len(called) == 1
        assert named._closed and called._closed
        with pytest.raises(ValueError, match="on must"):
            context.live.view(S.alert(V.level), on="invalid")
