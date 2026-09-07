"""Purpose: prove the websocket row classes an absent backend as absent.

`is_transport_failure` decides whether a client retries or reports. The library
this package names raises a timeout that does NOT subclass OSError, which is
the shape the obvious test misses and the whole reason for the row.
Open Obligations:
  To Do: None
  Hacks: None
  Future Enhancements: None
"""

from __future__ import annotations

import metta_websocket  # noqa: F401  -- imported for the transport-error row
import pytest

from metta import seam
from metta.errors import is_transport_failure

websocket = pytest.importorskip("websocket")


def test_the_row_is_registered_against_the_transport_error_point():
    """One row, holding the module NAME rather than the module."""
    row = seam.transport_error.find("websocket")
    assert row is not None
    assert row.module == "websocket"


def test_a_websocket_timeout_reads_as_an_absent_backend():
    """The class this library raises under load, which is not an OSError."""
    assert not isinstance(websocket.WebSocketTimeoutException(), OSError)
    assert is_transport_failure(websocket.WebSocketTimeoutException("gone"))


def test_a_closed_stream_reads_as_an_absent_backend():
    """The other half: a stream the peer closed is absence, not an error."""
    assert is_transport_failure(websocket.WebSocketConnectionClosedException("closed"))


def test_an_application_error_does_not():
    """Wrong is not absent, which is the distinction the door exists for."""
    assert not is_transport_failure(ValueError("wrong, not absent"))
