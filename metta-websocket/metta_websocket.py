"""Purpose: websocket-client's own timeouts, so an absent backend reads as absent.

`metta.errors.is_transport_failure` asks whether a failure means the backend is
ABSENT rather than wrong, which decides whether a client retries or reports.
The obvious test does not separate them: a socket timeout raises OSError, but
websocket-client's own timeout does NOT subclass it, so "is the cause an
OSError" misses exactly the shape a broken event stream takes under load. One
row against the seat's `transport-error` point says which classes those are.

Installed beside pymetta, or through `pip install 'pymetta[das]'`. The row
holds the module NAME and imports nothing; the classes are read only when a
failure is actually classified, and only if websocket-client is imported.

Assumes:
  - the two exception classes keep their names, which is the only thing this
    row knows about the library
Guarantees:
  - a websocket timeout and a closed stream both read as transport failures,
    and an ordinary application error does not [tested: tests/test_websocket.py;
    commit=94057a0f073c0fab0a35c42beff2c324d8a0addd]
Open Obligations:
  To Do: None
  Hacks: None
  Future Enhancements: None
"""

from __future__ import annotations

from metta import seam


def _classes(module) -> tuple[type[BaseException], ...]:
    """websocket-client's own timeout and closed-stream exceptions."""
    return (module.WebSocketTimeoutException, module.WebSocketConnectionClosedException)


def register() -> None:
    """This package's one row, against the seat's `transport-error` point."""
    seam.transport_error.register("websocket", module="websocket", classes=_classes)


register()
