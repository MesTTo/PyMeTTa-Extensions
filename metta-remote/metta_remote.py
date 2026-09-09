"""Purpose: register the remote accessor using deferred implementation references.

Guarantees: registration imports no implementation and accessor calls preserve
  its behavior [tested: test_remote_namespace_preserves_transport_and_server_lifetimes; commit=b615b5a33b43252ef9826e5387da7c9bd7f6b543].
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

import metta.doors as _doors
from metta import seam
from metta.remote._defaults import _CURSOR_IDLE, _CURSOR_LIMIT, _MUTATION_LIMIT, _MUTATION_TTL

if TYPE_CHECKING:
    from collections.abc import Callable

    from metta import SpaceLike
    from metta.remote._gateway import Request, Server
    from metta.remote._transport import Transport



@_doors.door(
    kind=_doors.Kind.provider,
    answers=_doors.AnswersAs.callable,
    effect=_doors.EffectClass.oracleIO,
    determinism=_doors.Determinism.det,
    tiers=(_doors.Tier.sync, _doors.Tier.context),
    provider=_doors.Provider('metta-remote', 'remote'),
    evidence=('extensions/python/ext/metta-remote/tests/test_remote_doors.py::test_remote_namespace_preserves_transport_and_server_lifetimes',),
)
def connect(url: str, timeout: float=30.0, *, token: str | None=None, headers: dict[str, str] | None=None, ssl_context: Any=None) -> Transport:
    """The HTTP transport for a serve()d engine: one POST per operation,
    JSON both ways, errors surfaced with the remote's own message.

    token sends Bearer authentication, headers adds anything else a
    deployment needs (an API key, a tenant id), and ssl_context is
    Python's own ssl.SSLContext for https urls, certificate pinning
    included, so the transport composes with whatever security the
    serving side asks for. Only absolute http and https URLs are accepted.
    Credentials require https. Each mutation negotiates a replay key through
    GET /health before its POST; authorization policies must permit that read.
    An unadvertised extension leaves mutations unkeyed, and OutcomeUnknown
    refuses to resend those requests after a lost response.
    """  # noqa: D205 -- preserve the declared documentation
    return import_module('metta.remote').connect(url, timeout, token=token, headers=headers, ssl_context=ssl_context)

@_doors.door(
    kind=_doors.Kind.lifecycle,
    answers=_doors.AnswersAs.context,
    effect=_doors.EffectClass.oracleIO,
    determinism=_doors.Determinism.det,
    tiers=(_doors.Tier.sync, _doors.Tier.context),
    provider=_doors.Provider('metta-remote', 'remote'),
    evidence=('extensions/python/ext/metta-remote/tests/test_remote_doors.py::test_remote_namespace_preserves_transport_and_server_lifetimes',),
)
def serve(m: SpaceLike, host: str='127.0.0.1', port: int=0, spaces: list[str] | None=None, *, token: str | None=None, authorize: Callable[[Request], bool] | None=None, ssl_context: Any=None, cursor_idle: float=_CURSOR_IDLE, cursor_limit: int=_CURSOR_LIMIT, mutation_ttl: float=_MUTATION_TTL, mutation_limit: int=_MUTATION_LIMIT) -> Server:
    """Expose this engine's spaces over HTTP; port 0 picks a free one.

    Every operation answers for the space the request names, restricted
    to `spaces` when given. Security is the caller's to define, library
    fashion: token requires Bearer authentication, authorize is the
    general hook (a Request in, carrying the operation, the space and
    the headers, and a verdict out, so read-only, per-space and
    per-tenant policies all fit), and ssl_context, Python's own
    ssl.SSLContext with a certificate loaded, serves TLS directly;
    anything heavier still composes behind a fronting proxy. match runs
    the engine's own match with the pattern as its template, so the
    instantiated atoms cross, and the caller's engine re-unifies them.

    `cursor_idle` and `cursor_limit` bound the ask/next/stop lifecycle's
    server-side state: how long a cursor nobody pulls from survives, and
    how many live at once before a further ask is refused. The defaults
    are pengines' own, 300 seconds and a ceiling.

    mutation_ttl and mutation_limit bound the keyed mutation replay ledger,
    as documented on Gateway. Authorization must admit health for clients
    that negotiate mutation keys.

    A context is a PROCESS: serving and attaching within one process
    cannot join through the local engine, because one runtime lock guards
    both sides of that call and the serving thread would wait on the very
    evaluation that is waiting on it. Two engines, two processes, is the
    deployment this exists for; in-process, spaces already share the
    engine and need no wire. Gateway is the same protocol with no
    transport under it, for a test or a framework that wants the
    operations without a socket.

    m may be a context or a space, as Gateway takes either.
    """
    return import_module('metta.remote').serve(m, host, port, spaces, token=token, authorize=authorize, ssl_context=ssl_context, cursor_idle=cursor_idle, cursor_limit=cursor_limit, mutation_ttl=mutation_ttl, mutation_limit=mutation_limit)


def register() -> None:
    """Publish this package's complete accessor declaration atomically."""
    seam.door.register('metta-remote', doors=_doors.declarations(__name__))


register()
