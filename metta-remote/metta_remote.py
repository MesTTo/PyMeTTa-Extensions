"""Purpose: register the remote accessor using deferred implementation references.

Guarantees: registration imports no implementation and accessor calls preserve
  its behavior [tested: test_remote_namespace_preserves_transport_and_server_lifetimes; commit=b615b5a33b43252ef9826e5387da7c9bd7f6b543].
"""

from __future__ import annotations

from metta import seam
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
from metta.vocabularies import Determinism, EffectClass

# closed-set: decides; policy=this package owns these accessor contracts; reads=the named implementation signatures checked by tools/doorgen.py
DOORS: tuple[Door, ...] = (
    Door(
        owner=Owner.namespace, name='connect', kind=Kind.provider,
        signatures=(Signature('url: str, timeout: float=30.0, *, token: str | None=None, headers: dict[str, str] | None=None, ssl_context: Any=None', returns='Transport'),), answers=AnswersAs.callable,
        effect=EffectClass.oracleIO, determinism=Determinism.det,
        tiers=(Tier.sync, Tier.context), body=Body('metta.remote', 'connect', Receiver.none),
        provider=Provider('metta-remote', 'remote'),
        docs="The HTTP transport for a serve()d engine: one POST per operation,\nJSON both ways, errors surfaced with the remote's own message.\n\ntoken sends Bearer authentication, headers adds anything else a\ndeployment needs (an API key, a tenant id), and ssl_context is\nPython's own ssl.SSLContext for https urls, certificate pinning\nincluded, so the transport composes with whatever security the\nserving side asks for. Only absolute http and https URLs are accepted.\nCredentials require https. Each mutation negotiates a replay key through\nGET /health before its POST; authorization policies must permit that read.\nAn unadvertised extension leaves mutations unkeyed, and OutcomeUnknown\nrefuses to resend those requests after a lost response.",
        evidence=('extensions/python/ext/metta-remote/tests/test_remote_doors.py::test_remote_namespace_preserves_transport_and_server_lifetimes',),
    ),
    Door(
        owner=Owner.namespace, name='serve', kind=Kind.lifecycle,
        signatures=(Signature("m, host: str='127.0.0.1', port: int=0, spaces: list[str] | None=None, *, token: str | None=None, authorize: Callable[[Request], bool] | None=None, ssl_context: Any=None, cursor_idle: float=_CURSOR_IDLE, cursor_limit: int=_CURSOR_LIMIT, mutation_ttl: float=_MUTATION_TTL, mutation_limit: int=_MUTATION_LIMIT", returns='Server'),), answers=AnswersAs.context,
        effect=EffectClass.oracleIO, determinism=Determinism.det,
        tiers=(Tier.sync, Tier.context), body=Body('metta.remote', 'serve', Receiver.space),
        provider=Provider('metta-remote', 'remote'),
        docs="Expose this engine's spaces over HTTP; port 0 picks a free one.\n\nEvery operation answers for the space the request names, restricted\nto `spaces` when given. Security is the caller's to define, library\nfashion: token requires Bearer authentication, authorize is the\ngeneral hook (a Request in, carrying the operation, the space and\nthe headers, and a verdict out, so read-only, per-space and\nper-tenant policies all fit), and ssl_context, Python's own\nssl.SSLContext with a certificate loaded, serves TLS directly;\nanything heavier still composes behind a fronting proxy. match runs\nthe engine's own match with the pattern as its template, so the\ninstantiated atoms cross, and the caller's engine re-unifies them.\n\n`cursor_idle` and `cursor_limit` bound the ask/next/stop lifecycle's\nserver-side state: how long a cursor nobody pulls from survives, and\nhow many live at once before a further ask is refused. The defaults\nare pengines' own, 300 seconds and a ceiling.\n\nmutation_ttl and mutation_limit bound the keyed mutation replay ledger,\nas documented on Gateway. Authorization must admit health for clients\nthat negotiate mutation keys.\n\nA context is a PROCESS: serving and attaching within one process\ncannot join through the local engine, because one runtime lock guards\nboth sides of that call and the serving thread would wait on the very\nevaluation that is waiting on it. Two engines, two processes, is the\ndeployment this exists for; in-process, spaces already share the\nengine and need no wire. Gateway is the same protocol with no\ntransport under it, for a test or a framework that wants the\noperations without a socket.\n\nm may be a context or a space, as Gateway takes either.",
        evidence=('extensions/python/ext/metta-remote/tests/test_remote_doors.py::test_remote_namespace_preserves_transport_and_server_lifetimes',),
    ),
)


def register() -> None:
    """Publish this package's complete accessor declaration atomically."""
    seam.door.register('metta-remote', doors=DOORS)


register()
