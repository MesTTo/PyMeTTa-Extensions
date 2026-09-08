"""Purpose: verify deferred remote doors reach the existing transport and server.

Guarantees: a namespace-served socket accepts a namespace-connected client
  [tested: test_remote_namespace_preserves_transport_and_server_lifetimes; commit=b615b5a33b43252ef9826e5387da7c9bd7f6b543].
Owns resources: the server context closes its socket, worker, and answer cursors.
"""

import metta_remote

from metta import MeTTa, S
from metta.remote import RemoteSpace


def test_remote_namespace_preserves_transport_and_server_lifetimes():
    """Remote namespace preserves transport and server lifetimes."""
    metta_remote.register()
    with MeTTa() as context:
        context.add(S.remote_row(1))
        with context.remote.serve(spaces=[context.self.name]) as server:
            transport = context.self.remote.connect(server.url)
            assert transport.health()["protocol"]
            remote = RemoteSpace(transport, context.self.name)
            assert list(remote.atoms()) == [S.remote_row(1)]
            remote.add(S.remote_row(2))
            assert context.self.atoms() == [S.remote_row(1), S.remote_row(2)]
        assert server._closed
        assert not server._thread.is_alive() and not server._worker.thread.is_alive()
