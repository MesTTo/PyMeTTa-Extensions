<!--
Purpose: show every PyMeTTa integration, its installation and its extension points.
Guarantees: distribution names and entry-point rows come from each member's
pyproject.toml, and doors come from its registration source [source:
*/pyproject.toml and metta_*.py; commit=WORKTREE].
-->

# PyMeTTa Extensions

Separate Python distributions extend PyMeTTa by registering rows against declared extension points discovered through `metta.extensions`.

```bash
pip install metta-pandas               # one integration by name
pip install 'PyMeTTa[dataframes]'      # or the extra that names a set
```

PyMeTTa's core names no third-party integration library; pandas, polars, DuckDB, NumPy, faiss and torch reach it through the seam.

An extra is a convenience name for a set of distributions.

| Extra | Distributions |
|---|---|
| `dataframes` | `metta-tables`, `metta-pandas`, `metta-polars` |
| `arrays` | `metta-arrays`, `metta-faiss`, `metta-numpy` |
| `arrow` | `metta-nanoarrow`, `metta-pyarrow` |
| `sql` | `metta-tables`, `metta-duckdb`, `metta-sqlite` |
| `graphql` | `metta-graphql` |
| `models` | `metta-pydantic` |
| `live` | `metta-live` |
| `remote` | `metta-remote` |
| `das` | `metta-websocket` |
| `otel` | `metta-otel` |

Fifteen distributions advertise `metta.extensions` rows; `metta-otel` and `metta-benchmarking` are imported directly and register none.

## metta-pandas

Adds pandas DataFrame conversion and ingestion through `frame` row `pandas` and `door` row `metta-pandas`, including `rows.to_df()`, `answers.to_df()`, `m.tables.to_df(rows)` and `df.metta`.

```bash
pip install metta-pandas
```

```toml
[project.entry-points."metta.extensions"]
metta-pandas = "metta_pandas"
```

```python
from metta import G, Rows

rows = Rows(("n",), [(G(1),), (G(2),)])
assert rows.to_df()["n"].tolist() == [1, 2]
```

## metta-polars

Adds polars DataFrame conversion and ingestion through `frame` row `polars` and `door` row `metta-polars`, including `rows.to_pl()`, `answers.to_pl()`, `m.tables.to_pl(rows)` and `df.metta`.

```bash
pip install metta-polars
```

```toml
[project.entry-points."metta.extensions"]
metta-polars = "metta_polars"
```

```python
from metta import G, Rows

rows = Rows(("n",), [(G(1),), (G(2),)])
assert rows.to_pl()["n"].to_list() == [1, 2]
```

## metta-tables

Adds the table accessor through `door` row `metta-tables`, with `m.tables.add`, `m.tables.declare`, `m.tables.accessors` and `m.tables.sql_function`.

```bash
pip install metta-tables
```

```toml
[project.entry-points."metta.extensions"]
metta-tables = "metta_tables"
```

```python
from metta import S, V, space

m = space()
assert m.tables.add(S.user, [(1, "Ada")]) == 1
assert m.match(S.user(V.id, V.name)).to_dicts() == [{"id": 1, "name": "Ada"}]
```

## metta-arrays

Adds DLPack and Array API tensor operations, shape rules and `EmbeddingStore`, with `door` row `metta-arrays` exposing `m.arrays.install`, `m.arrays.uninstall`, `m.arrays.ops` and `m.arrays.backend`.

```bash
pip install metta-arrays
pip install metta-numpy                # the default backend used below
```

```toml
[project.entry-points."metta.extensions"]
metta-arrays = "metta_arrays_doors"

[project.entry-points."metta.libraries"]
lib_arrays = "metta_arrays_library:sources"
```

```python
from metta import space

m = space()
m.arrays.install()
assert m.arrays.backend() == "numpy"
m.arrays.uninstall()
```

## metta-numpy

Adds NumPy as the default array library through `array` row `numpy`, including column conversion and NumPy scalar strategies.

```bash
pip install metta-numpy
```

```toml
[project.entry-points."metta.extensions"]
metta-numpy = "metta_numpy"
```

```python
import numpy as np
from metta import G, Rows

rows = Rows(("n",), [(G(1),), (G(2),)])
assert np.asarray(rows.n).tolist() == [1, 2]
```

## metta-faiss

Adds an exact inner-product vector index through `index` row `faiss`, selectable by `EmbeddingStore(..., backend="faiss")` when `metta-arrays` is installed.

```bash
pip install metta-faiss
```

```toml
[project.entry-points."metta.extensions"]
metta-faiss = "metta_faiss"
```

```python
from metta import seam

backend = seam.index.find("faiss")
index = backend.build([[1.0, 0.0], [0.0, 1.0]])
assert backend.search(index, [1.0, 0.0], 1) == [(0, 1.0)]
```

## metta-nanoarrow

Adds Arrow C schema and stream capsules plus batch ingestion through `arrow` row `nanoarrow`, including `rows.__arrow_c_schema__()` and `rows.__arrow_c_stream__()`.

```bash
pip install metta-nanoarrow
```

```toml
[project.entry-points."metta.extensions"]
metta-nanoarrow = "metta_nanoarrow"
```

```python
import nanoarrow as na
from metta import G, Rows

rows = Rows(("n",), [(G(1),), (G(2),)])
with na.ArrayStream(rows) as stream:
    assert [row for chunk in stream.iter_chunks() for row in chunk.iter_tuples()] == [(1,), (2,)]
```

## metta-pyarrow

Adds Arrow IPC stream encoding, decoding and table concatenation through `ipc` row `pyarrow`.

```bash
pip install metta-pyarrow
```

```toml
[project.entry-points."metta.extensions"]
metta-pyarrow = "metta_pyarrow"
```

```python
from metta import seam

codec = seam.ipc.claim().row
schema = codec.schema(("n",), ("int64",), ("Number",))
assert codec.read(codec.stream(schema, ([1, 2],))).column("n").to_pylist() == [1, 2]
```

## metta-duckdb

Adds MeTTa heads as DuckDB scalar SQL functions through `sql` row `duckdb`, using each head's declared argument and result types.

```bash
pip install metta-duckdb
```

```toml
[project.entry-points."metta.extensions"]
metta-duckdb = "metta_duckdb"
```

```python
import duckdb
from metta import space, tables

m = space()
m.run("(: dbl (-> Number Number)) (= (dbl $x) (* 2 $x))")
with duckdb.connect(":memory:") as connection:
    tables.sql_function(connection, m.fn.dbl)
    assert connection.sql("select dbl(21)").fetchone() == (42.0,)
```

## metta-sqlite

Adds MeTTa heads as SQLite scalar SQL functions through `sql` row `sqlite3`, using each head's arity.

```bash
pip install metta-sqlite
```

```toml
[project.entry-points."metta.extensions"]
metta-sqlite = "metta_sqlite"
```

```python
import sqlite3
from contextlib import closing
from metta import space, tables

m = space()
m.run("(= (dbl $x) (* 2 $x))")
with closing(sqlite3.connect(":memory:")) as connection:
    tables.sql_function(connection, m.fn.dbl)
    assert connection.execute("select dbl(21)").fetchone() == (42,)
```

## metta-graphql

Adds GraphQL query execution against a served MeTTa space through `graphql` row `graphql-core`.

```bash
pip install metta-graphql
```

```toml
[project.entry-points."metta.extensions"]
metta-graphql = "metta_graphql"
```

```python
from metta import S, space
from metta.remote import Gateway

m = space()
m.run("(: users (-> Number String Bool))")
m.add(S.users(1, "Ada"))
with Gateway(m) as gateway:
    assert gateway("graphql", {"query": "{ users { x1 x2 } }"})["data"] == {
        "users": [{"x1": 1, "x2": "Ada"}]
    }
```

## metta-pydantic

Adds Pydantic models as MeTTa constructor expressions through `image` row `pydantic`, retaining field names, types and validation on reconstruction.

```bash
pip install metta-pydantic
```

```toml
[project.entry-points."metta.extensions"]
metta-pydantic = "metta_pydantic"
```

```python
from pydantic import BaseModel
from metta.convert import project

class Point(BaseModel):
    x: int
    y: int

assert str(project(Point(x=1, y=2)).atom) == "(Point 1 2)"
```

## metta-live

Adds maintained queries through `door` row `metta-live`, with `m.live.view(...)` and its callable form `m.live(...)`.

```bash
pip install metta-live
```

```toml
[project.entry-points."metta.extensions"]
metta-live = "metta_live"
```

```python
from metta import S, V, space

m = space()
with m.live(S.alert(V.level)) as alerts:
    m.add(S.alert(S.red))
    assert alerts.count(S.alert(S.red)) == 1
```

## metta-remote

Adds HTTP serving and connection accessors through `door` row `metta-remote`, with `m.remote.serve(...)` and `m.remote.connect(...)`.

```bash
pip install metta-remote
```

```toml
[project.entry-points."metta.extensions"]
metta-remote = "metta_remote"
```

```python
from metta import space

m = space()
with m.remote.serve(spaces=[m.name]) as server:
    assert server.url.startswith("http://127.0.0.1:")
```

## metta-websocket

Adds websocket-client timeout and closed-connection classification through `transport-error` row `websocket`, consumed by `is_transport_failure`.

```bash
pip install metta-websocket
```

```toml
[project.entry-points."metta.extensions"]
metta-websocket = "metta_websocket"
```

```python
from websocket import WebSocketTimeoutException
from metta import is_transport_failure

assert is_transport_failure(WebSocketTimeoutException("backend unavailable"))
```

## metta-otel

Adds reduction traces as OpenTelemetry spans and engine counters as metrics through `metta_otel.spans` and `metta_otel.observe`, calling the `observe` seam service without registering an entry point or row.

```bash
pip install metta-otel
```

```python
from opentelemetry import trace
from metta import S, space
from metta_otel import spans

m = space()
spans(m.trace(S["+"](1, 2)), tracer=trace.get_tracer("example"))  # uses your configured tracer
```

## metta-benchmarking

Adds counter baselines, instruction pins and scaling checks through `metta_benchmarking`, with no registered entry point, door or seam row.

```bash
pip install metta-benchmarking
```

```python
from metta import S
from metta_benchmarking import count_atoms

assert count_atoms(S.edge(S.a, S.b)) == 4
```

Write your own integration with [EXTENDING.md](https://github.com/MesTTo/MeTTa/blob/main/EXTENDING.md#the-python-seat).
