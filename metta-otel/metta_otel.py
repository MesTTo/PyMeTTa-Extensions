"""Purpose: the engine's own trace and counters as OpenTelemetry spans and metrics.

A reduction trace is a call tree with times on it, which is what a span is, and
`m.stats()` is four counters over a block, which is what a histogram is. Neither
needs a new mechanism: `spans()` turns a finished trace into back-dated spans
and `observe()` holds the engine's trace session over a block of your own calls
and does the same for what happened inside it.

    from opentelemetry import trace, metrics

    import metta_otel

    with metta_otel.observe(m, tracer=trace.get_tracer("app"),
                            meter=metrics.get_meter("app")):
        m.run("!(solve puzzle)")

This is a DISTRIBUTION of its own, `metta-otel`, because every line of it is
the OpenTelemetry API and pymetta names no library. Install it beside pymetta,
or take `pymetta[otel]`, which is what that extra now installs. Holding the
engine's trace session is NOT OpenTelemetry's, so that half stayed behind as
the seam's `observe` service and this package calls it; a registrant that emits
log lines or events instead of spans calls the same service.

Only `opentelemetry-api` is imported. The SDK, the exporters and the collector
are the deployment's, which is the split the API package exists for.

The `metta.*` loggers need no code here at all: they are ordinary
`logging.Logger`s, so attaching `opentelemetry.sdk._logs.LoggingHandler` to
`logging.getLogger("metta")` puts every engine, transport and provider message
into the same pipeline these spans go to.

Assumes:
  - `tracer` and `meter` are the API's own objects; anything with
    `start_span` and `create_histogram` works, which is what makes the test's
    in-memory exporter the same shape a collector is
Guarantees:
  - one span per recorded reduction, nested by the events' own depth, carrying
    the times the engine recorded rather than the times the spans were built
    [tested: test_a_trace_becomes_one_span_per_reduction,
    test_spans_nest_by_the_events_own_depth; commit=0fb68d75871c57f2421c335e9faef3561f8dfdd5]
  - a reduction that answered nothing is an ERROR span and one a bound cut is an
    ERROR span ending where the trace does, so neither is silently missing
    [tested: test_a_failed_reduction_is_an_error_span,
    test_a_reduction_a_bound_cut_ends_with_the_trace; commit=0fb68d75871c57f2421c335e9faef3561f8dfdd5]
  - observe() holds the engine's ONE trace session through the seam's
    `observe` service, so a trace or debug session inside it refuses and so
    does an observe inside one of those
    [tested: test_a_trace_inside_an_observed_block_refuses,
    test_observing_inside_a_debug_session_refuses; commit=94057a0f073c0fab0a35c42beff2c324d8a0addd]
  - the recording bound stops the RECORDING and never the observed work, because
    the work is the caller's and a telemetry budget must not become its error
    [tested: test_a_recording_bound_stops_the_recording_not_the_work; commit=0fb68d75871c57f2421c335e9faef3561f8dfdd5]
Owns resources:
  - observe() owns the engine's trace session for the block through the seam's
    `observe` service, which releases it in a finally, so a raising block
    leaves the wrappers off
    [tested: test_a_raising_block_still_releases_the_session; commit=94057a0f073c0fab0a35c42beff2c324d8a0addd]
Fails when:
  - a caller wants spans to arrive WHILE the block runs. The engine's tracer
    records into its own store and is read at the end -- "Nothing here streams
    yet" [source: engine/tracer.pl, the metta_trace_cell_budget/1 note] -- so
    every span is emitted at the block's end with the time it really happened.
    Back-dating is what `start_span(start_time=)` and `Span.end(end_time=)` are
    for, so an exporter sees the true shape either way, just later.
Open Obligations:
  To Do: None
  Hacks: None
  Future Enhancements: None.
"""

from __future__ import annotations

import time
from contextlib import ExitStack, contextmanager
from typing import TYPE_CHECKING, Any, Final

from metta import seam
from metta.atoms import Expression
from metta.errors import MettaError

if TYPE_CHECKING:
    from collections.abc import Iterator

__all__ = ["observe", "spans"]

#: The two seat services this package calls: the optional import with this
#: package's own guidance, and the held trace session. Bound at import, because
#: a service lookup is two dictionary reads and `spans` runs per trace.
require_module = seam.at("module").call()
_observed = seam.at("observe").call()

_OTEL_EXTRA: Final = (
    "the telemetry doors speak the OpenTelemetry API, which is not installed; "
    "install pymetta[otel]. Only the API package is needed here: the SDK, the "
    "exporters and the collector are your deployment's"
)

#: The four counters a stats block measures, as the instrument each becomes and
#: the unit UCUM spells for it [source:
#: https://opentelemetry.io/docs/specs/semconv/general/metrics/, the unit
#: guidance; `1` is a dimensionless count and `By` is bytes].
_HISTOGRAMS: Final = (
    ("metta.inferences", "inferences", "1", "engine steps the block spent"),
    ("metta.cputime", "cputime", "s", "engine CPU seconds the block spent"),
    ("metta.gc.freed", "gc_freed", "By", "bytes the block's collections freed"),
    ("metta.table_bytes", "table_bytes", "By", "answer-table bytes the block grew"),
)


def _head(term: Any) -> str:
    """A span's name: the head of the term that reduced.

    A span name is a low-cardinality label by OpenTelemetry's own rule, so it is
    the head rather than the whole term; the term itself is the `metta.term`
    attribute, where high cardinality belongs.
    """
    if isinstance(term, Expression) and term.children:
        return str(term.children[0])
    return str(term)


def spans(
    trace: Any,
    *,
    tracer: Any,
    space: str | None = None,
    start_time: int | None = None,
    parent: Any = None,
) -> None:
    """Emit one back-dated span per reduction of a finished trace.

        rec = m.record("!(fib 10)")
        metta.telemetry.spans(rec, tracer=trace.get_tracer("app"))

    `trace` is a `Trace` or a `Recording`, which carries one and knows its space.
    A `call` opens a span named by the head with `metta.term`, `metta.depth`,
    `metta.seq` and, when it is known, `metta.space`; the matching `exit` ends it
    with `metta.answer`; a `fail` ends it with status ERROR and
    `metta.exit=fail`; a reduction a bound cut before either ends where the trace
    does with `metta.exit=absent`. Nesting follows the events' own depth.

    `start_time` is the wall nanoseconds the traced run began at. Left out, the
    trace is placed so that it ENDS now, which is right for a trace just taken
    and is why a caller who knows when the run started should say so. `parent`
    is the span the outermost reductions hang under; left out they are roots,
    which is what a trace taken on its own is.
    """
    otel = require_module("opentelemetry.trace", _OTEL_EXTRA)
    events = getattr(trace, "events", trace)
    if space is None:
        space = getattr(trace, "space", None)
    if not events:
        return
    if start_time is None:
        start_time = time.time_ns() - max(event.time for event in events)
    ended = start_time + max(event.time for event in events)
    open_spans: dict[int, Any] = {}
    for event in events:
        moment = start_time + event.time
        if event.kind == "call":
            above = open_spans.get(event.depth - 1, parent)
            attributes = {
                "metta.term": str(event.term),
                "metta.depth": event.depth,
                "metta.seq": event.seq,
            }
            if space is not None:
                attributes["metta.space"] = str(space)
            open_spans[event.depth] = tracer.start_span(
                _head(event.term),
                context=None if above is None else otel.set_span_in_context(above),
                start_time=moment,
                attributes=attributes,
            )
            continue
        span = open_spans.pop(event.depth, None)
        if span is None:
            # An exit or a fail whose call the trace does not hold, which is
            # what a FILTERED trace answers: the filter selects events before
            # recording and a selected head can sit under an excluded one.
            continue
        if event.kind == "exit":
            span.set_attribute("metta.answer", str(event.answer))
        else:
            span.set_attribute("metta.exit", "fail")
            span.set_status(otel.Status(otel.StatusCode.ERROR, "the reduction answered nothing"))
        span.end(end_time=moment)
    # Whatever a bound cut mid-reduction. Deepest first, so a parent still
    # closes after its children.
    for depth in sorted(open_spans, reverse=True):
        span = open_spans[depth]
        span.set_attribute("metta.exit", "absent")
        span.set_status(otel.Status(otel.StatusCode.ERROR, "a bound cut the trace here"))
        span.end(end_time=ended)


def _histograms(meter: Any) -> list[tuple[Any, str]]:
    """The four instruments, created once per observed block."""
    return [
        (meter.create_histogram(name, unit=unit, description=description), field)
        for name, field, unit, description in _HISTOGRAMS
    ]


@contextmanager
def observe(
    m: Any,
    *,
    tracer: Any = None,
    meter: Any = None,
    name: str = "metta",
    max_events: int | None = None,
    filter: Any = None,  # noqa: A002 -- the trace door's own selector spelling
) -> Iterator[Any]:
    """Observe a block of engine work: its reductions as spans, its counters as metrics.

        with metta.telemetry.observe(m, tracer=tracer, meter=meter) as recorded:
            m.run("!(solve puzzle)")
        len(recorded)          # the events the block recorded

    With a `tracer`, the block runs under the engine's trace session and every
    compiled reduction inside it becomes a span under one span named `name`; the
    spans are emitted when the block ends, carrying the times the engine
    recorded. With a `meter`, the block's `m.stats()` deltas are recorded as the
    four histograms `metta.inferences`, `metta.cputime`, `metta.gc.freed` and
    `metta.table_bytes`, attributed with the space and the workload `name`.
    Either may be left out; both left out refuses, since there would be nothing
    to observe with.

    The yielded `Trace` fills in as the block ends, so reading it inside the
    block answers nothing and reading it after answers what was recorded, with
    `.stopped` naming the recording bound if one cut it. That bound stops the
    RECORDING and never the work: the work is yours and a telemetry budget must
    not become your program's error.

    `max_events` and `filter` are `m.trace`'s own two recording controls and mean
    what they mean there. The longhand for one program rather than a block is
    `spans(m.record(source), tracer=tracer)`, which records and emits in two
    steps instead of one.
    """
    if tracer is None and meter is None:
        msg = (
            "observe() needs a tracer, a meter, or both: with neither there is "
            "nothing for it to observe with. metta_otel.spans(trace, "
            "tracer=...) is the door for a trace you already have"
        )
        raise MettaError(msg)
    instruments = _histograms(meter) if meter is not None else []
    attributes = {"metta.space": str(m.name), "metta.workload": name}
    origin = time.time_ns()
    # The session is entered by hand rather than with a `with`, because the
    # spans can only be built AFTER it closes -- the engine's tracer records
    # into its own store and is read at the end -- and they must still be
    # built when the caller's block raised. arm= is the meter-only case: the
    # wrappers on every compiled function are not paid for by a caller who
    # asked for no spans, and the service yields an empty trace either way.
    session = ExitStack()
    recorded = session.enter_context(
        _observed(m, max_events, filter, arm=tracer is not None)
    )
    block = None if tracer is None else tracer.start_span(
        name, start_time=origin, attributes=attributes
    )
    try:
        with m.stats() as counters:
            yield recorded
    finally:
        try:
            session.close()
        finally:
            if block is not None:
                spans(
                    recorded,
                    tracer=tracer,
                    space=m.name,
                    start_time=origin,
                    parent=block,
                )
                block.end()
            for histogram, field in instruments:
                histogram.record(getattr(counters, field), attributes)
