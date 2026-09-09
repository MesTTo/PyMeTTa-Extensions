"""Purpose: the trace as OpenTelemetry spans and the counters as histograms.

A reduction trace is a call tree with times on it, so the questions here are the
ones a span carries: is there one per reduction, is it nested the way the
reductions were, does it carry the time the engine recorded rather than the time
the span was built, and does a reduction that answered nothing say so.

Guarantees:
  - one span per reduction, nested by depth, at the recorded times [tested:
    test_a_trace_becomes_one_span_per_reduction,
    test_spans_nest_by_the_events_own_depth,
    test_a_span_carries_the_time_the_engine_recorded; commit=8cdcb4a74b13418097d56c29ac2d296f14c7940e]
  - a block's spans hang under one span and its counters become four histograms
    [tested: test_an_observed_block_hangs_its_reductions_under_one_span,
    test_an_observed_block_records_four_histograms; commit=8cdcb4a74b13418097d56c29ac2d296f14c7940e]
  - the session is released whatever the block does, and a second session opens
    after [tested: test_a_raising_block_still_releases_the_session; commit=8cdcb4a74b13418097d56c29ac2d296f14c7940e]
  - this file's footprint on the engine is the size of its subject and not of
    its scenario count: the two functions every reading scenario reduces are
    compiled ONCE, because a child space falls back to `&self` for equations
    and a per-scenario definition leaves another copy of them there. Fifteen
    copies were enough to move a neighbouring file's inference comparison by
    two [measured 2026-09-07: `pytest ext/metta-otel/tests/test_otel.py
    tests/ch14_seeing_your_program/test_explain_plan.py -p no:randomly` failed
    test_analyze_numbers_equal_the_stats_of_the_same_query at 660 against 658,
    and passes with the shared definition; commit=8cdcb4a74b13418097d56c29ac2d296f14c7940e]
Open Obligations:
  To Do: None
  Hacks: None
  Future Enhancements: None.
"""

from __future__ import annotations

import pytest
from metta_otel import observe, spans

from metta import MettaError, S

pytest.importorskip("opentelemetry.trace")
pytest.importorskip("opentelemetry.sdk.trace")


@pytest.fixture()
def space(metta):
    """Each scenario that compiles a function of its own gets its own space."""
    with metta._new_space() as scratch:
        yield scratch


@pytest.fixture(scope="module")
def nested(metta):
    """A two-level reduction, so depth is something the spans have to carry.

    ONE space for the scenarios that only read it. A child space falls back to
    `&self` for equations, so a per-scenario definition would put another copy
    of the same two equations in the process's home space for every scenario
    here; the scenarios that need their own functions take the function-scoped
    `metta` fixture above instead.
    """
    with metta._new_space() as shared:
        shared.run("(= (tl-double $x) (* 2 $x))")
        shared.run("(= (tl-quad $x) (tl-double (tl-double $x)))")
        yield shared


@pytest.fixture()
def exporter():
    """The SDK's in-memory exporter, which is the shape a collector is."""
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
        InMemorySpanExporter,
    )

    sink = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(sink))
    sink.tracer = provider.get_tracer("metta-test")
    return sink


@pytest.fixture()
def reader():
    """The SDK's in-memory metric reader, and a meter over it."""
    from opentelemetry.sdk.metrics import MeterProvider
    from opentelemetry.sdk.metrics.export import InMemoryMetricReader

    sink = InMemoryMetricReader()
    sink.meter = MeterProvider(metric_readers=[sink]).get_meter("metta-test")
    return sink


def _points(reader):
    """Every recorded metric, by name, with its unit, sum and attributes."""
    recorded = {}
    for resource in reader.get_metrics_data().resource_metrics:
        for scope in resource.scope_metrics:
            for metric in scope.metrics:
                point = next(iter(metric.data.data_points))
                recorded[metric.name] = (metric.unit, point.sum, dict(point.attributes))
    return recorded


def test_a_trace_becomes_one_span_per_reduction(nested, exporter):
    """Three reductions, three spans, each named by the head that reduced.

    The name is the HEAD and not the whole term, which is OpenTelemetry's own
    low-cardinality rule for a span name; the term is an attribute, where high
    cardinality belongs.
    """
    recorded = nested.trace(S["tl-quad"](3))
    spans(recorded, tracer=exporter.tracer, space=nested.name)
    finished = exporter.get_finished_spans()
    assert [span.name for span in finished] == ["tl-double", "tl-double", "tl-quad"]
    outermost = finished[-1]
    assert outermost.attributes["metta.term"] == "(tl-quad 3)"
    assert outermost.attributes["metta.depth"] == 0
    assert outermost.attributes["metta._atoms.answer"] == "12"
    assert outermost.attributes["metta.space"] == nested.name


def test_spans_nest_by_the_events_own_depth(nested, exporter):
    """The two inner reductions are children of the outer one."""
    spans(nested.trace(S["tl-quad"](3)), tracer=exporter.tracer)
    finished = exporter.get_finished_spans()
    outer = next(span for span in finished if span.name == "tl-quad")
    inner = [span for span in finished if span.name == "tl-double"]
    assert outer.parent is None
    assert {span.parent.span_id for span in inner} == {outer.context.span_id}


def test_a_span_carries_the_time_the_engine_recorded(nested, exporter):
    """A back-dated span is the point: the times are the reductions', not now.

    The events carry wall nanoseconds since the run began, so a span's start is
    the run's origin plus that, which is why an exporter sees the shape of the
    reduction rather than the shape of the loop that built the spans.
    """
    recorded = nested.trace(S["tl-quad"](3))
    origin = 1_700_000_000_000_000_000
    spans(recorded, tracer=exporter.tracer, start_time=origin)
    finished = exporter.get_finished_spans()
    by_seq = {span.attributes["metta.seq"]: span for span in finished}
    open_at: dict[int, int] = {}
    for event in recorded:
        if event.kind == "call":
            open_at[event.depth] = event.seq
            assert by_seq[event.seq].start_time == origin + event.time
        else:
            opened = open_at.pop(event.depth)
            assert by_seq[opened].end_time == origin + event.time
    assert not open_at, "every reduction of this trace closed"


def test_a_failed_reduction_is_an_error_span(space, exporter):
    """A reduction that answered nothing is an ERROR span saying so.

    The `fail` port is the engine's own word for it, so the span carries
    `metta.exit=fail` rather than being silently absent from the tree.
    """
    from opentelemetry.trace import StatusCode

    space.run("(= (tl-picky 1) yes)")
    spans(space.trace(S["tl-picky"](2)), tracer=exporter.tracer)
    finished = exporter.get_finished_spans()
    failed = [span for span in finished if span.attributes.get("metta.exit") == "fail"]
    assert failed, [span.name for span in finished]
    assert failed[0].status.status_code is StatusCode.ERROR


def test_a_reduction_a_bound_cut_ends_with_the_trace(space, exporter):
    """A call with neither exit nor fail is ERROR and ends where the trace does.

    A recording bound leaves the outermost reductions open, and a span left open
    is a span nothing exports; ending them at the trace's own last moment is
    what makes the prefix readable.
    """
    from opentelemetry.trace import StatusCode

    space.run("(= (tl-deep $n) (if (== $n 0) 0 (tl-deep (- $n 1))))")
    recorded = space.trace(S["tl-deep"](6), max_events=3)
    assert recorded.truncated
    spans(recorded, tracer=exporter.tracer)
    finished = exporter.get_finished_spans()
    cut = [span for span in finished if span.attributes.get("metta.exit") == "absent"]
    assert cut, [dict(span.attributes) for span in finished]
    assert cut[0].status.status_code is StatusCode.ERROR
    assert cut[0].end_time == max(span.end_time for span in finished)


def test_an_observed_block_hangs_its_reductions_under_one_span(nested, exporter):
    """Every reduction of the block, from every call in it, under one span."""
    with observe(nested, tracer=exporter.tracer, name="workload") as recorded:
        nested.run("!(tl-quad 5)")
        nested.run("!(tl-double 1)")
    assert len(recorded) == 8
    finished = exporter.get_finished_spans()
    roots = [span for span in finished if span.parent is None]
    assert [span.name for span in roots] == ["workload"]
    assert sorted(span.name for span in finished) == [
        "tl-double", "tl-double", "tl-double", "tl-quad", "workload",
    ]


def test_an_observed_block_records_four_histograms(nested, reader):
    """The stats block's four counters, with the space and the workload on each."""
    with observe(nested, meter=reader.meter, name="workload"):
        nested.run("!(tl-quad 5)")
    recorded = _points(reader)
    assert set(recorded) == {
        "metta.inferences", "metta.cputime", "metta.gc.freed", "metta.table_bytes",
    }
    assert recorded["metta.inferences"][0] == "1"
    assert recorded["metta.inferences"][1] > 0
    assert recorded["metta.table_bytes"][0] == "By"
    assert recorded["metta.inferences"][2] == {
        "metta.space": nested.name,
        "metta.workload": "workload",
    }


def test_observing_with_neither_instrument_refuses(space):
    """There would be nothing to observe with, and the refusal names the door."""
    with pytest.raises(MettaError) as refusal, observe(space):
        pass
    assert "tracer, a meter, or both" in str(refusal.value)


def test_a_trace_inside_an_observed_block_refuses(nested, exporter):
    """One session holds the wrappers, and observe() is holding them."""
    with observe(nested, tracer=exporter.tracer), pytest.raises(Exception, match="nested"):
        nested.trace(S["tl-quad"](2))


def test_observing_inside_a_debug_session_refuses(nested, exporter):
    """The other direction, and the refusal names THIS door and its remedy.

    The engine's own sentence names the trace door, which a caller of observe()
    never went near, so it is re-raised naming the session rule and the way out.
    """
    with nested.debug(S["tl-quad"](3), on=[S["tl-double"]]) as session:
        for _stop in session:
            with pytest.raises(MettaError) as refusal:
                with observe(nested, tracer=exporter.tracer):
                    pass
            assert "only one at a time owns the wrappers" in str(refusal.value)
            break


def test_a_meter_alone_arms_nothing(nested, reader):
    """Without a tracer there is no session, so a trace inside still works."""
    with observe(nested, meter=reader.meter):
        recorded = nested.trace(S["tl-quad"](2))
    assert len(recorded) == 6
    assert _points(reader)["metta.inferences"][1] > 0


def test_a_raising_block_still_releases_the_session(nested, exporter):
    """A block that raises leaves the wrappers off, so the next one arms."""
    marker = "the block's own failure"
    with pytest.raises(RuntimeError, match=marker), observe(nested, tracer=exporter.tracer):
        nested.run("!(tl-double 1)")
        raise RuntimeError(marker)
    with observe(nested, tracer=exporter.tracer) as recorded:
        nested.run("!(tl-double 2)")
    assert len(recorded) == 2


def test_a_recording_bound_stops_the_recording_not_the_work(nested, exporter):
    """A telemetry budget may not become the observed program's error.

    A trace session's bound stops the run with the recording, which is what
    bounds a traced program's time. An observed block's work is the caller's, so
    the bound disarms the recorder and the work runs to the end.
    """
    with observe(nested, tracer=exporter.tracer, max_events=2) as recorded:
        answers = nested.run("!(tl-quad 5)")
    assert answers == [[20]]
    assert len(recorded) == 2
    assert recorded.stopped is not None
    assert recorded.truncated


def test_a_filter_selects_what_the_block_records(nested, exporter):
    """`filter` is m.trace's own selector, and means the same here."""
    with observe(nested, tracer=exporter.tracer, filter=[S["tl-double"]]) as recorded:
        nested.run("!(tl-quad 5)")
    assert {str(event.term.children[0]) for event in recorded} == {"tl-double"}


def test_the_doors_name_the_extra_when_opentelemetry_is_absent(nested, monkeypatch):
    """The refusal names the package and the extra, and says the API is enough."""
    monkeypatch.setattr("metta._lazy.import_module", _no_opentelemetry)
    with pytest.raises(ImportError) as refusal:
        spans(nested.trace(S["tl-double"](1)), tracer=None)
    assert "pymetta[otel]" in str(refusal.value)
    assert "the SDK" in str(refusal.value)


def _no_opentelemetry(name, *arguments, **options):
    """importlib.import_module with the OpenTelemetry API taken out."""
    import importlib

    if name.split(".")[0] == "opentelemetry":
        absent = "No module named 'opentelemetry'"
        raise ModuleNotFoundError(absent, name="opentelemetry")
    return importlib.import_module(name, *arguments, **options)
