"""Purpose: verify reusable benchmark setup, counter, and perf plumbing.
Guarantees:
  - steady-state warmup garbage is collected before perf opens; collection
    failure still releases the workload [tested:
    test_steady_workloads_collect_before_the_window,
    test_collection_failure_releases_the_workload; commit=8ca8a387fc61d0918484b19a1a3baf85b6523043]
  - checkout shape compares each declared dimension independently [tested:
    test_a_baseline_compares_checkout_length_and_depth; commit=8ca8a387fc61d0918484b19a1a3baf85b6523043]
  - a built chapter-19 handle extension makes the round-trip benchmark execute
    rather than skip [tested:
    test_handle_benchmark_reaches_the_built_chapter_19_library;
    commit=49cb09f7a208810c81ef4ca78b608ca85f32af96]
  - counted measurement refuses off Linux with the reason and spawns nothing:
    perf for what it measures, and the spawner on Windows for the process
    group it needs [tested: test_perf_refuses_off_linux_before_it_spawns,
    test_the_spawner_refuses_windows_before_it_spawns; commit=WORKTREE]
Open Obligations:
  To Do: None
  Hacks: None
  Future Enhancements: None.
"""  # noqa: D205  -- the scenario narrative is one continuous invariant, not summary-and-body prose

import gc
import json
import os
import shutil
import signal
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
from bench import CASES, _write_merged_json
from bench import main as benchmark_main
from benchmarks.check_instructions import _CASES as INSTRUCTION_CASES
from benchmarks.conftest import pytest_benchmark_update_machine_info
from benchmarks.engine_workloads import (
    alpha_unique_case,
    close_engine_case,
    digest_case,
    let_heavy,
    let_space,
    py_method_case,
    sort_atom_case,
    source_load_case,
    space_name_case,
)
from benchmarks.pure import _CASES as PERF_CASES
from benchmarks.pure import _acknowledge
from benchmarks.pure import main as perf_workload_main
from benchmarks.subscription import (
    close_subscription_case,
    subscription_dispatch_case,
)
from benchmarks.workloads import json_payload, json_wire, term_operators, wire_atom, wire_codec
from metta_benchmarking import (
    ESTIMATED_CYCLES,
    PERF_CONTROL_REFUSED,
    BenchmarkBaseline,
    MeasurementRefusedError,
    _run_cachegrind,
    _run_perf,
    _spawn_and_reap,
    _workload,
    benchmark_case,
    benchmark_counter_slope,
    count_atoms,
    estimated_cycles,
    measure_counters,
    measure_instructions,
    measure_simulated,
    measured_main,
    refusal_is_fatal,
)

from metta import S


class _Stats:
    def __init__(self, inferences):
        self.inferences = inferences

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        return False


class _Engine:
    def __init__(self, inferences):
        self.inferences = inferences

    def stats(self):
        return _Stats(self.inferences)


class _MutableStats:
    def __init__(self, state):
        self.state = state

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        return False

    @property
    def inferences(self):
        return self.state.inferences


class _Benchmark:
    def __init__(self):
        self.extra_info = {}
        self.stats = None

    def pedantic(
        self,
        target,
        *,
        setup,
        teardown,
        rounds,
        warmup_rounds,
    ):
        result = None
        for _ in range(rounds + warmup_rounds):
            args, kwargs = setup()
            result = target(*args, **kwargs)
            teardown(*args, **kwargs)
        self.stats = SimpleNamespace(stats=SimpleNamespace(min=1.0))
        return result


class _DisabledBenchmark(_Benchmark):
    def pedantic(self, target, *, setup, teardown, **_options):
        args, kwargs = setup()
        try:
            return target(*args, **kwargs)
        finally:
            teardown(*args, **kwargs)


def test_benchmark_case_uses_fresh_state(tmp_path):  # noqa: D103  -- pytest discovers or injects this callable; its descriptive name states the contract
    baseline = BenchmarkBaseline(tmp_path / "baseline.json", update=True)
    created = []
    reaped = []

    def setup():
        state = SimpleNamespace(serial=len(created), engine=_Engine(7))
        created.append(state)
        return state

    fixture = _Benchmark()
    benchmark_case(
        fixture,
        baseline,
        name="fresh",
        unit="items",
        operations=1,
        operation=lambda _state: 1,
        setup=setup,
        teardown=reaped.append,
        engine=lambda state: state.engine,
        rounds=2,
        warmup_rounds=1,
    )

    assert len(created) == 6
    assert reaped == created
    assert fixture.extra_info["inference_samples"] == [7, 7, 7]


def test_benchmark_case_runs_with_wall_timing_disabled(tmp_path):  # noqa: D103  -- pytest discovers or injects this callable; its descriptive name states the contract
    baseline = BenchmarkBaseline(tmp_path / "baseline.json", update=True)
    fixture = _DisabledBenchmark()

    assert (
        benchmark_case(
            fixture,
            baseline,
            name="counter-only",
            unit="items",
            operations=1,
            operation=lambda _state: 1,
            setup=lambda: SimpleNamespace(engine=_Engine(4)),
            teardown=lambda _state: None,
            engine=lambda state: state.engine,
        )
        == 1
    )
    assert "wall_seconds_per_operation" not in fixture.extra_info


def test_baseline_rejects_inference_movement_beyond_the_allowance(tmp_path):  # noqa: D103  -- pytest discovers or injects this callable; its descriptive name states the contract
    path = tmp_path / "baseline.json"
    updating = BenchmarkBaseline(path, update=True)
    updating.observe_counter("engine", unit="answers", operations=2, samples=[10, 10, 10])
    updating.observe_wall("engine", 0.25)
    updating.finish()

    baseline = BenchmarkBaseline(path)
    assert baseline.observe_counter("engine", unit="answers", operations=2, samples=[9, 9, 9]) == 9
    # A shift inside the absolute allowance is measurement artifact, not work:
    # the committed +2 join phantom does not scale with the workload and is
    # produced by predicates the benchmark never calls.
    assert (
        baseline.observe_counter("engine", unit="answers", operations=2, samples=[14, 14, 14])
        == 14
    )
    with pytest.raises(AssertionError, match="inference regression"):
        baseline.observe_counter("engine", unit="answers", operations=2, samples=[15, 15, 15])
    # The band is two-sided: a drop beyond the allowance is a stale pin, and
    # a stale-high pin masks real regressions up to its own margin, so it
    # fails until re-pinned with its mechanism recorded.
    assert baseline.observe_counter("engine", unit="answers", operations=2, samples=[6, 6, 6]) == 6
    with pytest.raises(AssertionError, match="improvement left unpinned"):
        baseline.observe_counter("engine", unit="answers", operations=2, samples=[5, 5, 5])
    # ONE low sample is not an improvement. min-of-n is the right statistic for
    # "did anything get slower" and the wrong one for "did this get faster", and
    # taking it for both directions made an anomalous reading claim a win: the C
    # seat's boot read [1484396, 1485362, 1485362] inside the gate against a
    # 1485363 pin while nine consecutive samples outside it read 1485362.
    assert (
        baseline.observe_counter("engine", unit="answers", operations=2, samples=[5, 14, 14])
        == 5
    )


def test_baseline_update_is_atomic_json(tmp_path):  # noqa: D103  -- pytest discovers or injects this callable; its descriptive name states the contract
    path = tmp_path / "baseline.json"
    baseline = BenchmarkBaseline(path, update=True)
    baseline.observe_counter("pure", unit="terms", operations=3, samples=None)
    baseline.observe_wall("pure", 0.5)
    baseline.finish()

    document = json.loads(path.read_text())
    assert document["benchmarks"]["pure"] == {
        "inferences": None,
        "operations": 3,
        "unit": "terms",
        "wall_seconds_per_operation": 0.5,
    }
    assert list(tmp_path.glob(".baseline.json.*")) == []


def test_benchmark_counter_slope_uses_fresh_state_and_gates_growth(tmp_path):  # noqa: D103  -- pytest discovers or injects this callable; its descriptive name states the contract
    path = tmp_path / "baseline.json"
    baseline = BenchmarkBaseline(path, update=True)
    baseline.observe_counter("growth", unit="rows", operations=8, samples=[30, 30, 30])
    created = []
    reaped = []

    def setup():
        state = SimpleNamespace(inferences=0)
        created.append(state)
        return state

    def operation(completed, inferences):
        def run(state):
            state.inferences = inferences
            return completed

        return run

    assert (
        benchmark_counter_slope(
            baseline,
            name="growth",
            unit="rows",
            small_operations=2,
            small_operation=operation(2, 11),
            large_operations=8,
            large_operation=operation(8, 35),
            setup=setup,
            teardown=reaped.append,
            engine=lambda state: SimpleNamespace(stats=lambda: _MutableStats(state)),
        )
        == 24
    )
    baseline.finish()

    assert len(created) == 6
    assert reaped == created
    assert json.loads(path.read_text())["benchmarks"]["growth"]["inference_slope"] == {
        "delta_inferences": 24,
        "large_operations": 8,
        "small_operations": 2,
    }

    comparison = BenchmarkBaseline(path)
    assert (
        comparison.observe_counter_slope(
            "growth",
            unit="rows",
            small_operations=2,
            large_operations=8,
            small_samples=[11, 11, 11],
            large_samples=[39, 39, 39],
        )
        == 28
    )
    with pytest.raises(AssertionError, match="inference slope regression"):
        comparison.observe_counter_slope(
            "growth",
            unit="rows",
            small_operations=2,
            large_operations=8,
            small_samples=[11, 11, 11],
            large_samples=[40, 40, 40],
        )
    # Two-sided for the same reason as the flat counter: a slope that fell
    # beyond the allowance is a stale pin masking growth regressions.
    assert (
        comparison.observe_counter_slope(
            "growth",
            unit="rows",
            small_operations=2,
            large_operations=8,
            small_samples=[11, 11, 11],
            large_samples=[31, 31, 31],
        )
        == 20
    )
    with pytest.raises(AssertionError, match="slope improvement left unpinned"):
        comparison.observe_counter_slope(
            "growth",
            unit="rows",
            small_operations=2,
            large_operations=8,
            small_samples=[11, 11, 11],
            large_samples=[30, 30, 30],
        )


def test_baseline_bands_instructions_on_both_sides(tmp_path):  # noqa: D103  -- pytest discovers or injects this callable; its descriptive name states the contract
    path = tmp_path / "baseline.json"
    updating = BenchmarkBaseline(path, update=True)
    updating.observe_counter("pure", unit="terms", operations=3, samples=None)
    updating.observe_wall("pure", 0.5)
    updating.observe_instructions("pure", [1000, 1000, 1000])
    updating.finish()

    baseline = BenchmarkBaseline(path)
    assert baseline.observe_instructions("pure", [1010, 1010, 1010]) == 1010
    assert baseline.observe_instructions("pure", [990, 990, 990]) == 990
    with pytest.raises(AssertionError, match="instruction regression"):
        baseline.observe_instructions("pure", [1011, 1011, 1011])
    with pytest.raises(AssertionError, match="instruction improvement left unpinned"):
        baseline.observe_instructions("pure", [989, 989, 989])


def test_baseline_stamps_and_verifies_counter_configuration(tmp_path):  # noqa: D103  -- pytest discovers or injects this callable; its descriptive name states the contract
    path = tmp_path / "baseline.json"
    updating = BenchmarkBaseline(path, update=True)
    updating.observe_counter("engine", unit="answers", operations=2, samples=[10, 10, 10])
    updating.observe_configuration({"c_reader": True})
    updating.finish()
    assert json.loads(path.read_text())["counter_configuration"] == {"c_reader": True}

    BenchmarkBaseline(path).observe_configuration({"c_reader": True})
    with pytest.raises(AssertionError, match="counter configuration drift"):
        BenchmarkBaseline(path).observe_configuration({"c_reader": False})


def test_baseline_without_configuration_stamp_refuses_counter_comparison(tmp_path):  # noqa: D103  -- pytest discovers or injects this callable; its descriptive name states the contract
    path = tmp_path / "baseline.json"
    updating = BenchmarkBaseline(path, update=True)
    updating.observe_counter("engine", unit="answers", operations=2, samples=[10, 10, 10])
    updating.finish()
    with pytest.raises(AssertionError, match="no counter_configuration stamp"):
        BenchmarkBaseline(path).observe_configuration({"c_reader": True})




def test_measure_instructions_parses_perf_csv(monkeypatch):  # noqa: D103  -- pytest discovers or injects this callable; its descriptive name states the contract
    calls = []

    def run(command, environment, *, controlled, timeout, events):
        calls.append((command, environment, controlled, timeout, events))
        return 0, "", "12345,,instructions:u,1000,100.00,,\n"

    monkeypatch.setattr("metta_benchmarking._run_perf", run)
    assert measure_instructions(["python", "work.py"]) == (12345, 12345, 12345)
    assert all(
        call[0] == ["python", "work.py"]
        and not call[2]
        and call[3] == 60.0
        and tuple(call[4]) == ("instructions:u",)
        for call in calls
    )


def test_measure_counters_reads_every_requested_event(monkeypatch):
    """One run counts several events, and its own stdout comes back with them.

    The event NAME is matched in perf's third CSV field rather than as a
    substring, because task-clock carries the unit `msec` where instructions:u
    carries none, so a run asking for both has two differently shaped rows to
    read [source: perf-stat(1), -x SEP].
    """
    asked = []

    def run(command, environment, *, controlled, timeout, events):
        asked.append((command, environment, controlled, timeout, tuple(events)))
        return 0, "inferences 4242\n", (
            "Events disabled\n"
            "700155618,,instructions:u,57673473,100.00,,\n"
            "56.42,msec,task-clock,57673473,100.00,,\n"
        )

    monkeypatch.setattr("metta_benchmarking._run_perf", run)
    runs = measure_counters(
        ["cases", "boot"], events=("instructions:u", "task-clock"), controlled=True
    )
    assert runs.events["instructions:u"] == (700155618, 700155618, 700155618)
    assert runs.events["task-clock"] == (56.42, 56.42, 56.42)
    assert runs.outputs == ("inferences 4242\n",) * 3
    assert [(call[0], call[2], call[4]) for call in asked] == [
        (["cases", "boot"], True, ("instructions:u", "task-clock"))
    ] * 3


def test_measure_counters_refuses_a_counter_perf_did_not_produce(monkeypatch):
    """`<not counted>` is refused rather than read as a zero that gates nothing."""

    def run(*_arguments, **_keywords):
        return 0, "", "<not counted>,,instructions:u,0,0.00,,\n"

    monkeypatch.setattr("metta_benchmarking._run_perf", run)
    # `<not counted>` is the box refusing rather than the workload answering, so
    # the refusal carries its own type and names the knob that decides it.
    with pytest.raises(MeasurementRefusedError, match="never armed"):
        measure_counters(["cases", "boot"])


def test_a_declared_policy_is_written_on_every_update(tmp_path):
    """The runner's source owns the prose, so a re-pin cannot revert it.

    A per-row noise band is measured and lives in the file; a policy sentence
    is authored and lives in the runner, because a document created by a seat
    whose counters are not the default ones would otherwise carry the default
    seat's rule and state the opposite of its own.
    """
    path = tmp_path / "baseline.json"
    declared = {"counter_policy": "instructions:u and CPU time, paired, decide"}
    first = BenchmarkBaseline(path, update=True, policies=declared)
    first.observe_counter("c-boot", unit="boots", operations=1, samples=[7, 7, 7])
    first.finish()
    assert json.loads(path.read_text())["counter_policy"] == declared["counter_policy"]

    stale = json.loads(path.read_text())
    stale["counter_policy"] = "inferences decide"
    path.write_text(json.dumps(stale))
    again = BenchmarkBaseline(path, update=True, policies=declared)
    again.observe_counter("c-boot", unit="boots", operations=1, samples=[7, 7, 7])
    again.finish()
    written = json.loads(path.read_text())
    assert written["counter_policy"] == declared["counter_policy"]
    # The default a seat did not override is still there beside the one it did.
    assert "instructions:u minimum of three" in written["instruction_policy"]


def test_an_estimated_cycle_pin_bands_on_both_sides(tmp_path):
    """Estimated cycles gate the same way instructions do, with their own band.

    Both directions fail: a costlier run is the regression, and a cheaper one
    is a stale pin, which is what a foreign boundary needs because the
    inference counter is blind there and cannot referee either direction.
    """
    path = tmp_path / "baseline.json"
    updating = BenchmarkBaseline(path, update=True)
    updating.observe_counter("c-step", unit="steps", operations=1, samples=None)
    updating.observe_measurement("c-step", ESTIMATED_CYCLES, [1000, 1004, 1009])
    updating.finish()

    stored = json.loads(path.read_text())["benchmarks"]["c-step"]
    assert stored["estimated_cycles"] == 1000
    assert stored["estimated_cycles_noise_percent"] == 1.0

    baseline = BenchmarkBaseline(path)
    assert baseline.observe_measurement("c-step", ESTIMATED_CYCLES, [1010, 1011, 1012]) == 1010
    with pytest.raises(AssertionError, match="estimated cycle regression"):
        baseline.observe_measurement("c-step", ESTIMATED_CYCLES, [1011, 1012, 1013])
    with pytest.raises(AssertionError, match="estimated cycle improvement left unpinned"):
        baseline.observe_measurement("c-step", ESTIMATED_CYCLES, [989, 990, 991])
    # A count is whole: a fractional sample is a caller's defect, not a reading.
    with pytest.raises(ValueError, match="invalid estimated cycle samples"):
        baseline.observe_measurement("c-step", ESTIMATED_CYCLES, [1000.5, 1001, 1002])


def test_estimated_cycles_prices_each_access_by_the_level_that_served_it():
    """One for a first-level hit, five for a last-level hit, thirty-five for memory.

    Twenty accesses that all hit cost twenty; one first-level miss the last
    level caught adds four; one that went on to memory adds thirty-four. The
    last case is a real windowed run of the C seat's term-out, whose summary
    line reads these nine counts.
    """
    hits = {"Ir": 10, "I1mr": 0, "ILmr": 0, "Dr": 5, "D1mr": 0, "DLmr": 0,
            "Dw": 5, "D1mw": 0, "DLmw": 0}
    assert estimated_cycles(hits) == 20
    assert estimated_cycles(hits | {"I1mr": 1}) == 24
    assert estimated_cycles(hits | {"D1mw": 1, "DLmw": 1}) == 54
    term_out = dict(zip(
        ("Ir", "I1mr", "ILmr", "Dr", "D1mr", "DLmr", "Dw", "D1mw", "DLmw"),
        (652218324, 5050396, 1491, 100821057, 2610687, 2217, 340254838, 2497564, 560),
        strict=True,
    ))
    assert estimated_cycles(term_out) == 1134056847


def _which(name, **_options):
    """shutil.which for a test that substitutes the whole act of running a tool.

    Every bare name answers as installed under /usr/bin, and a name with a
    directory part comes back as given, because shutil.which checks such a name
    where it points instead of searching PATH.
    """
    return name if os.sep in name else f"/usr/bin/{name}"


def _fake_cachegrind(monkeypatch, summary, *, exit_status=0, header=None):
    """Substitute the whole act of running valgrind, writing its output file."""
    runs = []
    monkeypatch.setattr("metta_benchmarking.shutil.which", _which)
    monkeypatch.setattr("metta_benchmarking.os.access", lambda _path, _mode: True)
    events = header or "Ir I1mr ILmr Dr D1mr DLmr Dw D1mw DLmw"

    def spawn(argv, _environment, **_options):
        runs.append(list(argv))
        target = next(word for word in argv if word.startswith("--cachegrind-out-file="))
        Path(target.split("=", 1)[1]).write_text(
            f"events: {events}\nfn=main\n1 2 3\nsummary: {summary}\n", encoding="utf-8"
        )
        return exit_status, "inferences 7\n", "==1== the log\n"

    monkeypatch.setattr("metta_benchmarking._spawn_and_reap", spawn)
    return runs


def test_measure_simulated_reads_every_run_and_its_output(monkeypatch):
    """Every run's nine counts, in run order, beside the output it printed.

    The simulated hierarchy is fixed on the command line rather than copied
    from the host, and a controlled run starts uninstrumented, so only the
    workload's own client requests open the window.
    """
    runs = _fake_cachegrind(monkeypatch, "10 1 0 5 0 0 5 0 0")
    measured = measure_simulated(["./cases", "term-out", "600", "--controlled"],
                                 controlled=True)
    assert measured.events["Ir"] == (10, 10, 10)
    assert estimated_cycles({event: counts[0] for event, counts in measured.events.items()}) == 24
    assert measured.outputs == ("inferences 7\n",) * 3
    assert len(runs) == 3
    for argv in runs:
        assert argv[:3] == ["/usr/bin/setarch", "-R", "/usr/bin/valgrind"]
        assert {"--tool=cachegrind", "--cache-sim=yes", "--I1=32768,8,64",
                "--D1=32768,8,64", "--LL=8388608,16,64", "--instr-at-start=no"} <= set(argv)
        assert argv[-4:] == ["./cases", "term-out", "600", "--controlled"]


def test_a_simulated_window_that_never_opened_is_refused(monkeypatch):
    """A workload that never asked for the window counted nothing, and says so.

    That is a build defect rather than a free workload or a busy box, so it is
    an ordinary RuntimeError naming the header, as is a run the simulation
    never reached, a summary without the cache events, and a failed workload.
    """
    _fake_cachegrind(monkeypatch, "0 0 0 0 0 0 0 0 0")
    with pytest.raises(RuntimeError, match="window never opened"):
        measure_simulated(["./cases", "term-out", "600", "--controlled"], controlled=True)
    _fake_cachegrind(monkeypatch, "10", header="Ir")
    with pytest.raises(RuntimeError, match="cache simulation did not run"):
        measure_simulated(["./cases", "boot", "1"])
    _fake_cachegrind(monkeypatch, "10 1 0 5 0 0 5 0 0", exit_status=1)
    with pytest.raises(RuntimeError, match="failed under cachegrind with exit 1"):
        measure_simulated(["./cases", "boot", "1"])
    monkeypatch.setattr("metta_benchmarking.shutil.which", lambda _name, **_options: None)
    with pytest.raises(FileNotFoundError, match="valgrind is required"):
        measure_simulated(["./cases", "boot", "1"])


def test_perf_timeout_kills_and_reaps_process_group(monkeypatch):  # noqa: D103  -- pytest discovers or injects this callable; its descriptive name states the contract
    waits = []
    killed = []

    monkeypatch.setattr("metta_benchmarking.os.posix_spawn", lambda *_args, **_kwargs: 42)
    # This test substitutes the whole act of running a process; the tools it
    # would have run are part of that, and requiring them installed would make
    # a timeout-and-reap test depend on the machine having perf.
    monkeypatch.setattr("metta_benchmarking.shutil.which", _which)
    monkeypatch.setattr("metta_benchmarking.os.access", lambda _path, _mode: True)

    def waitpid(process, options):
        waits.append((process, options))
        return (0, 0) if options == os.WNOHANG else (process, signal.SIGKILL)

    ticks = iter([0.0, 2.0])
    monkeypatch.setattr("metta_benchmarking.os.waitpid", waitpid)
    monkeypatch.setattr("metta_benchmarking.os.killpg", lambda *args: killed.append(args))
    monkeypatch.setattr("metta_benchmarking.time.monotonic", lambda: next(ticks))

    with pytest.raises(TimeoutError, match="1 second limit"):
        _run_perf(
            ["python"],
            {},
            controlled=False,
            timeout=1.0,
            events=("instructions:u",),
        )
    assert killed == [(42, signal.SIGKILL)]
    assert waits == [(42, os.WNOHANG), (42, 0)]


def _shadowing_workload(directory: Path) -> Path:
    """An executable named like one /usr/bin also holds, in a directory of its own."""
    directory.mkdir()
    workload = directory / "true"
    workload.write_text("#!/bin/sh\n", encoding="utf-8")
    workload.chmod(0o755)
    return workload


def _tools_present(monkeypatch):
    """perf, valgrind and setarch answer as installed; any other name resolves for real."""
    resolve = shutil.which
    monkeypatch.setattr(
        "metta_benchmarking.shutil.which",
        lambda name, path=None: f"/usr/bin/{name}" if name in {"perf", "valgrind"} else resolve(name, path=path),
    )
    monkeypatch.setattr("metta_benchmarking.os.access", lambda _path, _mode: True)


def test_a_counting_tool_starts_the_workload_the_measurement_path_names(monkeypatch, tmp_path):
    """The workload reaches perf ABSOLUTE, as the measurement PATH resolves it.

    perf puts its own directories ahead of the PATH it passes to the workload,
    so a bare `true` would start /usr/bin/true whatever the measurement
    environment's first directory holds.
    """
    workload = _shadowing_workload(tmp_path / "bin")
    _tools_present(monkeypatch)
    started = []

    def spawn(argv, _environment, **_keywords):
        started.append(list(argv))
        return 0, "", "1,,instructions:u,1,100.00,,\n"

    monkeypatch.setattr("metta_benchmarking._spawn_and_reap", spawn)
    _run_perf(["true", "--flag"], {"PATH": f"{workload.parent}{os.pathsep}/usr/bin"},
              controlled=False, timeout=1.0, events=("instructions:u",))
    assert started[0][started[0].index("--") + 1:] == [str(workload), "--flag"]
    with pytest.raises(FileNotFoundError, match="not on the measurement environment's PATH"):
        _run_perf(["no-such-workload"], {"PATH": str(workload.parent)},
                  controlled=False, timeout=1.0, events=("instructions:u",))


def test_a_workload_named_by_its_path_is_the_file_it_names(monkeypatch, tmp_path):
    """A driver started as ./cases is the one it names, whatever PATH holds.

    shutil.which checks a name with a directory part where it points and never
    searches PATH, so the measured workload is that file; a path naming no
    executable is refused before any tool starts, and the refusal says a file
    is missing rather than that PATH lacks it.
    """
    driver = tmp_path / "cases"
    driver.write_text("#!/bin/sh\n", encoding="utf-8")
    driver.chmod(0o755)
    monkeypatch.chdir(tmp_path)
    elsewhere = {"PATH": str(tmp_path / "empty")}
    assert _workload(["./cases", "boot", "1"], elsewhere) == ["./cases", "boot", "1"]
    with pytest.raises(FileNotFoundError, match=r"^\./absent names no executable file$"):
        _workload(["./absent"], elsewhere)
    with pytest.raises(FileNotFoundError, match=r"^cases is not on the measurement environment's PATH$"):
        _workload(["cases"], elsewhere)


def test_cachegrind_starts_the_workload_the_measurement_path_names(monkeypatch, tmp_path):
    """Cachegrind is handed the same resolved workload, one rule for every tool."""
    workload = _shadowing_workload(tmp_path / "bin")
    _tools_present(monkeypatch)
    started = []

    def spawn(argv, _environment, **_keywords):
        started.append(list(argv))
        out = next(word.split("=", 1)[1] for word in argv if word.startswith("--cachegrind-out-file="))
        Path(out).write_text("events: Ir I1mr ILmr Dr D1mr DLmr Dw D1mw DLmw\n"
                             "summary: 9 1 1 3 1 1 2 1 1\n", encoding="utf-8")
        return 0, "", ""

    monkeypatch.setattr("metta_benchmarking._spawn_and_reap", spawn)
    _run_cachegrind("/usr/bin/valgrind", ["true", "--flag"],
                    {"PATH": f"{workload.parent}{os.pathsep}/usr/bin"}, controlled=False, timeout=1.0)
    assert started[0][-2:] == [str(workload), "--flag"]


def test_perf_acknowledgement_accepts_the_native_nul_terminator():  # noqa: D103  -- pytest discovers or injects this callable; its descriptive name states the contract
    reader, writer = os.pipe()
    try:
        os.write(writer, b"ack\n\0")
        _acknowledge(reader)
    finally:
        os.close(reader)
        os.close(writer)


def test_perf_workload_setup_and_teardown_stay_outside_control(monkeypatch):  # noqa: D103  -- pytest discovers or injects this callable; its descriptive name states the contract
    events = []

    def factory():
        events.append("setup")

        def operation():
            events.append("operation")
            return 1

        return operation, lambda: events.append("teardown")

    def controlled(operation):
        events.append("enable")
        result = operation()
        events.append("disable")
        return result

    monkeypatch.setitem(PERF_CASES, "probe", factory)
    monkeypatch.setattr("benchmarks.pure._controlled", controlled)

    assert perf_workload_main(["probe", "--controlled"]) == 0
    assert events == ["setup", "enable", "operation", "disable", "teardown"]


def test_the_controlled_window_holds_the_cyclic_collector_off(monkeypatch):
    """No collection pass can land inside the window, and the collector comes back.

    A pass fires on allocation thresholds, so whether one lands in the window
    depends on the whole process's allocation history rather than on the
    operation: that flipped save-load-metta by 8.9 percent on a never-called
    block of code. The window runs with the collector off and restores exactly
    the state it found, and it does not collect first: a pass before the window
    rearranges the free lists the operation allocates from, which moved a
    Python-only workload by 1.8 percent.
    """
    from benchmarks.pure import _controlled

    control_read, control_write = os.pipe()
    ack_read, ack_write = os.pipe()
    spare_read, spare_write = os.pipe()
    # perf answers each command once, so the enable's ack is ready before the
    # window opens and the disable's is written from inside it.
    os.write(ack_write, b"ack\n")
    monkeypatch.setenv("METTA_PERF_CONTROL_FD", str(control_write))
    monkeypatch.setenv("METTA_PERF_ACK_FD", str(ack_read))
    monkeypatch.setenv("METTA_PERF_CLOSE_FDS", str(spare_read))
    seen = {}

    def operation():
        seen["enabled"] = gc.isenabled()
        seen["collections"] = [entry["collections"] for entry in gc.get_stats()]
        os.write(ack_write, b"ack\n")
        return 1

    was_enabled = gc.isenabled()
    before = [entry["collections"] for entry in gc.get_stats()]
    try:
        assert _controlled(operation) == 1
        assert os.read(control_read, 64) == b"enable\ndisable\n"
    finally:
        for descriptor in (control_read, control_write, ack_read, ack_write, spare_write):
            os.close(descriptor)
    assert seen["enabled"] is False
    assert seen["collections"] == before
    assert gc.isenabled() is was_enabled


def test_the_controlled_window_holds_the_prolog_gc_thread_off(monkeypatch):
    """SWI's gc thread is stopped for the window and started again after it.

    Its work finished inside the window or after it by thread schedule alone,
    so save-load-metta's samples spread 1.5 percent with it and 0.011 percent
    without it. A workload that never booted SWI is left alone, since
    importing janus_swi to ask would boot it.
    """
    from benchmarks.pure import _controlled

    events = []
    prolog = ModuleType("janus_swi")
    prolog.query_once = lambda _goal: {"Mode": "true"}
    prolog.cmd = lambda _module, predicate, mode: events.append((predicate, mode))
    control_read, control_write = os.pipe()
    ack_read, ack_write = os.pipe()
    spare_read, spare_write = os.pipe()
    os.write(ack_write, b"ack\n")
    monkeypatch.setenv("METTA_PERF_CONTROL_FD", str(control_write))
    monkeypatch.setenv("METTA_PERF_ACK_FD", str(ack_read))
    monkeypatch.setenv("METTA_PERF_CLOSE_FDS", str(spare_read))
    monkeypatch.setitem(sys.modules, "janus_swi", prolog)

    def operation():
        events.append("operation")
        os.write(ack_write, b"ack\n")
        return 1

    try:
        assert _controlled(operation) == 1
    finally:
        for descriptor in (control_read, control_write, ack_read, ack_write, spare_write):
            os.close(descriptor)
    assert events == [
        ("set_prolog_gc_thread", "false"),
        "operation",
        ("set_prolog_gc_thread", "true"),
    ]


def test_perf_workload_teardown_runs_after_failure(monkeypatch):  # noqa: D103  -- pytest discovers or injects this callable; its descriptive name states the contract
    events = []

    def factory():
        def operation():
            events.append("operation")
            msg = "workload failed"
            raise LookupError(msg)

        return operation, lambda: events.append("teardown")

    monkeypatch.setitem(PERF_CASES, "failing-probe", factory)

    with pytest.raises(LookupError, match="workload failed"):
        perf_workload_main(["failing-probe"])
    assert events == ["operation", "teardown"]


@pytest.mark.parametrize('case', ['alpha-unique', 'subscription-dispatch', 'cold-probe'])
def test_steady_workloads_collect_before_the_window(case, monkeypatch):
    """Warmup garbage belongs to setup; a cold workload keeps its first call."""
    events = []

    def operation():
        events.append('operation')
        return 1

    def collect(module, predicate):
        assert (module, predicate) == ('system', 'garbage_collect')
        events.append('collect')

    def window(goal):
        events.append('enable')
        result = goal()
        events.append('disable')
        return result

    monkeypatch.setitem(PERF_CASES, case, lambda: (operation, lambda: events.append('close')))
    monkeypatch.setattr('janus_swi.cmd', collect)
    monkeypatch.setattr('benchmarks.pure._controlled', window)
    assert perf_workload_main([case, '--controlled']) == 0
    setup = [] if case == 'cold-probe' else ['operation', 'collect']
    assert events == [*setup, 'enable', 'operation', 'disable', 'close']


def test_collection_failure_releases_the_workload(monkeypatch):
    """A failed collection cannot open the window or abandon the workload."""
    events = []

    def collect(_module, _predicate):
        events.append('collect')
        msg = 'collection failed'
        raise RuntimeError(msg)

    monkeypatch.setitem(PERF_CASES, 'alpha-unique',
                        lambda: (lambda: 1, lambda: events.append('close')))
    monkeypatch.setattr('janus_swi.cmd', collect)
    monkeypatch.setattr('benchmarks.pure._controlled', lambda _goal: events.append('enable'))
    with pytest.raises(RuntimeError, match='collection failed'):
        perf_workload_main(['alpha-unique', '--controlled'])
    assert events == ['collect', 'close']


def test_count_atoms_derives_the_wire_workload_size():  # noqa: D103  -- pytest discovers or injects this callable; its descriptive name states the contract
    atom = S.deep(*(S.node(i, float(i), S.leaf) for i in range(50)))
    assert count_atoms(atom) == 252


def test_pure_workload_counts_are_derived():  # noqa: D103  -- pytest discovers or injects this callable; its descriptive name states the contract
    atom = wire_atom()
    assert wire_codec(atom, trips=2) == 2 * count_atoms(atom)
    assert json_wire(json_payload(), trips=2) == 2
    assert term_operators(terms=3) == 3


def test_instruction_inventory_covers_primitive_heavy_engine_paths():  # noqa: D103  -- pytest discovers or injects this callable; its descriptive name states the contract
    engine_cases = {
        "alpha-unique",
        "let-heavy",
        "py-method-call",
        "sort-atom",
        "source-load",
        "space-digest",
        "space-name",
    }
    assert engine_cases <= PERF_CASES.keys()
    assert set(INSTRUCTION_CASES) == PERF_CASES.keys()


def test_handle_benchmark_reaches_the_built_chapter_19_library():
    """A built fixture is a benchmark capability, not grounds for a skip."""
    from benchmarks.test_benchmarks import _handle_space

    try:
        space = _handle_space()
    except pytest.skip.Exception as skipped:
        pytest.fail(f"the built chapter-19 handle fixture was skipped: {skipped}")
    try:
        assert space.run("!(vector-new 1)")
    finally:
        space.drop()


@pytest.mark.parametrize(
    ("factory", "operations"),
    [
        (alpha_unique_case, 20),
        (digest_case, 20),
        (py_method_case, 3),
        (sort_atom_case, 20),
        (source_load_case, 5),
        (space_name_case, 3),
    ],
)
def test_primitive_workloads_check_public_results(factory, operations):  # noqa: D103  -- pytest discovers or injects this callable; its descriptive name states the contract
    state = factory(operations)
    try:
        assert state[1]() == operations
    finally:
        close_engine_case(state)


def test_let_workload_checks_its_bignum_result():  # noqa: D103  -- pytest discovers or injects this callable; its descriptive name states the contract
    space = let_space()
    try:
        assert let_heavy(space, 10) == 10
    finally:
        # The workload raises max-stack-depth in its setup, and a pragma is
        # ONE engine-wide setting rather than a property of this space. Each
        # bench.py case owns a process, so nothing there has to undo it; a
        # test shares its process with every other test, so this one does.
        space.run("!(pragma! max-stack-depth none)")
        space.drop()


def test_benchmark_cli_lists_and_rejects_case_names(capsys):  # noqa: D103  -- pytest discovers or injects this callable; its descriptive name states the contract
    assert benchmark_main(["--list"]) == 0
    assert capsys.readouterr().out.splitlines() == sorted(CASES)
    with pytest.raises(SystemExit) as stopped:
        benchmark_main(["misspelled"])
    assert stopped.value.code == 2


def test_benchmark_cli_spawns_each_case(monkeypatch):  # noqa: D103  -- pytest discovers or injects this callable; its descriptive name states the contract
    processes = []

    class Process:
        exitcode = 0

        def __init__(self, **options):
            self.options = options
            self.joined = []
            processes.append(self)

        def start(self):
            return None

        def join(self, timeout=None):
            self.joined.append(timeout)

        def is_alive(self):
            return False

    context = SimpleNamespace(Process=Process)
    monkeypatch.setattr("bench.multiprocessing.get_context", lambda _method: context)

    assert benchmark_main(["add-batch", "add-single", "--counter-only"]) == 0
    assert [process.options["name"] for process in processes] == [
        "metta-benchmark-add-batch",
        "metta-benchmark-add-single",
    ]
    assert all(process.joined == [120.0] for process in processes)


def test_benchmark_json_merge_is_atomic(tmp_path):  # noqa: D103  -- pytest discovers or injects this callable; its descriptive name states the contract
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    target = tmp_path / "merged.json"
    metadata = {"machine_info": {"cpu": "fixed"}, "commit_info": {"id": "abc"}}
    first.write_text(json.dumps({"benchmarks": [{"name": "first"}], "schema": 1} | metadata))
    second.write_text(json.dumps({"benchmarks": [{"name": "second"}], "schema": 1} | metadata))

    _write_merged_json([first, second], target)

    assert [item["name"] for item in json.loads(target.read_text())["benchmarks"]] == [
        "first",
        "second",
    ]
    assert list(tmp_path.glob(".merged.json.*")) == []


def test_benchmark_json_merge_preserves_unselected_cases(tmp_path):  # noqa: D103  -- pytest discovers or injects this callable; its descriptive name states the contract
    selected = tmp_path / "selected.json"
    target = tmp_path / "baseline.json"
    selected.write_text(
        json.dumps(
            {
                "benchmarks": [{"name": "selected", "stats": {"min": 1}}],
                "machine_info": {"cpu": "current"},
                "commit_info": {"id": "new"},
            }
        )
    )
    target.write_text(
        json.dumps(
            {
                "benchmarks": [
                    {"name": "selected", "stats": {"min": 9}},
                    {"name": "untouched", "stats": {"min": 2}},
                ],
                "machine_info": {"cpu": "old"},
                "commit_info": {"id": "old"},
            }
        )
    )

    _write_merged_json([selected], target)

    document = json.loads(target.read_text())
    assert document["benchmarks"] == [
        {"name": "selected", "stats": {"min": 1}},
        {"name": "untouched", "stats": {"min": 2}},
    ]
    assert document["machine_info"] == {"cpu": "current"}
    assert document["commit_info"] == {"id": "new"}


def test_benchmark_machine_info_is_stable():  # noqa: D103  -- pytest discovers or injects this callable; its descriptive name states the contract
    machine_info = {
        "cpu": {
            "brand_raw": "processor",
            "hz_actual": [5_000_000_000, 0],
            "hz_actual_friendly": "5.0 GHz",
            "hz_advertised": [5_000_000_000, 0],
            "hz_advertised_friendly": "5.0 GHz",
        }
    }
    pytest_benchmark_update_machine_info(None, machine_info)
    assert machine_info == {"cpu": {"brand_raw": "processor"}}


def test_subscription_dispatch_case_measures_writes_only():
    """Building the subscriptions is setup, not the measurement.

    A thousand standing queries cost more to CREATE than two hundred writes
    cost to dispatch, so a window that held the construction would report
    the construction. This pins that the operation the case hands back does
    the writes and nothing else, and that the writes really are dispatched:
    one of the thousand matches each, so a run delivering nothing would be
    measuring an empty loop.
    """
    state = subscription_dispatch_case(subscriptions=20, writes=5)
    space, standing, delivered, run = state
    try:
        assert len(standing) == 20
        assert delivered[0] == 0, "construction delivered before the window"
        assert run() == 5
        assert delivered[0] == 5, "the measured writes dispatched nothing"
        assert len(space) == 5
    finally:
        close_subscription_case(state)
    assert all(not subscription._active for subscription in standing)


def test_the_benchmark_suite_prices_a_file_load():
    """P0.18: loader regressions used to land in no counter case, because
    `source-load` runs a string through the engine and the save-load pair
    prices a round trip whose baselines had drifted high. The `file-load`
    bench replace-loads a 20,001-atom file every round (withdrawal plus
    re-add plus content digest, the loader's whole path) and follows with
    an unchanged `import!`, the skip branch. This test pins the wiring:
    the registry row, the runner function, and a live integer baseline.
    """  # noqa: D205  -- the scenario narrative is one continuous invariant, not summary-and-body prose
    import json

    from _workspace import SEAT

    registry = (SEAT / "bench.py").read_text()
    assert '"file-load": "test_file_load"' in registry
    suite = (SEAT / "benchmarks" / "test_benchmarks.py").read_text()
    assert "def test_file_load(" in suite
    data = json.loads(
        (SEAT / "benchmarks" / "baseline.json").read_text()
    )
    entry = data["benchmarks"]["file-load"]
    assert isinstance(entry["inferences"], int) and entry["inferences"] > 0


def test_the_json_wire_row_is_not_registered_engine_free():
    """The JSON codec crosses into the engine, so its row carries inferences.

    metta/_json.py IS the engine's codec: dumps and loads each reach
    engine/json_codec.pl through janus. The bench registered ``engine=None``
    anyway, which pinned the row at ``"inferences": null`` and made
    ``_compare_counter`` require it to STAY null, so the heaviest crossing in
    the roster was gated on retired instructions alone. This requires the
    engine to have been charged PER TRIP, then requires the shipped row to
    carry the integer pin. That pin is what defends the wiring: with it in
    place, registering the row engine-free again raises "json-wire is
    engine-free but its baseline has inferences" instead of passing green.

    The property is the SCALING, not the magnitude. A floor on one trip's
    count was the Prolog codec's own size and went red the day the C codec
    landed, which moved a round trip from 84,725 inferences to 72 without
    moving it off the engine at all. An engine-free codec charges a constant,
    so the discriminating question is whether ten trips cost about ten times
    one, and both configurations answer it the same way: 725 against 77 with
    engine/json_codec.so, 847,259 against 84,729 under METTA_C_JSON=off
    [measured 2026-08-28].
    """
    import json

    from metta import MeTTa

    space = MeTTa().space()
    try:
        # The first trip in a process pays a one-time 75 inferences the rest
        # do not, which is enough to sink a ratio taken against it.
        json_wire(json_payload(), trips=1)
        with space.stats() as once:
            assert json_wire(json_payload(), trips=1) == 1
        with space.stats() as ten_times:
            assert json_wire(json_payload(), trips=10) == 10
    finally:
        space.drop()
    assert once.inferences > 0, "the JSON codec charged the engine nothing"
    assert ten_times.inferences > once.inferences * 5, (
        "the JSON codec's charge does not scale with trips, so it is not the "
        "engine doing the work"
    )

    from _workspace import SEAT

    assert '"json-wire": "test_json_wire"' in (SEAT / "bench.py").read_text()
    entry = json.loads(
        (SEAT / "benchmarks" / "baseline.json").read_text()
    )["benchmarks"]["json-wire"]
    assert isinstance(entry["inferences"], int) and entry["inferences"] > 0


def test_check_instructions_reports_every_failing_case(tmp_path):
    """A regression in one case never hides another.

    The runner's old loop let ``observe_instructions`` raise on the first
    regressing case, so four stale pins and one real overrun sat hidden
    behind whichever red came first, on every tree, for days. This drives
    ``observe_all`` with a fake sampler over one passing and two failing
    cases and requires all three to have been measured, both failures
    reported, and the passing case's observation unharmed.
    """
    import json

    from benchmarks.check_instructions import observe_all
    from metta_benchmarking import BenchmarkBaseline

    from _workspace import SEAT

    real = json.loads(
        (SEAT / "benchmarks" / "baseline.json").read_text()
    )
    document = {
        key: value for key, value in real.items() if key != "benchmarks"
    }
    document["benchmarks"] = {
        name: {
            "inferences": 1,
            "instructions": 1_000,
            "instruction_noise_percent": 1.0,
            "operations": 1,
            "unit": "calls",
            "wall_seconds_per_operation": 1.0,
        }
        for name in ("alpha", "beta", "gamma")
    }
    path = tmp_path / "baseline.json"
    path.write_text(json.dumps(document))

    baseline = BenchmarkBaseline(path, update=False)
    measured: list[str] = []

    def sampler(name: str) -> list[int]:
        measured.append(name)
        return [5_000, 5_001, 5_002] if name != "beta" else [1_000, 1_001, 1_002]

    failures = observe_all(baseline, ["alpha", "beta", "gamma"], sampler)

    assert measured == ["alpha", "beta", "gamma"]
    assert len(failures) == 2
    assert "alpha" in failures[0]
    assert "gamma" in failures[1]


def test_a_declared_instruction_band_survives_a_re_pin(tmp_path):
    """Re-pinning re-measures the count and leaves the declared band standing.

    A row's instruction count is measured; the noise percent beside it is
    declared by hand with the measurement that justified it. Every
    ``--update`` wrote the 1.0 default back over both declarations in
    baseline.json: typed-call's 5.0, raised for a code-layout swing measured
    at 3.13%, and json-wire's 2.5, widened for one measured at 1.56%. Each
    lane was left gated inside its own noise, where it goes red for layout
    and is then re-pinned past a real regression. This drives the re-pin
    route the runner uses, ``observe_all`` against an update-mode baseline,
    with a second row that declares nothing as the control.
    """
    from benchmarks.check_instructions import observe_all

    path = tmp_path / "baseline.json"
    updating = BenchmarkBaseline(path, update=True)
    for name in ("layout-sensitive", "ordinary"):
        updating.observe_counter(name, unit="calls", operations=1, samples=None)
        updating.observe_instructions(name, [1_000, 1_000, 1_000])
    updating.finish()

    document = json.loads(path.read_text())
    assert document["benchmarks"]["ordinary"]["instruction_noise_percent"] == 1.0
    document["benchmarks"]["layout-sensitive"]["instruction_noise_percent"] = 5.0
    path.write_text(json.dumps(document))

    repinning = BenchmarkBaseline(path, update=True)
    assert (
        observe_all(
            repinning,
            ["layout-sensitive", "ordinary"],
            lambda _name: [2_000, 2_000, 2_000],
        )
        == []
    )
    repinning.finish()

    rows = json.loads(path.read_text())["benchmarks"]
    assert rows["layout-sensitive"]["instructions"] == 2_000
    assert rows["layout-sensitive"]["instruction_noise_percent"] == 5.0
    assert rows["ordinary"]["instructions"] == 2_000
    assert rows["ordinary"]["instruction_noise_percent"] == 1.0

    # The declaration reaches the gate and not only the file: 4% over the
    # fresh pin passes on the widened row and fails on the row that kept 1.0.
    comparing = BenchmarkBaseline(path)
    assert comparing.observe_instructions("layout-sensitive", [2_080] * 3) == 2_080
    with pytest.raises(AssertionError, match="instruction regression"):
        comparing.observe_instructions("ordinary", [2_080] * 3)


def test_baseline_remove_case_is_update_only(tmp_path):  # noqa: D103  -- pytest discovers or injects this callable; its descriptive name states the contract
    path = tmp_path / "baseline.json"
    updating = BenchmarkBaseline(path, update=True)
    updating.observe_counter("engine", unit="answers", operations=2, samples=[10, 10, 10])
    updating.observe_counter("stale", unit="answers", operations=2, samples=[10, 10, 10])
    updating.remove_case("stale")
    with pytest.raises(KeyError):
        updating.remove_case("missing")
    updating.finish()
    assert list(json.loads(path.read_text())["benchmarks"]) == ["engine"]
    with pytest.raises(AssertionError, match="outside update mode"):
        BenchmarkBaseline(path).remove_case("engine")


def test_a_subset_updater_verifies_without_restamping(tmp_path):  # noqa: D103  -- pytest discovers or injects this callable; its descriptive name states the contract
    path = tmp_path / "baseline.json"
    updating = BenchmarkBaseline(path, update=True)
    updating.observe_counter("engine", unit="answers", operations=2, samples=[10, 10, 10])
    updating.observe_configuration({"c_reader": True})
    updating.finish()

    subset = BenchmarkBaseline(path, update=True)
    with pytest.raises(AssertionError, match="counter configuration drift"):
        subset.observe_configuration({"c_reader": False}, stamp=False)
    subset.observe_configuration({"c_reader": True}, stamp=False)

    # An absent stamp is left for the owning full-battery updater rather
    # than written by a subset run that measured only part of the document.
    bare = tmp_path / "bare.json"
    first = BenchmarkBaseline(bare, update=True)
    first.observe_counter("engine", unit="answers", operations=2, samples=[10, 10, 10])
    first.finish()
    second = BenchmarkBaseline(bare, update=True)
    second.observe_configuration({"c_reader": True}, stamp=False)
    second.finish()
    assert "counter_configuration" not in json.loads(bare.read_text())


def test_cpu_seconds_are_recorded_and_never_compared(tmp_path):
    """CPU time is written beside a row's pins and never refereed.

    A time reading prices the queue as well as the work, and the box these
    gates run on is never quiet, so the row's verdict belongs to its counts:
    instructions:u, and estimated cycles where the row crosses a foreign
    boundary.
    """
    path = tmp_path / "baseline.json"
    updating = BenchmarkBaseline(path, update=True)
    updating.observe_counter("crossing", unit="calls", operations=1, samples=None)
    updating.observe_cpu("crossing", 0.400)
    updating.finish()

    assert json.loads(path.read_text())["benchmarks"]["crossing"][
        "cpu_seconds_per_operation"
    ] == 0.400

    # Twice the pinned time, and four times it, both accepted in silence: this
    # door has no opinion, which is exactly what distinguishes it.
    baseline = BenchmarkBaseline(path)
    assert baseline.observe_cpu("crossing", 0.800) is None
    assert baseline.observe_cpu("crossing", 1.600) is None

    # What it does refuse is an impossible reading and an unknown case, because
    # those are defects in the caller rather than movements in the workload.
    with pytest.raises(ValueError, match="CPU time must be positive"):
        baseline.observe_cpu("crossing", 0.0)
    with pytest.raises(KeyError, match="no counter observation"):
        baseline.observe_cpu("never-measured", 0.400)


def test_a_refused_window_is_told_apart_from_a_workload_that_failed(monkeypatch):
    """The box refusing to count and the tree answering wrongly are two answers.

    A controlled workload that never got perf's acknowledgement exits
    PERF_CONTROL_REFUSED, and a run whose counter never armed reports
    `<not counted>` where a number belongs. Both mean no measurement was
    taken, so both raise MeasurementRefusedError. Every OTHER nonzero exit is the
    workload's own failure and stays an ordinary RuntimeError, which is what
    keeps a real regression from reading as contention and passing.
    """
    def refused(*_command, **_perf):
        return PERF_CONTROL_REFUSED, "", "Events disabled\nworkload: perf did not acknowledge\n"

    monkeypatch.setattr("metta_benchmarking._run_perf", refused)
    with pytest.raises(MeasurementRefusedError, match="never opened"):
        measure_instructions(["cases", "boot"], controlled=True)

    def uncounted(*_command, **_perf):
        return 0, "", "<not counted>,,instructions:u,0,100.00,,\n"

    monkeypatch.setattr("metta_benchmarking._run_perf", uncounted)
    with pytest.raises(MeasurementRefusedError, match="never armed"):
        measure_instructions(["cases", "boot"])

    def broken(*_command, **_perf):
        return 3, "", "workload: the operation answered 0, expected 2000\n"

    monkeypatch.setattr("metta_benchmarking._run_perf", broken)
    with pytest.raises(RuntimeError, match="perf stat failed with exit 3") as failure:
        measure_instructions(["cases", "boot"])
    assert not isinstance(failure.value, MeasurementRefusedError)


def test_a_benchmark_lane_skips_a_refusal_locally_and_refuses_it_in_ci(monkeypatch, capsys):
    """One policy for every benchmark entry point, and it depends on CI alone.

    A developer's box is shared, so a PMU another session holds is a note and
    a SKIP; a CI runner that cannot count is a broken runner, so there the same
    refusal is an error and an exit 1. An ordinary answer passes through either
    way.

    The local skip is PERF_CONTROL_REFUSED and not 0, because check.sh reads
    that number as `skipped` and 0 as `ok`, and a lane that compared nothing
    must not read the same as one that compared every row: mork-bench reported
    `ok` on four of five full gate runs while another session held the PMU.
    """
    def refuses() -> int:
        msg = "the measured window never opened"
        raise MeasurementRefusedError(msg)

    monkeypatch.delenv("CI", raising=False)
    assert measured_main(refuses) == PERF_CONTROL_REFUSED
    assert PERF_CONTROL_REFUSED != 0, "a skip that exits 0 reads as ok"
    local = capsys.readouterr()
    assert "note: the box refused the measurement" in local.out
    assert "never opened" in local.out

    monkeypatch.setenv("CI", "true")
    assert measured_main(refuses) == 1
    continuous = capsys.readouterr()
    assert "error: this benchmark lane measured nothing" in continuous.err
    assert "never opened" in continuous.err

    assert measured_main(lambda: 0) == 0
    assert measured_main(lambda: 1) == 1


def test_a_row_may_declare_a_wider_inference_allowance_than_the_default(tmp_path):
    """The instruction side has taken a per-row band since it was written.

    The inference side did not, so a row whose count MEASURABLY moves more than
    four with nothing but the state of the checkout reported a regression
    forever. The C seat's boot is that row: it reads about twenty-five higher
    in a tree whose tracked files have been written over, deterministically,
    with no source difference behind it. A declaration widens the allowance on
    BOTH sides, and a row that declares nothing keeps the four.
    """
    document = {
        "schema": 1,
        "benchmarks": {
            "declared": {
                "unit": "boots",
                "operations": 1,
                "inferences": 1000,
                "inference_allowance": 32,
            },
            "default": {"unit": "boots", "operations": 1, "inferences": 1000},
        },
    }
    path = tmp_path / "baseline.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    baseline = BenchmarkBaseline(path)

    # Inside the declaration and outside the default, on the regression side.
    baseline.observe_counter("declared", unit="boots", operations=1, samples=[1025, 1025, 1025])
    with pytest.raises(AssertionError, match="plus the 4 inference allowance"):
        baseline.observe_counter("default", unit="boots", operations=1, samples=[1025, 1025, 1025])

    # And on the improvement side, which is the half a stale-high pin hides in.
    baseline.observe_counter("declared", unit="boots", operations=1, samples=[975, 975, 975])
    with pytest.raises(AssertionError, match="minus the 4 inference allowance"):
        baseline.observe_counter("default", unit="boots", operations=1, samples=[975, 975, 975])

    # A move past the declaration still fails, which is what keeps it a gate.
    with pytest.raises(AssertionError, match="plus the 32 inference allowance"):
        baseline.observe_counter("declared", unit="boots", operations=1, samples=[1040, 1040, 1040])


def test_a_declared_inference_allowance_survives_a_re_pin(tmp_path):
    """Re-pinning re-measures the count; it does not re-decide the row's noise."""
    path = tmp_path / "baseline.json"
    path.write_text(
        json.dumps(
            {
                "schema": 1,
                "benchmarks": {
                    "boot": {
                        "unit": "boots",
                        "operations": 1,
                        "inferences": 1000,
                        "inference_allowance": 32,
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    baseline = BenchmarkBaseline(path, update=True)
    baseline.observe_counter("boot", unit="boots", operations=1, samples=[1100, 1100, 1100])
    baseline.finish()
    written = json.loads(path.read_text(encoding="utf-8"))["benchmarks"]["boot"]
    assert written["inferences"] == 1100
    assert written["inference_allowance"] == 32


def test_one_line_decides_whether_a_refusal_is_also_red(monkeypatch):
    """Three lanes draw it, so it is defined once.

    A runner that cannot measure is a broken runner and a lane that passes
    without measuring is worse than a red one; a developer's box is shared and
    a contended PMU is not a code change.
    """
    monkeypatch.delenv("CI", raising=False)
    assert refusal_is_fatal() is False
    monkeypatch.setenv("CI", "false")
    assert refusal_is_fatal() is False
    monkeypatch.setenv("CI", "true")
    assert refusal_is_fatal() is True


def test_a_baseline_compares_checkout_length_and_depth(tmp_path):
    """A declared dimension decides independently; undeclared ones do not."""
    checkout = Path('/aaaa/bbbb/cccc/dddd/eeeeeeee')
    silent = tmp_path / "silent.json"
    silent.write_text(json.dumps({"schema": 1, "benchmarks": {}}), encoding="utf-8")
    assert BenchmarkBaseline(silent).checkout_path_refusal(checkout) is None

    speaking = tmp_path / "speaking.json"
    speaking.write_text(
        json.dumps(
            {"schema": 1, "benchmarks": {}, "measurement":
             {"checkout_path_length": 29, "checkout_path_depth": 5}}
        ),
        encoding="utf-8",
    )
    baseline = BenchmarkBaseline(speaking)
    assert baseline.checkout_path_refusal(checkout) is None
    assert 'canonical length 29, depth 5' in baseline.checkout_path_refusal(checkout / 'x')
    assert baseline.checkout_path_refusal(Path('/aaaa/bbbb/cccc/eeeeeeeeeeeee'))

    for nonsense in ("29", True, None, 1.5):
        broken = tmp_path / "broken.json"
        broken.write_text(
            json.dumps(
                {
                    "schema": 1,
                    "benchmarks": {},
                    "measurement": {"checkout_path_length": nonsense},
                }
            ),
            encoding="utf-8",
        )
        assert BenchmarkBaseline(broken).checkout_path_refusal(checkout) is None


def test_collecting_reports_every_finding(tmp_path):  # noqa: D103  -- pytest discovers or injects the fixture by name
    path = tmp_path / "baseline.json"
    pinned = BenchmarkBaseline(path, update=True)
    pinned.observe_counter("count", unit="rows", operations=8, samples=[30, 30, 30])
    pinned.observe_counter("other", unit="rows", operations=8, samples=[30, 30, 30])
    pinned.finish()
    baseline = BenchmarkBaseline(path)
    # Outside a block the first failure raises where it is, as before.
    with pytest.raises(AssertionError, match="count inference regression"):
        baseline.observe_counter("count", unit="rows", operations=8, samples=[90, 90, 90])
    # Inside one, both comparisons run, each answers None, and one error names both.
    with pytest.raises(AssertionError) as failure, baseline.collecting():
        first = baseline.observe_counter("count", unit="rows", operations=8, samples=[90, 90, 90])
        second = baseline.observe_counter("other", unit="rows", operations=8, samples=[1, 1, 1])
        assert first is None
        assert second is None
    assert "count inference regression" in str(failure.value)
    assert "other inference improvement left unpinned" in str(failure.value)
    # A block with no finding raises nothing and the comparison answers its value.
    with baseline.collecting():
        assert baseline.observe_counter("count", unit="rows", operations=8, samples=[30, 30, 30]) == 30
    with pytest.raises(RuntimeError, match="do not nest"), baseline.collecting(), baseline.collecting():
        pass


@pytest.mark.parametrize("platform", ["win32", "darwin"])
def test_perf_refuses_off_linux_before_it_spawns(monkeypatch, platform):
    """perf_event_open is Linux's, so perf is refused elsewhere with that reason and nothing starts."""
    spawned = []
    monkeypatch.setattr("metta_benchmarking.shutil.which", _which)
    monkeypatch.setattr("metta_benchmarking.os.access", lambda _path, _mode: True)
    monkeypatch.setattr("metta_benchmarking._spawn_and_reap", lambda *args, **_options: spawned.append(args))
    monkeypatch.setattr("metta_benchmarking.sys.platform", platform)
    with pytest.raises(RuntimeError, match="perf_event_open, a Linux facility"):
        _run_perf(["true"], {}, controlled=False, timeout=5.0)
    assert spawned == []


def test_the_spawner_refuses_windows_before_it_spawns(monkeypatch):
    """A timeout kills the child's whole process group, which Windows lacks, so it refuses first."""
    spawned = []
    monkeypatch.setattr("metta_benchmarking.os.posix_spawn", lambda *args, **_options: spawned.append(args))
    monkeypatch.setattr("metta_benchmarking.sys.platform", "win32")
    with pytest.raises(RuntimeError, match="Windows has no process groups"):
        _spawn_and_reap(["true"], {}, timeout=5.0, what="cachegrind")
    assert spawned == []
