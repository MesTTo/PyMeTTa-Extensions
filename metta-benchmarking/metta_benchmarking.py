"""Purpose: reusable benchmark plumbing for metta and sibling packages.

A DISTRIBUTION of its own, `metta-benchmarking`, because nothing in the core
needs it: it measures a workload from outside, reads the engine's counters
through the public space surface and imports two atom classes and nothing
else. `metta.testing` used to re-export its ten names; the compatibility
ruling is that nothing binds, so a caller imports them from here.
Guarantees:
  - checkout_path_refusal reports differing declared length or depth with
    the baseline's measured reason [tested:
    test_a_baseline_compares_checkout_length_and_depth; commit=8ca8a387fc61d0918484b19a1a3baf85b6523043]
  - benchmark_case uses fresh untimed setup for every counter sample,
    warmup, and timed round [tested test_benchmark_case_uses_fresh_state]
  - engine movement is decided by the minimum of three inference counts
    against a TWO-SIDED band: a drop beyond the allowance fails as a stale
    pin, because a stale-high pin masks regressions up to its own margin;
    wall time is recorded for advice only [tested
    test_baseline_rejects_inference_movement_beyond_the_allowance]
  - counter slopes compare the inference growth between two fixed workload
    sizes, with fresh state at each point and the same two-sided band
    [tested test_benchmark_counter_slope_uses_fresh_state_and_gates_growth]
  - instruction pins band on both sides of the noise allowance, and a row's
    DECLARED band survives every re-pin: the count is measured, the band is
    declared, and an update writes only the count [tested
    test_baseline_bands_instructions_on_both_sides,
    test_a_declared_instruction_band_survives_a_re_pin]
  - counter comparisons declare their measurement configuration and refuse a
    missing or differing stamp, because artifact presence alone has moved a
    pin 12x with zero code change [tested
    test_baseline_stamps_and_verifies_counter_configuration,
    test_baseline_without_configuration_stamp_refuses_counter_comparison]
  - perf instruction measurements fail loudly when perf or its event output
    fails [tested test_measure_instructions_parses_perf_csv]
  - a box that would not count is a MeasurementRefusedError and not a moved
    row: perf answering `<not counted>` for a requested event, and a controlled
    workload exiting PERF_CONTROL_REFUSED because no acknowledgement arrived,
    both raise it, while every other nonzero exit stays an ordinary
    RuntimeError, which is the workload's own failure
    [tested: test_a_refused_window_is_told_apart_from_a_workload_that_failed;
    commit=11afdcdbad5bbbe37168b5d8528c23a21c42b4b6]
  - one policy decides what a benchmark lane does with that refusal, so no two
    lanes can drift into disagreeing: measured_main skips it with a name and
    exits PERF_CONTROL_REFUSED on a desk, which check.sh's summary renders
    `skipped` without failing the run, and refuses it with an error and exits 1
    where CI=true. It is NOT 0: a lane that measured nothing and exits 0 reads
    `ok`, indistinguishable from one that compared every row
    [tested: test_a_benchmark_lane_skips_a_refusal_locally_and_refuses_it_in_ci;
    commit=0e33a6c1666b3d28c546c252ecaa8eeb87bee759]
  - one perf run may count several events, matched on the event NAME field so
    a unit-carrying event reads beside a bare one, and it hands back each
    run's own standard output so a workload can report a counter perf cannot
    see [tested test_measure_counters_reads_every_requested_event]
  - a counting tool starts the workload the measurement environment's PATH
    names, resolved before the tool runs, because perf puts its own
    directories ahead of the PATH it passes on [tested:
    test_a_counting_tool_starts_the_workload_the_measurement_path_names,
    test_cachegrind_starts_the_workload_the_measurement_path_names;
    commit=310b6a9a2da78cf14d3cbc2919cc1eb88b05f096]
  - a workload named with a directory part is the file it names, which the
    PATH does not decide, and a name that resolves to no executable is refused
    before a tool starts, saying which of the two it was [tested:
    test_a_workload_named_by_its_path_is_the_file_it_names; commit=WORKTREE]
  - an instruction pin and an estimated-cycle pin are ONE mechanism under two
    Metric declarations, so a counter that crosses a foreign boundary is gated
    on both, which is the only safe reading there: foreign code retires no
    inferences at all [tested test_an_estimated_cycle_pin_bands_on_both_sides]
  - a document's policy prose is owned by its runner's SOURCE and rewritten on
    every update, so an extension whose deciding counter is not the default one
    cannot ship a file that states the opposite of its own rule [tested
    test_a_declared_policy_is_written_on_every_update]
  - time never decides. observe_cpu records process CPU seconds beside a
    row's pins and never compares them, because a time reading prices the
    queue as well as the work and the box these gates run on is never quiet.
    What a foreign boundary needs beside instructions:u, a counter that moves
    when a change keeps every instruction and wrecks the memory behaviour
    behind them, is ESTIMATED_CYCLES: Cachegrind's cost of the same window
    over a fixed simulated cache, which no other process on the box can move
    [tested: test_cpu_seconds_are_recorded_and_never_compared,
    test_estimated_cycles_prices_each_access_by_the_level_that_served_it,
    test_measure_simulated_reads_every_run_and_its_output,
    test_a_simulated_window_that_never_opened_is_refused]
Owns:
  - BenchmarkBaseline owns an update file only until its atomic replace
    completes [tested test_baseline_update_is_atomic_json]; update mode may
    prune a case nothing measures [tested
    test_baseline_remove_case_is_update_only], and a subset updater
    verifies the configuration stamp without rewriting it [tested
    test_a_subset_updater_verifies_without_restamping]
  - every measured process, under perf or under Cachegrind, is spawned into a
    process group of its own by _spawn_and_reap, which kills the group and
    reaps the child on timeout or interruption [tested
    test_perf_timeout_kills_and_reaps_process_group]
Open Obligations:
  To Do: None
  Hacks: None
  Future Enhancements: None.
"""

from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess  # nosec B404
import sys
import tempfile
import time
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager, suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from metta import Atom, Expression

_SCHEMA = 1
_COUNTER_SAMPLES = 3
# What a run counts when its caller names nothing else. One literal, so the
# public entry point and the perf invocation behind it cannot disagree.
_DEFAULT_EVENTS = ("instructions:u",)
# A regression must clear a small absolute allowance. The join benchmarks
# reproduce a +2 shift that three measurements prove is not work: the changed
# predicates are never called, the delta does not scale with the workload,
# and an unrelated edit cancels it. Real regressions scale with operations,
# so at these workload sizes a handful of inferences is four orders of
# magnitude below anything worth catching, while a real per-operation shift
# still lands far above the allowance.
_COUNTER_TOLERANCE = 4
#: The key a row uses to declare a WIDER inference allowance than the four
#: above, beside the measurement that justifies it. Symmetric with the
#: instruction side, which has taken a per-row band since it was written, and
#: added for the same reason: a row whose noise was MEASURED wider than the
#: default is a row the default reports forever. The C seat's boot is the case
#: it was added for, whose count moves about twenty-five with nothing but
#: whether the tree has been written over
#: [source: extensions/cmetta/benchmarks/baseline.json,
#: release_0_8_0_boot_environment_note]. A row that declares nothing keeps the
#: four, and a declaration without a measurement beside it is a defect this
#: file cannot catch: say what moved the row and by how much, or leave it.
_COUNTER_ALLOWANCE_KEY = "inference_allowance"
# The band an instruction row gets when it declares none. It is a DEFAULT and
# never a policy the measurement path imposes: a row whose layout noise was
# measured wider declares its own percent beside the reason, and re-pinning
# re-measures the count while leaving that declaration standing. Writing this
# value back on every update reverted typed-call's measured 5.0 and json-wire's
# 2.5 in silence, so both lanes stood gated tighter than their own documented
# noise (3.13% and 1.56%) and would go red for code layout alone.
_INSTRUCTION_NOISE_PERCENT = 1.0
# The band an estimated-cycle row gets when it declares none. No other
# process can move a simulated count, but the program being simulated is not
# deterministic: its threads interleave by the host's scheduling, so the heap
# a window starts from differs run to run, and with it which accesses miss.
# Three windowed runs of each C seat case spread 0.016% to 0.286% in estimated
# cycles while their simulated D1 misses alone spread up to 30% [measured
# 2026-09-23: extensions/cmetta/benchmarks/cases at a tenth of each size under
# the geometry below, loadavg 25 to 50;
# commit=b0ddc4cf84df2780b7bac2680eefc457ea318552]. Across trees it moves
# further, because Cachegrind indexes its cache by virtual address and where
# the stack and heap land follows the loaded program and the size of the
# environment: over 23 layouts at one commit the same rows' minimum-of-three
# spread 0.14% to 1.80% [measured 2026-09-24: never-called facts appended to
# engine/metta.pl crossed with environment paddings across residues mod 64;
# commit=8d45268e3b374e97fda87d618ab21c10c359fbee]. A row that measured wider than this declares its own band
# beside the measurement.
_ESTIMATED_CYCLES_NOISE_PERCENT = 1.0
#: Cachegrind's simulated hierarchy, FIXED rather than read from the host:
#: left alone, Cachegrind copies the cache geometry of the CPU it runs on, so a
#: pin taken on one machine would move on another with no change to the
#: program. These are the values pythonspeed's benchmarking script fixes for
#: the same reason, a 32 KiB 8-way first-level pair and an 8 MiB 16-way last
#: level [source: https://github.com/pythonspeed/cachegrind-benchmarking/blob/bc45f72bf7b4656d723104fe95a318bbd34dda7b/cachegrind.py].
_SIMULATED_CACHES = ("--I1=32768,8,64", "--D1=32768,8,64", "--LL=8388608,16,64")
#: What Cachegrind's cache simulation counts, in the order its summary line
#: writes them: instruction reads, data reads and data writes, each with its
#: first-level misses and its last-level misses [source: valgrind 3.26
#: Cachegrind manual, the Ir, I1mr, ILmr, Dr, D1mr, DLmr, Dw, D1mw and DLmw
#: events].
_CACHEGRIND_EVENTS = ("Ir", "I1mr", "ILmr", "Dr", "D1mr", "DLmr", "Dw", "D1mw", "DLmw")
# What a document says about itself when its owner declares nothing else. A
# reader meets these before any number, so they have to be true of the rows
# below them; an extension measuring across a foreign boundary overrides them.
_DEFAULT_POLICIES = {
    "counter_policy": (
        "stats().inferences minimum of three and fixed two-point growth "
        "slopes decide; wall time advises"
    ),
    "instruction_policy": (
        "perf instructions:u minimum of three, one percent noise "
        "allowance unless a row declares its own beside the "
        "measurement that justified it"
    ),
}


@dataclass(frozen=True)
class Metric:
    """One percent-banded, two-sided pin over the minimum of several samples.

    `noun` names it in a failure message, and `value_key` and `band_key` are
    its two fields inside a baseline case. Retired instructions and estimated
    cycles both count whole events, and nothing else about the two pins
    differs, which is why they are one mechanism wearing two faces rather than
    two mechanisms.
    """

    noun: str
    value_key: str
    band_key: str
    default_percent: float


#: perf's retired-instruction counter, and the reason its band is DECLARED
#: rather than imposed is written at _INSTRUCTION_NOISE_PERCENT.
INSTRUCTIONS = Metric(
    "instruction", "instructions", "instruction_noise_percent",
    _INSTRUCTION_NOISE_PERCENT,
)
#: What the same window would cost in cycles on a simulated cache. It exists
#: because an instruction count cannot see time: a change that keeps every
#: instruction and wrecks the memory behaviour behind them is invisible to the
#: first counter and plain in this one. nanobench reports ins/op beside cyc/op
#: for that reason [source: https://github.com/martinus/nanobench README,
#: "6.65 instructions are executed in 24.07 CPU cycles"], and this tree has the
#: failure on record: a C wire encoder measured 526x faster on the inference
#: counter while CPU time said it was 1.8x SLOWER. Pair the two across any
#: foreign boundary; neither alone decides. It is simulated rather than read
#: from task-clock because a time reading prices the queue as well as the work:
#: at loadavg 30 a task-clock triple spread 38% to 64% over runs whose
#: instructions:u spread under 0.13% [source: extensions/cmetta/benchmarks/
#: baseline.json, measurement_conditions], and these gates run at loadavg 25 to
#: 100 [measured 2026-09-23]. estimated_cycles gives the weights. It is a model
#: and not a measurement of hardware: it cannot see kernel work, TLB misses or
#: speculation, it indexes the cache by virtual address, it counts every
#: iteration of a REP-prefixed instruction where perf counts one, and valgrind
#: schedules a threaded program's threads its own way [source: valgrind 3.26
#: Cachegrind manual, section 5.8.3 Accuracy]. So its count and the instruction
#: count are two pins, never one number converted into the other.
ESTIMATED_CYCLES = Metric(
    "estimated cycle", "estimated_cycles", "estimated_cycles_noise_percent",
    _ESTIMATED_CYCLES_NOISE_PERCENT,
)


def count_atoms(atom: Any) -> int:
    """Count every atom node in a term without recursing."""
    if not isinstance(atom, Atom):
        msg = f"count_atoms expects an Atom, got {type(atom).__name__}"
        raise TypeError(msg)
    count = 0
    stack = [atom]
    while stack:
        node = stack.pop()
        count += 1
        if isinstance(node, Expression):
            stack.extend(node.children)
    return count


def _counter_observation(
    name: str,
    samples: Sequence[int] | None,
) -> tuple[list[int] | None, int | None]:
    sample_values = None if samples is None else list(samples)
    if sample_values is None:
        return None, None
    if len(sample_values) < _COUNTER_SAMPLES:
        msg = f"benchmark counter needs at least {_COUNTER_SAMPLES} samples"
        raise ValueError(msg)
    if any(
        not isinstance(value, int) or isinstance(value, bool) or value < 0
        for value in sample_values
    ):
        msg = f"invalid inference samples for {name}: {sample_values!r}"
        raise ValueError(msg)
    return sample_values, min(sample_values)


def _compare_counter(
    name: str,
    expected: Mapping[str, Any],
    sample_values: list[int] | None,
    observed: int | None,
) -> int | None:
    baseline = expected.get("inferences")
    if observed is None:
        if baseline is not None:
            msg = f"{name} is engine-free but its baseline has inferences {baseline!r}"
            raise AssertionError(
                msg
            )
        return None
    if isinstance(baseline, bool) or not isinstance(baseline, int):
        msg = f"{name} baseline has invalid inferences {baseline!r}"
        raise AssertionError(msg)  # noqa: TRY004  -- the harness is checking its own invariant, so AssertionError is the intended contract
    allowance = expected.get(_COUNTER_ALLOWANCE_KEY, _COUNTER_TOLERANCE)
    if not isinstance(allowance, int) or isinstance(allowance, bool) or allowance < 0:
        msg = f"{name} baseline has invalid {_COUNTER_ALLOWANCE_KEY} {allowance!r}"
        raise AssertionError(msg)
    if observed > baseline + allowance:
        msg = (
            f"{name} inference regression: minimum of {sample_values!r} is "
            f"{observed}, baseline {baseline} plus the {allowance} "
            f"inference allowance"
        )
        raise AssertionError(
            msg
        )
    #The band is two-sided because a stale-high pin masks real regressions
    #up to its own margin: file-load sat at 8704891 while the tree measured
    #722264, so anything under 12x slower would still have read green. A
    #drop beyond the allowance therefore fails until the pin is re-measured
    #and its mechanism recorded beside it.
    #
    #The improvement side reads the HIGHEST sample where the regression side
    #reads the lowest, and the asymmetry is the point: min-of-n is the right
    #statistic for "did anything get slower" and the wrong one for "did this
    #get faster". One anomalous low sample is not an improvement, and the C
    #seat's boot produced exactly that inside the gate, [1484396, 1485362,
    #1485362] against a 1485363 pin, where nine consecutive samples measured
    #outside it read 1485362 every time [measured 2026-09-04]. A row that
    #really improved has EVERY sample below the pin, so this still fails on
    #one and cannot be quieted by a noisy run.
    highest = max(sample_values) if sample_values else observed
    if highest < baseline - allowance:
        msg = (
            f"{name} inference improvement left unpinned: every sample of "
            f"{sample_values!r} is under baseline {baseline} minus the "
            f"{allowance} inference allowance; re-pin with "
            f"--update-baseline and record the mechanism beside the pin"
        )
        raise AssertionError(
            msg
        )
    return observed


def _counter_samples(
    operation: Callable[[Any], int],
    *,
    operations: int,
    setup: Callable[[], Any],
    teardown: Callable[[Any], None],
    engine: Callable[[Any], Any],
) -> list[int]:
    samples = []
    for _ in range(_COUNTER_SAMPLES):
        state = setup()
        try:
            with engine(state).stats() as stats:
                completed = operation(state)
            if completed != operations:
                msg = f"counter sample completed {completed} operations, expected {operations}"
                raise AssertionError(
                    msg
                )
            samples.append(stats.inferences)
        finally:
            teardown(state)
    return samples


def _required_counter_observation(name: str, samples: Sequence[int]) -> tuple[list[int], int]:
    values, observed = _counter_observation(name, samples)
    if values is None or observed is None:
        msg = f"{name} lost its required inference samples"
        raise RuntimeError(msg)
    return values, observed


def _counter_slope_observation(
    name: str,
    small_operations: int,
    large_operations: int,
    small_samples: Sequence[int],
    large_samples: Sequence[int],
) -> tuple[list[int], list[int], int]:
    if small_operations <= 0 or large_operations <= small_operations:
        msg = "counter slope needs positive operation counts in increasing order"
        raise ValueError(msg)
    small_values, small = _required_counter_observation(f"{name} small", small_samples)
    large_values, large = _required_counter_observation(f"{name} large", large_samples)
    observed = large - small
    if observed < 0:
        msg = f"{name} inference count fell from {small} to {large} as the workload grew"
        raise ValueError(
            msg
        )
    return small_values, large_values, observed


def _counter_slope_case(
    document: Mapping[str, Any], name: str, unit: str
) -> dict[str, Any]:
    case = document["benchmarks"].get(name)
    if case is None:
        msg = f"benchmark {name!r} has no counter observation"
        raise KeyError(msg)
    if case.get("unit") != unit:
        msg = f"{name} unit changed from {case.get('unit')!r} to {unit!r}"
        raise AssertionError(msg)
    return case


def _compare_counter_slope(
    name: str,
    expected: Any,
    *,
    small_operations: int,
    large_operations: int,
    small_values: list[int],
    large_values: list[int],
    observed: int,
) -> int:
    if not isinstance(expected, dict):
        msg = f"{name} has no valid inference slope baseline"
        raise AssertionError(msg)  # noqa: TRY004  -- the harness is checking its own invariant, so AssertionError is the intended contract
    if expected.get("small_operations") != small_operations:
        msg = (
            f"{name} slope small operation count changed from "
            f"{expected.get('small_operations')!r} to {small_operations}"
        )
        raise AssertionError(
            msg
        )
    if expected.get("large_operations") != large_operations:
        msg = (
            f"{name} slope large operation count changed from "
            f"{expected.get('large_operations')!r} to {large_operations}"
        )
        raise AssertionError(
            msg
        )
    baseline = expected.get("delta_inferences")
    if isinstance(baseline, bool) or not isinstance(baseline, int) or baseline < 0:
        msg = f"{name} has an invalid inference slope baseline"
        raise AssertionError(msg)
    if observed > baseline + _COUNTER_TOLERANCE:
        msg = (
            f"{name} inference slope regression: {large_values!r} minus "
            f"{small_values!r} has minimum growth {observed}, baseline {baseline} "
            f"plus the {_COUNTER_TOLERANCE} inference allowance"
        )
        raise AssertionError(
            msg
        )
    if observed < baseline - _COUNTER_TOLERANCE:
        msg = (
            f"{name} inference slope improvement left unpinned: {large_values!r} "
            f"minus {small_values!r} has minimum growth {observed}, baseline "
            f"{baseline} minus the {_COUNTER_TOLERANCE} inference allowance; "
            f"re-pin with --update-baseline and record the mechanism beside the pin"
        )
        raise AssertionError(
            msg
        )
    return observed


def _measurement_observation(
    name: str, metric: Metric, samples: Sequence[int]
) -> int:
    if len(samples) < _COUNTER_SAMPLES or any(
        not isinstance(value, int) or isinstance(value, bool) or value <= 0
        for value in samples
    ):
        msg = f"invalid {metric.noun} samples for {name}: {samples!r}"
        raise ValueError(msg)
    return min(samples)


def _compare_measurement(
    name: str,
    metric: Metric,
    case: Mapping[str, Any],
    samples: Sequence[int],
    observed: int,
) -> int:
    baseline = case.get(metric.value_key)
    allowance = case.get(metric.band_key)
    if isinstance(baseline, bool) or not isinstance(baseline, int) or baseline <= 0:
        msg = f"{name} has no valid {metric.noun} baseline"
        raise AssertionError(msg)
    if (
        isinstance(allowance, bool)
        or not isinstance(allowance, (int, float))
        or allowance < 0
    ):
        msg = f"{name} has no valid {metric.noun} noise allowance"
        raise AssertionError(msg)
    ceiling = baseline * (1.0 + allowance / 100.0)
    if observed > ceiling:
        msg = (
            f"{name} {metric.noun} regression: minimum of {list(samples)!r} is "
            f"{observed}, baseline {baseline} plus {allowance:g}% is "
            f"{ceiling:.0f}"
        )
        raise AssertionError(
            msg
        )
    floor = baseline * (1.0 - allowance / 100.0)
    if observed < floor:
        msg = (
            f"{name} {metric.noun} improvement left unpinned: minimum of "
            f"{list(samples)!r} is {observed}, baseline {baseline} minus "
            f"{allowance:g}% is {floor:.0f}; re-pin with --update and "
            f"record the mechanism beside the pin"
        )
        raise AssertionError(
            msg
        )
    return observed


class BenchmarkBaseline:
    """Committed counter and advisory wall baselines for benchmark_case."""

    def __init__(  # noqa: D107  -- the enclosing class documents construction and the object invariants
        self,
        path: str | os.PathLike[str],
        *,
        update: bool = False,
        compare_counters: bool = True,
        policies: Mapping[str, str] | None = None,
    ):
        self.path = Path(path)
        self.update = update
        self.compare_counters = compare_counters or update
        # An extension whose deciding counter is not the default one says so
        # here, and says it in SOURCE rather than in the file: the C extension's
        # counters are instructions:u and estimated cycles paired, because
        # foreign code retires no inferences, so a document of its rows
        # carrying the sentence "stats().inferences ... decide" would state
        # the opposite of its own rule. Declared policies are written on every
        # update, unlike a per-row noise band, because prose is authored and a
        # band is measured.
        self.policies = dict(_DEFAULT_POLICIES | dict(policies or {}))
        # Inside a collecting() block a failed comparison is recorded here and the
        # next comparison still runs; outside one it raises where it fails.
        self._findings: list[str] | None = None
        if not self.path.is_file():
            if not update:
                msg = f"benchmark baseline does not exist: {self.path}"
                raise FileNotFoundError(msg)
            self._document: dict[str, Any] = {
                "schema": _SCHEMA,
                **self.policies,
                "benchmarks": {},
            }
            return
        with self.path.open(encoding="utf-8") as handle:
            document = json.load(handle)
        if document.get("schema") != _SCHEMA:
            msg = f"benchmark baseline schema must be {_SCHEMA}, got {document.get('schema')!r}"
            raise ValueError(
                msg
            )
        if not isinstance(document.get("benchmarks"), dict):
            msg = "benchmark baseline benchmarks must be an object"
            raise ValueError(msg)  # noqa: TRY004  -- the harness is checking its own invariant, so AssertionError is the intended contract
        self._document = document

    @property
    def cases(self) -> Mapping[str, Mapping[str, Any]]:  # noqa: D102  -- the enclosing type and implemented protocol supply this method contract
        return self._document["benchmarks"]

    def checkout_path_refusal(self, checkout: Path) -> str | None:
        """Explain a different declared checkout shape, or allow comparison.

        Only counters whose driver declares the boot window use this guard.
        The baseline owns each dimension and the measured reason for it.
        """
        measurement = self._document.get("measurement")
        if not isinstance(measurement, Mapping):
            return None
        actual = {"length": len(str(checkout)), "depth": len(checkout.parts) - 1}
        pinned = {
            dimension: value
            for dimension in actual
            if type(value := measurement.get(f"checkout_path_{dimension}")) is int
        }
        if all(actual[dimension] == value for dimension, value in pinned.items()):
            return None
        reason = measurement.get("checkout_path_reason", "location-sensitive boot counters")
        return (
            f"checkout length {actual['length']}, depth {actual['depth']}; "
            f"canonical length {pinned.get('length', 'unspecified')}, "
            f"depth {pinned.get('depth', 'unspecified')}; {reason}"
        )

    @contextmanager
    def collecting(self):
        """Every comparison inside the block reaches the log.

        A test that pins two things about one row, its count and its slope, or a
        tool that holds every tier of a table to a pin, used to stop at the first
        comparison that failed, so the second was never read: the direct-join
        and prepared-join slopes and the extension add-atom tier sat stale
        across a re-take because their point-count assertion failed first at
        every measured point [measured 2026-09-19: 0 slope findings at every
        ladder point and 2 on the tree that re-pinned the counts;
        commit=e81e5369ecce854c09c75113cf1ca30e76d06154]. Inside this block a failed comparison is recorded and
        answers None, and one AssertionError carrying every finding is raised
        when the block ends [tested: test_collecting_reports_every_finding;
        commit=e81e5369ecce854c09c75113cf1ca30e76d06154].
        """
        if self._findings is not None:
            msg = "collecting() blocks do not nest"
            raise RuntimeError(msg)
        self._findings = []
        try:
            yield
        finally:
            findings, self._findings = self._findings, None
        if findings:
            raise AssertionError("\n".join(findings))

    def _verdict(self, compare: Callable[[], Any]) -> Any:
        """Run one comparison: inside collecting() a failure is recorded and answers None."""
        if self._findings is None:
            return compare()
        try:
            return compare()
        except AssertionError as failure:
            self._findings.append(str(failure))
            return None

    def observe_counter(
        self,
        name: str,
        *,
        unit: str,
        operations: int,
        samples: Sequence[int] | None,
    ) -> int | None:
        """Record or compare one deterministic engine counter."""
        if operations <= 0:
            msg = f"benchmark operations must be positive, got {operations}"
            raise ValueError(msg)
        sample_values, observed = _counter_observation(name, samples)

        if self.update:
            previous = self._document["benchmarks"].get(name, {})
            #`previous` first, so a declared inference_allowance and every
            #other field a row carries SURVIVE a re-pin: re-pinning re-measures
            #the count, it does not re-decide what the row's noise is.
            self._document["benchmarks"][name] = {
                **previous,
                "unit": unit,
                "operations": operations,
                "inferences": observed,
            }
            return observed

        expected = self._case(name, unit=unit, operations=operations)
        return self._verdict(lambda: _compare_counter(name, expected, sample_values, observed))

    def observe_counter_slope(
        self,
        name: str,
        *,
        unit: str,
        small_operations: int,
        large_operations: int,
        small_samples: Sequence[int],
        large_samples: Sequence[int],
    ) -> int:
        """Record or compare inference growth between two workload sizes."""
        small_values, large_values, observed = _counter_slope_observation(
            name,
            small_operations,
            large_operations,
            small_samples,
            large_samples,
        )
        case = _counter_slope_case(self._document, name, unit)
        if self.update:
            case["inference_slope"] = {
                "small_operations": small_operations,
                "large_operations": large_operations,
                "delta_inferences": observed,
            }
            return observed
        return self._verdict(lambda: _compare_counter_slope(
            name,
            case.get("inference_slope"),
            small_operations=small_operations,
            large_operations=large_operations,
            small_values=small_values,
            large_values=large_values,
            observed=observed,
        ))

    def remove_case(self, name: str) -> None:
        """Drop a pinned case during an update, for rows nothing measures.

        A pinned row no measurement reaches can never fail, so it survives
        renames and lost artifacts as a dead receipt; pruning is part of
        re-pinning and is therefore update-only.
        """
        if not self.update:
            msg = f"remove_case({name!r}) outside update mode"
            raise AssertionError(msg)
        if name not in self._document["benchmarks"]:
            msg = f"benchmark baseline has no case named {name!r}"
            raise KeyError(msg)
        del self._document["benchmarks"][name]

    def observe_configuration(
        self, live: Mapping[str, Any], *, stamp: bool | None = None
    ) -> None:
        """Stamp or verify the measurement configuration the counters ran in.

        Deterministic counters only compare within one configuration: the C
        reader's presence moved file-load 8704891 to 722264 with zero code
        change, so a tree measuring in one mode against pins from the other
        produces confounded verdicts. Update mode stamps the live
        configuration; comparison mode refuses a missing or differing stamp.

        ``stamp=False`` makes even an update verify-only: a runner that
        re-measures a SUBSET of the document (the instruction checker) must
        not rewrite the fingerprint the other pins were measured under, so
        it verifies when a stamp exists and leaves an absent stamp to the
        owning full-battery updater.
        """
        if stamp is None:
            stamp = self.update
        if self.update and stamp:
            self._document["counter_configuration"] = dict(live)
            return
        stored = self._document.get("counter_configuration")
        if stored is None and self.update:
            return
        if stored is None:
            msg = (
                f"benchmark baseline carries no counter_configuration stamp; "
                f"live configuration is {dict(live)!r}: re-pin with "
                f"--update-baseline so comparisons declare their configuration"
            )
            raise AssertionError(msg)
        if stored != dict(live):
            msg = (
                f"counter configuration drift: baseline pinned under "
                f"{stored!r} but this run measures under {dict(live)!r}; "
                f"restore the pinned configuration (build the artifact or "
                f"unset the mode override) or re-pin with --update-baseline"
            )
            raise AssertionError(msg)

    def validate_case(self, name: str, *, unit: str, operations: int) -> None:
        """Check metadata when a wall-only run deliberately skips counters."""
        if operations <= 0:
            msg = f"benchmark operations must be positive, got {operations}"
            raise ValueError(msg)
        self._case(name, unit=unit, operations=operations)

    def observe_wall(self, name: str, seconds_per_operation: float) -> None:
        """Record wall time or retain it as advisory comparison metadata."""
        if seconds_per_operation <= 0:
            msg = "benchmark wall time must be positive"
            raise ValueError(msg)
        case = self._document["benchmarks"].get(name)
        if case is None:
            msg = f"benchmark {name!r} has no counter observation"
            raise KeyError(msg)
        if self.update:
            case["wall_seconds_per_operation"] = seconds_per_operation

    def observe_cpu(self, name: str, seconds_per_operation: float) -> None:
        """Record process CPU time, the advisory counter beside instructions.

        Wall time advises for a workload the engine runs by itself. It cannot
        advise for one that crosses a foreign boundary, where the reason to
        record a second counter at all is that the first one is blind: SWI's
        inference counter retires nothing for work done inside C or Rust, and
        a change measured 526x faster by inferences was 1.8x SLOWER by CPU.
        instructions:u and ESTIMATED_CYCLES decide those rows and this is what
        they are checked against, so the pairing is an artifact rather than a
        claim.

        Never compared, for the reason wall time is never compared: CPU seconds
        move with frequency scaling and with what else the box is doing, and
        the box these gates run on is never quiet.
        """
        if seconds_per_operation <= 0:
            msg = "benchmark CPU time must be positive"
            raise ValueError(msg)
        case = self._document["benchmarks"].get(name)
        if case is None:
            msg = f"benchmark {name!r} has no counter observation"
            raise KeyError(msg)
        if self.update:
            case["cpu_seconds_per_operation"] = seconds_per_operation

    def observe_measurement(
        self, name: str, metric: Metric, samples: Sequence[int]
    ) -> int | None:
        """Record or compare one percent-banded counter beside its band.

        The count is measured; the noise band beside it is DECLARED, so an
        update writes the fresh count and leaves the declaration standing.
        A row whose band was widened for measured layout noise keeps that
        band across every re-pin, and a row that declares none is filled
        with the metric's default once.
        """
        observed = _measurement_observation(name, metric, samples)
        case = self._document["benchmarks"].get(name)
        if self.update:
            if case is None:
                msg = f"benchmark {name!r} has no wall/counter baseline"
                raise KeyError(msg)
            case[metric.value_key] = observed
            case.setdefault(metric.band_key, metric.default_percent)
            return observed

        def compared() -> int:
            if case is None:
                msg = f"benchmark baseline has no case named {name!r}"
                raise AssertionError(msg)
            return _compare_measurement(name, metric, case, samples, observed)

        return self._verdict(compared)

    def observe_instructions(self, name: str, samples: Sequence[int]) -> int | None:
        """Record or compare perf's retired-instruction counter; None for a finding a collecting() block recorded."""
        return self.observe_measurement(name, INSTRUCTIONS, samples)

    def finish(self) -> None:
        """Atomically write an update; normal comparison mode writes nothing."""
        if not self.update:
            return
        self._document.update(self.policies)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{self.path.name}.", dir=self.path.parent
        )
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                json.dump(self._document, handle, indent=2, sort_keys=True)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            Path(temporary_name).replace(self.path)
            directory_descriptor = os.open(self.path.parent, os.O_RDONLY)
            try:
                os.fsync(directory_descriptor)
            finally:
                os.close(directory_descriptor)
        except BaseException:
            with suppress(FileNotFoundError):
                Path(temporary_name).unlink()
            raise

    def _case(self, name: str, *, unit: str, operations: int) -> Mapping[str, Any]:
        case = self._document["benchmarks"].get(name)
        if case is None:
            msg = f"benchmark baseline has no case named {name!r}; regenerate it explicitly"
            raise AssertionError(
                msg
            )
        if case.get("unit") != unit:
            msg = f"{name} unit changed from {case.get('unit')!r} to {unit!r}"
            raise AssertionError(msg)
        if case.get("operations") != operations:
            msg = f"{name} operation count changed from {case.get('operations')!r} to {operations}"
            raise AssertionError(
                msg
            )
        return case


def benchmark_case(
    benchmark: Any,
    baseline: BenchmarkBaseline,
    *,
    name: str,
    unit: str,
    operations: int,
    operation: Callable[[Any], int],
    setup: Callable[[], Any],
    teardown: Callable[[Any], None],
    engine: Callable[[Any], Any] | None,
    rounds: int = 5,
    warmup_rounds: int = 2,
) -> int:
    """Measure one fixed workload through pytest-benchmark and exact counters."""

    def checked(state: Any) -> int:
        completed = operation(state)
        if completed != operations:
            msg = f"{name} completed {completed} {unit}, expected {operations}"
            raise AssertionError(msg)
        return completed

    samples: list[int] | None = None
    if engine is not None and baseline.compare_counters:
        samples = _counter_samples(
            checked,
            operations=operations,
            setup=setup,
            teardown=teardown,
            engine=engine,
        )

    if baseline.compare_counters:
        inference_min = baseline.observe_counter(
            name,
            unit=unit,
            operations=operations,
            samples=samples,
        )
    else:
        baseline.validate_case(name, unit=unit, operations=operations)
        inference_min = None
    benchmark.extra_info["unit"] = unit
    benchmark.extra_info["operations_per_round"] = operations
    benchmark.extra_info["inference_samples"] = samples
    benchmark.extra_info["inference_min"] = inference_min

    def timed_setup():
        return (setup(),), {}

    result = benchmark.pedantic(
        checked,
        setup=timed_setup,
        teardown=teardown,
        rounds=rounds,
        warmup_rounds=warmup_rounds,
    )
    if benchmark.stats is not None:
        seconds_per_operation = benchmark.stats.stats.min / operations
        baseline.observe_wall(name, seconds_per_operation)
        benchmark.extra_info["wall_seconds_per_operation"] = seconds_per_operation
    return result


def benchmark_counter_slope(
    baseline: BenchmarkBaseline,
    *,
    name: str,
    unit: str,
    small_operations: int,
    small_operation: Callable[[Any], int],
    large_operations: int,
    large_operation: Callable[[Any], int],
    setup: Callable[[], Any],
    teardown: Callable[[Any], None],
    engine: Callable[[Any], Any],
) -> int | None:
    """Gate inference growth between two fixed workload sizes."""
    if not baseline.compare_counters:
        return None
    small_samples = _counter_samples(
        small_operation,
        operations=small_operations,
        setup=setup,
        teardown=teardown,
        engine=engine,
    )
    large_samples = _counter_samples(
        large_operation,
        operations=large_operations,
        setup=setup,
        teardown=teardown,
        engine=engine,
    )
    return baseline.observe_counter_slope(
        name,
        unit=unit,
        small_operations=small_operations,
        large_operations=large_operations,
        small_samples=small_samples,
        large_samples=large_samples,
    )


#: What the kernel will let an unprivileged process count. Read when perf
#: answers nothing, so a refusal names the knob that decides it rather than
#: sending the reader into this harness: 2 or less is needed, and a container
#: needs `--security-opt seccomp=unconfined` before perf_event_open is
#: permitted at all.
PARANOID = Path("/proc/sys/kernel/perf_event_paranoid")

#: The status a CONTROLLED workload exits with when perf never acknowledged a
#: control command: no window ever opened, so the process measured nothing.
#: 125 is the status this tree and its tools already read as "the wrapper
#: failed rather than the command" -- timeout(1) uses it for a failure in
#: itself, `git bisect run` reads it as "this run says nothing about the
#: commit", and bounded.sh refuses with it when the process that started a
#: command had already exited [source: coreutils timeout(1) EXIT STATUS;
#: git-bisect(1), "run <cmd>"; bounded.sh, the arming-race refusal].
PERF_CONTROL_REFUSED = 125


class MeasurementRefusedError(RuntimeError):
    """The box would not take this measurement, so it says nothing about the tree.

    Raised where perf could not count -- another session holding the PMU, a
    kernel that will not let this process count itself, a container whose
    seccomp profile denies perf_event_open, a control window that never opened
    because an acknowledgement never came -- and never where a workload
    answered wrongly. A lane that catches this reports a SKIP by name, because
    reading contention as a regression is reading the box as a code change.
    """


#: The line the ordinary warm boot prints per governed artifact it left behind.
GOVERNED_ARTIFACT_MARKER = "governed-artifact "


def prepare_governed_artifacts(root: Path, *, swipl: str = "swipl") -> tuple[str, ...]:
    """Purge the governed .qlf set, warm it through the ordinary boot, list it.

    Two child processes, neither of them measured. The first runs
    engine/qlf_boot.pl's own purge door (metta_qlf_boot:purge_all_qlf/0),
    which deletes every artifact the boot governs: the engine units and the
    library halves an earlier lane compiled beside their source. The second is
    the ordinary boot through engine/bench.pl (metta_bench:bench_run(boot)),
    which generates the set the shipping boot loads and then lists it through
    the boot's own inventory door (metta_qlf_boot:qlf_files/2). The inventory
    comes back as repository-relative paths in the engine's order, so a driver
    pins its length and a fixture compares its members, and every boot row
    (engine/bench-baseline.json, the C fixture) reads one artifact state
    whatever ran before it: a boot reads 319,090 inferences after this
    preparation and 319,073 with two library artifacts another lane left
    beside the three the boot compiles, past the four-inference allowance
    [measured 2026-09-11: sh check.sh engine-bench after the re-pin lanes
    versus after this preparation; fixture=the merged tree 8e6968ecb; commit=23033852660c31aeadeb2719a1eead355c36e0bf].
    A generating boot is a different workload from a loading one (3,129,543
    against 612,598 inferences, engine/bench.py's header), which is why the
    warm boot is a child of its own and never the measured process.
    """
    engine = Path(root) / "engine"
    # Both argument vectors are a fixed executable name, fixed flags and goals
    # built from this tree's own file names, never from input.
    subprocess.run(  # noqa: S603  # nosec B603
        [swipl, "-q", "-s", str(engine / "qlf_boot.pl"),
         "-g", "metta_qlf_boot:purge_all_qlf", "-t", "halt"],
        check=True, capture_output=True, text=True,
    )
    warmed = subprocess.run(  # noqa: S603  # nosec B603
        [
            swipl, "-q", "--stack_limit=8g", "-g",
            "metta_bench:bench_run(boot),"
            "metta_bench:bench_engine_directory(Here),"
            "metta_qlf_boot:qlf_files(Here,Files),"
            "forall(lists:member(File,Files),"
            f"format('{GOVERNED_ARTIFACT_MARKER}~w~n',[File]))",
            "-t", "halt", str(engine / "bench.pl"),
        ],
        check=True, capture_output=True, text=True,
    )
    listed = [
        line[len(GOVERNED_ARTIFACT_MARKER):].strip()
        for line in warmed.stdout.splitlines()
        if line.startswith(GOVERNED_ARTIFACT_MARKER)
    ]
    if not listed:
        msg = f"the warm boot listed no governed artifact: {warmed.stdout!r}"
        raise RuntimeError(msg)
    resolved_root = Path(root).resolve()
    relative = []
    for path in listed:
        candidate = Path(path)
        try:
            relative.append(str(candidate.resolve().relative_to(resolved_root)))
        except ValueError:
            relative.append(path)
    return tuple(relative)


def _paranoid_reading() -> str:
    """What perf_event_paranoid says right now, or why it could not be read."""
    try:
        return PARANOID.read_text(encoding="utf-8").strip()
    except OSError:
        return "unreadable"


def refusal_is_fatal() -> bool:
    """Whether a box that would not measure should also fail the lane.

    One definition, because three lanes draw this line and a lane that drew it
    differently would pass in CI without measuring. It is the line check.sh
    already draws for a prerequisite the repository cannot provide: a runner
    that cannot measure is a broken runner, and a lane that passes without
    measuring is worse than a red one, while a developer's box is shared and a
    contended PMU is not a code change
    [source: tests/checks/check_upstream_parity.py, upstream_prerequisite].
    """
    return os.environ.get("CI") == "true"


def measured_main(entry: Callable[[], int]) -> int:
    """Run a benchmark lane and turn a refused measurement into a named skip.

    Every benchmark entry point in this tree goes through here rather than
    catching for itself, so no two lanes can drift into disagreeing about when
    a box that would not count is allowed to pass. The policy is the line
    check.sh already draws for a prerequisite the repository cannot provide:
    refuse where CI=true, because a runner that cannot count is a broken runner
    and a lane that passes without measuring is worse than a red one; print a
    named skip elsewhere, because a developer's box is shared and a PMU another
    session holds is not a code change
    [source: tests/checks/check_upstream_parity.py, upstream_prerequisite].

    The local skip exits PERF_CONTROL_REFUSED rather than 0, so check.sh's
    summary says `skipped` for it instead of `ok`. Exit 0 made the two
    indistinguishable in the one line a reader scans, and the difference is the
    whole point: one of them compared every row.
    """
    try:
        return entry()
    except MeasurementRefusedError as refusal:
        #The verdict goes on its own line and the diagnosis under it, because
        #the diagnosis carries perf's transcript and a reader scanning a gate
        #log has to see which of the two words this lane said without reading
        #the rest.
        if refusal_is_fatal():
            print(
                "error: this benchmark lane measured nothing and will not pass "
                "on that; a CI runner that cannot count is a broken runner.",
                file=sys.stderr,
            )
            print(f"  {refusal}", file=sys.stderr)
            return 1
        print(
            "note: the box refused the measurement, so nothing here says the "
            "tree moved; re-run it where the PMU is free."
        )
        print(f"  {refusal}")
        #Not 0. A lane that measured nothing and exits 0 reads `ok` in
        #check.sh's summary, indistinguishable from one that compared every row
        #and passed, and that is not a small difference: mork-bench reported
        #`ok` on four of five full gate runs while another session held the PMU
        #and it compared not one row. 125 is the number this tree already
        #spells "this run says nothing", and check.sh renders it `skipped`
        #without failing the run
        #[tested: test_a_benchmark_lane_skips_a_refusal_locally_and_refuses_it_in_ci;
        #commit=0e33a6c1666b3d28c546c252ecaa8eeb87bee759].
        return PERF_CONTROL_REFUSED


def _counter_request(
    command: Sequence[str],
    events: Sequence[str],
    rounds: int,
    timeout: float,
) -> float:
    if rounds < _COUNTER_SAMPLES:
        msg = f"counter measurement needs at least {_COUNTER_SAMPLES} rounds"
        raise ValueError(msg)
    if not command:
        msg = "counter measurement command cannot be empty"
        raise ValueError(msg)
    if not events:
        msg = "counter measurement needs at least one perf event"
        raise ValueError(msg)
    if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or timeout <= 0:
        msg = f"counter measurement timeout must be positive, got {timeout!r}"
        raise ValueError(msg)
    #Finding perf belongs to the code that runs it, not here. A caller that
    #substitutes _run_perf substitutes the whole act of running perf, and
    #should not still need perf installed to reach the parsing it is testing.
    return float(timeout)


def _parse_counter_sample(
    returncode: int, stdout: str, stderr: str, events: Sequence[str]
) -> dict[str, float]:
    if returncode == PERF_CONTROL_REFUSED:
        detail = " ".join((stderr.strip() or stdout.strip()).split())
        msg = (
            "the measured window never opened: perf did not acknowledge a "
            "control command, which is what it does when its counter failed to "
            f"arm while another session held the PMU. {PARANOID} reads "
            f"{_paranoid_reading()}. perf and the workload said: {detail[-300:]}"
        )
        raise MeasurementRefusedError(msg)
    if returncode != 0:
        detail = stderr.strip() or stdout.strip()
        msg = f"perf stat failed with exit {returncode}: {detail}"
        raise RuntimeError(msg)
    #perf's -x, row is value,unit,event,run-time,percentage, so the event NAME
    #is field 2 and matching there rather than on a substring is what lets one
    #run ask for several events: task-clock carries the unit `msec` where
    #instructions:u carries none [source: perf-stat(1), -x SEP].
    rows: dict[str, list[str]] = {event: [] for event in events}
    for line in stderr.splitlines():
        fields = line.split(",")
        if len(fields) > 2 and fields[2] in rows:
            rows[fields[2]].append(fields[0])
    sample: dict[str, float] = {}
    for event, values in rows.items():
        if len(values) != 1:
            msg = f"perf stat did not return one {event} counter: {stderr.strip()}"
            raise RuntimeError(msg)
        try:
            #An integer count stays an exact int rather than passing through a
            #float, so a billion-instruction pin never rounds. `<not counted>`
            #and `<not supported>` land in the handler rather than reading as
            #a zero that would gate nothing.
            sample[event] = int(values[0]) if values[0].isdigit() else float(values[0])
        except ValueError as error:
            #`<not counted>` and `<not supported>` are perf's own words for a
            #counter that never armed, so this is the box refusing rather than
            #the workload answering: nothing was measured and nothing here can
            #say the tree moved.
            msg = (
                f"perf answered {values[0]!r} for {event} rather than a count, "
                f"so its counter never armed. {PARANOID} reads "
                f"{_paranoid_reading()}, where 2 or less is needed, and a "
                "container needs --security-opt seccomp=unconfined before "
                f"perf_event_open is permitted at all. perf said: "
                f"{' '.join(stderr.split())[-300:]}"
            )
            raise MeasurementRefusedError(msg) from error
    return sample


@dataclass(frozen=True)
class CounterRuns:
    """What repeated runs of one command counted, and what each printed.

    `events` maps a perf event name to one value per run in run order, and
    `outputs` is each run's standard output, which is how a workload reports a
    counter perf cannot see -- the engine's own inferences -- from inside the
    very run that was counted.
    """

    events: Mapping[str, tuple[float, ...]]
    outputs: tuple[str, ...]


def _measurement_environment() -> dict[str, str]:
    """The environment every measured child gets, BUILT rather than inherited.

    Two measured reasons. PYTHONHASHSEED pinned: per-launch hash randomization
    moves a dict-heavy workload's retired-instruction count by more than the
    gate's whole noise allowance (json-wire spread 1.46% across four launches,
    0.098% with the seed pinned [measured 2026-08-17]), and a security feature
    has no place in a reproducibility harness. The allowlist: the SIZE of the
    environment block moves where the process heap starts, which selects how
    many times the engine's global stack grows mid-measurement; source-load
    measured a stable 957.6M instructions under check.sh's environment against
    a stable low mode under a bare shell, three samples each within 0.002%,
    inference counter identical [measured 2026-08-17]. A fixed environment
    makes the measurement caller-independent without touching the engine's own
    stack economics (presizing stacks instead cost save-load-metta +2.35%).
    """
    return {
        name: os.environ[name]
        for name in ("PATH", "HOME", "LD_LIBRARY_PATH", "SWI_HOME_DIR")
        if name in os.environ
    } | {"LC_ALL": "C", "PYTHONHASHSEED": "0"}


def measure_counters(
    command: Sequence[str],
    *,
    events: Sequence[str] = _DEFAULT_EVENTS,
    rounds: int = _COUNTER_SAMPLES,
    controlled: bool = False,
    timeout: float = 60.0,
) -> CounterRuns:
    """Run command under perf stat and return each run's counters and output."""
    timeout = _counter_request(command, events, rounds, timeout)
    environment = _measurement_environment()
    collected: dict[str, list[float]] = {event: [] for event in events}
    outputs: list[str] = []
    for _ in range(rounds):
        returncode, stdout, stderr = _run_perf(
            command,
            environment,
            controlled=controlled,
            timeout=timeout,
            events=events,
        )
        for event, value in _parse_counter_sample(
            returncode, stdout, stderr, events
        ).items():
            collected[event].append(value)
        outputs.append(stdout)
    return CounterRuns(
        {event: tuple(values) for event, values in collected.items()},
        tuple(outputs),
    )


def measure_instructions(
    command: Sequence[str],
    *,
    rounds: int = _COUNTER_SAMPLES,
    controlled: bool = False,
    timeout: float = 60.0,
) -> tuple[int, ...]:
    """Run command under perf stat and return retired instructions per run."""
    runs = measure_counters(
        command, rounds=rounds, controlled=controlled, timeout=timeout
    )
    return tuple(int(value) for value in runs.events["instructions:u"])


def _workload(command: Sequence[str], environment: Mapping[str, str]) -> list[str]:
    """The measured command, its executable resolved against the measurement PATH.

    A counting tool starts the workload itself, and perf puts its own
    directories AHEAD of the PATH it was handed before it does
    (tools/perf/util/exec-cmd.c, setup_path: perf's exec path, then the
    directory perf ran from, then PATH), so a bare name resolves in /usr/bin
    whatever the measurement environment says. Every `swipl` the engine
    benchmark counted therefore ran the stock build rather than the patched one
    first on PATH, silently until the engine's host check began refusing that
    pairing [measured 2026-09-24: `perf stat -- sh -c 'echo $PATH'` printed
    /usr/libexec/perf-core:/usr/bin: ahead of the PATH it was given, and
    `perf stat -- swipl` reported the stock build's compiled_at with the
    patched swipl first on PATH]. Resolved here, once, for every tool, the
    workload is the one the measurement environment names; a name that PATH
    does not hold is refused rather than left for the tool to find elsewhere.
    A name with a directory part, such as a driver started as `./cases`, is
    the file it names: shutil.which checks such a name where it points and
    never searches PATH for it.
    """
    resolved = shutil.which(command[0], path=environment.get("PATH", os.defpath))
    if resolved is None:
        where = ("names no executable file" if os.sep in command[0]
                 else "is not on the measurement environment's PATH")
        msg = f"{command[0]} {where}"
        raise FileNotFoundError(msg)
    return [resolved, *command[1:]]


def _run_perf(
    command: Sequence[str],
    environment: Mapping[str, str],
    *,
    controlled: bool,
    timeout: float,
    events: Sequence[str] = _DEFAULT_EVENTS,
) -> tuple[int, str, str]:
    """Find perf and run it without a shell, capturing both output streams."""
    executable = shutil.which("perf")
    if executable is None:
        msg = f"perf is required to measure {', '.join(events)}"
        raise FileNotFoundError(msg)
    if not os.access("/usr/bin/setarch", os.X_OK):
        msg = f"setarch is required to measure {', '.join(events)} reproducibly"
        raise FileNotFoundError(msg)
    if sys.platform != "linux":
        # Refused here with the reason, rather than reached and raising
        # AttributeError on os.POSIX_SPAWN_DUP2. `perf stat` measures
        # through perf_event_open, a Linux syscall, so this path is
        # Linux-only by WHAT IT MEASURES and not by how it spawns: making
        # the spawn portable would leave nothing to run.
        msg = (f"instruction counting needs `perf stat`, which measures "
               f"through perf_event_open, a Linux facility; this is "
               f"{sys.platform}. Wall-clock and inference counts work "
               f"everywhere, so gate on inferences here.")
        raise RuntimeError(msg)
    child_environment = dict(environment)
    control_descriptors: tuple[int, ...] = ()
    control_arguments: list[str] = []
    if controlled:
        control_read, control_write = os.pipe()
        acknowledge_read, acknowledge_write = os.pipe()
        control_descriptors = (
            control_read,
            control_write,
            acknowledge_read,
            acknowledge_write,
        )
        for descriptor in control_descriptors:
            os.set_inheritable(descriptor, True)  # noqa: FBT003  -- os.set_inheritable is positional-only and the literal states the requested descriptor state
        child_environment.update(
            {
                "METTA_PERF_CONTROL_FD": str(control_write),
                "METTA_PERF_ACK_FD": str(acknowledge_read),
                "METTA_PERF_CLOSE_FDS": f"{control_read},{acknowledge_write}",
            }
        )
        control_arguments = [
            "--delay=-1",
            f"--control=fd:{control_read},{acknowledge_write}",
        ]
    #setarch -R disables address-space randomization for the child tree:
    #with the environment and hash seed already pinned, the residual
    #spread (json-wire 0.3% across a triple) tracks the kernel moving
    #the heap and stack bases per launch, which selects the same
    #alignment modes the environment block does. ASLR is the third
    #security feature with no place in a reproducibility harness.
    event_arguments = [word for event in events for word in ("-e", event)]
    argv = [
        "/usr/bin/setarch",
        "-R",
        executable,
        "stat",
        "-x,",
        *event_arguments,
        *control_arguments,
        "--",
        *_workload(command, child_environment),
    ]
    return _spawn_and_reap(
        argv,
        child_environment,
        timeout=timeout,
        what="perf stat",
        inherited=control_descriptors,
    )


def _spawn_and_reap(
    argv: Sequence[str],
    environment: Mapping[str, str],
    *,
    timeout: float,
    what: str,
    inherited: Sequence[int] = (),
) -> tuple[int, str, str]:
    """Run argv in a process group of its own and return its status and both streams.

    `inherited` are descriptors the child was meant to receive, closed here
    once it has them. On a timeout or an interruption the whole group is
    killed and the child reaped, so no measured process outlives the harness
    that started it. That needs POSIX process groups, so Windows is refused
    before anything spawns.
    """
    if sys.platform == "win32":
        # Both callers refuse first, for what they measure: perf and
        # Cachegrind under setarch run on Linux alone. This is the spawner's
        # own requirement, stated where it is used, so a later caller meets
        # a reason rather than an AttributeError on os.posix_spawn.
        msg = (f"{what} runs in a process group of its own so a timeout can "
               f"kill it whole, and Windows has no process groups; counted "
               f"measurement needs Linux. Wall-clock and inference counts "
               f"work everywhere.")
        raise RuntimeError(msg)
    with (
        tempfile.TemporaryFile() as stdout,
        tempfile.TemporaryFile() as stderr,
    ):
        file_actions = [
            (os.POSIX_SPAWN_DUP2, stdout.fileno(), 1),
            (os.POSIX_SPAWN_DUP2, stderr.fileno(), 2),
        ]
        try:
            process = os.posix_spawn(
                argv[0],
                list(argv),
                dict(environment),
                file_actions=file_actions,
                setpgroup=0,
            )
        finally:
            for descriptor in inherited:
                os.close(descriptor)
        deadline = time.monotonic() + timeout
        try:
            while True:
                try:
                    finished, status = os.waitpid(process, os.WNOHANG)
                except InterruptedError:
                    continue
                if finished:
                    break
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    msg = f"{what} exceeded its {timeout:g} second limit"
                    raise TimeoutError(msg)  # noqa: TRY301  -- the raise stays inside this rollback boundary so the same handler records the failure
                time.sleep(min(0.01, remaining))
        except BaseException:
            with suppress(ProcessLookupError):
                os.killpg(process, signal.SIGKILL)
            with suppress(ChildProcessError):
                os.waitpid(process, 0)
            raise
        stdout.seek(0)
        stderr.seek(0)
        return (
            os.waitstatus_to_exitcode(status),
            stdout.read().decode(errors="replace"),
            stderr.read().decode(errors="replace"),
        )


def estimated_cycles(counts: Mapping[str, int]) -> int:
    """One run's cost in cycles on the simulated hierarchy, from its nine counts.

    Every instruction fetch and every data access is priced by the level that
    served it: one the last level missed went to memory and costs thirty-five,
    one the first level missed and the last level caught costs five, and the
    rest hit the first level and cost one. The weights are Itamar
    Turner-Trauring's, which iai-callgrind adopts as its Estimated Cycles
    [source: https://github.com/pythonspeed/cachegrind-benchmarking/blob/bc45f72bf7b4656d723104fe95a318bbd34dda7b/cachegrind.py,
    get_counts and combined_instruction_estimate].
    Time: O(1).
    """
    memory = counts["ILmr"] + counts["DLmr"] + counts["DLmw"]
    last_level = counts["I1mr"] + counts["D1mr"] + counts["D1mw"] - memory
    accesses = counts["Ir"] + counts["Dr"] + counts["Dw"]
    return (accesses - last_level - memory) + 5 * last_level + 35 * memory


def measure_simulated(
    command: Sequence[str],
    *,
    rounds: int = _COUNTER_SAMPLES,
    controlled: bool = False,
    timeout: float = 600.0,
) -> CounterRuns:
    """Run command under Cachegrind `rounds` times at once; return each run's counts and output.

    `events` maps each of Cachegrind's nine cache events to one count per run,
    in run order, and estimated_cycles turns one run's counts into the number a
    row pins. The runs go concurrently because nothing another process does
    can move a simulated count, which is the reason to simulate at all, and a
    simulated run costs twenty to fifty times the native one. A `controlled`
    run counts only between the workload's own
    CACHEGRIND_START_INSTRUMENTATION and CACHEGRIND_STOP_INSTRUMENTATION
    client requests, the window perf's control descriptors open for
    measure_counters.
    """
    timeout = _counter_request(command, _CACHEGRIND_EVENTS, rounds, timeout)
    executable = shutil.which("valgrind")
    if executable is None:
        msg = "valgrind is required to measure estimated cycles"
        raise FileNotFoundError(msg)
    if not os.access("/usr/bin/setarch", os.X_OK):
        msg = "setarch is required to measure estimated cycles reproducibly"
        raise FileNotFoundError(msg)
    environment = _measurement_environment()

    def run(_round: int) -> tuple[dict[str, int], str]:
        return _run_cachegrind(
            executable, command, environment, controlled=controlled, timeout=timeout
        )

    with ThreadPoolExecutor(max_workers=rounds) as pool:
        runs = list(pool.map(run, range(rounds)))
    return CounterRuns(
        {event: tuple(counts[event] for counts, _ in runs) for event in _CACHEGRIND_EVENTS},
        tuple(output for _, output in runs),
    )


def _run_cachegrind(
    executable: str,
    command: Sequence[str],
    environment: Mapping[str, str],
    *,
    controlled: bool,
    timeout: float,
) -> tuple[dict[str, int], str]:
    """One Cachegrind run: its nine summary counts and the workload's standard output."""
    with tempfile.TemporaryDirectory() as scratch:
        output = Path(scratch) / "cachegrind.out"
        #setarch -R for the reason _run_perf gives: the program's own
        #address-dependent choices are part of what is simulated.
        argv = [
            "/usr/bin/setarch",
            "-R",
            executable,
            "--tool=cachegrind",
            "--cache-sim=yes",
            *_SIMULATED_CACHES,
            f"--instr-at-start={'no' if controlled else 'yes'}",
            f"--cachegrind-out-file={output}",
            *_workload(command, environment),
        ]
        returncode, stdout, stderr = _spawn_and_reap(
            argv, environment, timeout=timeout, what="cachegrind"
        )
        if returncode != 0:
            msg = (
                f"{command[0]} failed under cachegrind with exit {returncode}: "
                f"{' '.join(stderr.split())[-300:]}"
            )
            raise RuntimeError(msg)
        counts = _parse_cachegrind(output.read_text(encoding="utf-8", errors="replace"))
    if controlled and counts["Ir"] == 0:
        #A build defect, not the box refusing: the workload ran to completion
        #and simply never asked for the window, which is what it does when it
        #was compiled without <valgrind/cachegrind.h>.
        msg = (
            f"the simulated window never opened: {command[0]} ran but issued no "
            "CACHEGRIND_START_INSTRUMENTATION, so it was built without "
            "<valgrind/cachegrind.h> on its include path"
        )
        raise RuntimeError(msg)
    return counts, stdout


def _parse_cachegrind(text: str) -> dict[str, int]:
    """The nine summary counts of one Cachegrind output file, by event name."""
    header: list[str] | None = None
    summary: list[str] | None = None
    for line in text.splitlines():
        if line.startswith("events:"):
            header = line.split()[1:]
        elif line.startswith("summary:"):
            summary = line.split()[1:]
    if header is None or summary is None or len(header) != len(summary):
        msg = f"cachegrind wrote no matching events and summary lines: {text[:300]!r}"
        raise RuntimeError(msg)
    counts = dict(zip(header, (int(value) for value in summary), strict=True))
    missing = [event for event in _CACHEGRIND_EVENTS if event not in counts]
    if missing:
        msg = (
            f"cachegrind counted {header} without {missing}, so its cache "
            "simulation did not run"
        )
        raise RuntimeError(msg)
    return {event: counts[event] for event in _CACHEGRIND_EVENTS}

__all__ = [
    "ESTIMATED_CYCLES",
    "GOVERNED_ARTIFACT_MARKER",
    "INSTRUCTIONS",
    "PERF_CONTROL_REFUSED",
    "BenchmarkBaseline",
    "CounterRuns",
    "MeasurementRefusedError",
    "Metric",
    "benchmark_case",
    "benchmark_counter_slope",
    "count_atoms",
    "estimated_cycles",
    "measure_counters",
    "measure_instructions",
    "measure_simulated",
    "measured_main",
    "prepare_governed_artifacts",
    "refusal_is_fatal",
]
