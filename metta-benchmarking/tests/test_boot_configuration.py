"""Purpose: keep location-sensitive boot pins separate from runtime comparisons.

Guarantees: both boot counters decline a different checkout shape, including
updates, while comparable boot and runtime regressions still fail
[tested: test_boot_path_refuses_both_counters_and_preserves_pins,
test_comparable_counters_still_gate; commit=WORKTREE].
"""

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[5]
CANONICAL = Path('/aaaa/bbbb/cccc/dddd/eeeeeeee')


def load_driver(seat, monkeypatch):
    """Load a driver's real entry point without invoking its command line."""
    source = ROOT / ('engine/bench.py' if seat == 'engine'
                     else 'extensions/cmetta/benchmarks/bench.py')
    spec = importlib.util.spec_from_file_location(f'boot_configuration_{seat}', source)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(params=['engine', 'c'])
def driver(request, monkeypatch, tmp_path):
    """Run each real driver with deterministic samples and a private baseline."""
    seat = request.param
    module = load_driver(seat, monkeypatch)
    baseline = tmp_path / 'baseline.json'
    baseline.write_text(json.dumps({
        'schema': 1, 'counter_configuration': {},
        'measurement': {'checkout_path_length': 29, 'checkout_path_depth': 5,
                        'checkout_path_reason': 'non-monotonic inventory sensitivity'},
        'benchmarks': {name: {'unit': 'boot', 'operations': 1, 'inferences': 100,
                              'boot_qlf_count': 21,
                              'instructions': 1000, 'cpu_seconds': 0.01,
                              'instruction_noise_percent': 1.0,
                              'cpu_noise_percent': 10.0}
                       for name in ('boot', 'work')},
    }))
    counts = {'boot': 100, 'work': 100}
    monkeypatch.setattr(module, 'BASELINE', baseline)
    monkeypatch.setattr(module, 'ROOT', CANONICAL)
    monkeypatch.delenv('CI', raising=False)
    if seat == 'engine':
        cases = {name: module.Case('boot', 1, name == 'boot') for name in counts}
        monkeypatch.setattr(module, 'describe', lambda: (cases, ()))
        monkeypatch.setattr(module, 'stamp', lambda _sources: {})
        monkeypatch.setattr(module, '_run', lambda _goal: '')
        monkeypatch.setattr(module, 'counter_samples',
                            lambda name: ([counts[name]] * 3, 0.01, 0.01))
        monkeypatch.setattr(module, 'instruction_samples',
                            lambda name: (counts[name] * 10,) * 3)
        update_flag = '--update-baseline'
    else:
        cases = tuple(module.Case(name, 'boot', 1, name == 'boot') for name in counts)
        monkeypatch.setattr(module, 'CASES', cases)
        monkeypatch.setattr(module, 'BY_NAME', {case.name: case for case in cases})
        monkeypatch.setattr(module, 'DRIVER', Path(__file__))
        monkeypatch.setattr(module, 'counter_configuration', lambda: {})
        monkeypatch.setattr(module, 'anchor', lambda: None)
        monkeypatch.setattr(module, 'warm', lambda: None)
        monkeypatch.setattr(module.subprocess, 'run', lambda *_args, **_kwargs:
                            SimpleNamespace(stdout='boot-qlf-count 21\n'))
        monkeypatch.setattr(module, 'time_is_measurable', lambda: True)
        monkeypatch.setattr(module, 'seats_differing_from_head', lambda: [])
        monkeypatch.setattr(module, 'sample', lambda case, _rounds:
                            ((counts[case.name] * 10,) * 3, (0.01,) * 3,
                             (counts[case.name],) * 3))
        update_flag = '--update'
    return SimpleNamespace(module=module, counts=counts, baseline=baseline,
                           update_flag=update_flag)


@pytest.mark.parametrize('checkout', [
    Path('/aaaa/bbbb/cccc/eeeeeeeeeeeee'),
    Path('/aaaa/bbbb/cccc/dddd/eeeeeeeee'),
    Path('/aaaa/bbbb/cccc/dddd/eee/eeee'),
])
@pytest.mark.parametrize('update', [False, True])
def test_boot_path_refuses_both_counters_and_preserves_pins(
    driver, checkout, update, monkeypatch, capsys,
):
    """Different length or depth is explicit even when the samples match."""
    monkeypatch.setattr(driver.module, 'ROOT', checkout)
    if update:
        driver.counts['boot'] = 200
    arguments = ['boot', 'work'] + ([driver.update_flag] if update else [])
    assert driver.module.main(arguments) == 0
    output = capsys.readouterr()
    reported = output.out + output.err
    assert 'NOT MEASURED IN THIS CONFIGURATION' in reported
    assert f'length {len(str(checkout))}, depth {len(checkout.parts) - 1}' in reported
    assert 'canonical length 29, depth 5' in reported
    assert 'non-monotonic inventory sensitivity' in reported
    pinned = json.loads(driver.baseline.read_text())['benchmarks']['boot']
    assert (pinned['inferences'], pinned['instructions']) == (100, 1000)


@pytest.mark.parametrize('moved', ['boot', 'work'])
def test_comparable_counters_still_gate(driver, moved, monkeypatch, capsys):
    """The boot refusal leaves every runtime row's counter comparison active."""
    driver.counts[moved] = 200
    if moved == 'work':
        monkeypatch.setattr(driver.module, 'ROOT', CANONICAL / 'another')
    assert driver.module.main(['boot', 'work']) == 1
    output = capsys.readouterr()
    assert moved in output.out + output.err


@pytest.mark.parametrize('count', [20, 21, 22])
def test_c_boot_normalises_the_governed_cache_set(count, monkeypatch, tmp_path):
    """The real boot purge removes optional caches before the ordinary warm."""
    module = load_driver('c', monkeypatch)
    monkeypatch.setattr(module, 'ROOT', tmp_path)
    library = tmp_path / 'lib' / 'optional'
    library.mkdir(parents=True)
    cache = library / 'optional.qlf'
    cache.write_bytes(b'ignored cache')
    source = library / 'optional.pl'
    source.write_text('% source stays\n')
    engine = tmp_path / 'engine'
    engine.mkdir()
    engine_cache = engine / 'metta.qlf'
    engine_cache.write_bytes(b'engine cache')
    observer_cache = engine / 'source_observation.qlf'
    observer_cache.write_bytes(b'optional engine cache')
    (engine / 'qlf_boot.pl').write_bytes((ROOT / 'engine/qlf_boot.pl').read_bytes())
    ungoverned = tmp_path / 'outside.qlf'
    ungoverned.write_bytes(b'not governed by this boot')
    run = module.subprocess.run

    def warm_engine(argv, **kwargs):
        if 'metta_qlf_boot:purge_all_qlf' in argv:
            return run(argv, **kwargs)
        assert not cache.exists()
        assert not engine_cache.exists() and not observer_cache.exists()
        assert source.is_file() and ungoverned.is_file()
        assert str(engine / 'bench.pl') in argv
        assert 'metta_bench:bench_run(boot)' in argv[argv.index('-g') + 1]
        return SimpleNamespace(stdout=f'boot-qlf-count {count}\n')

    monkeypatch.setattr(module.subprocess, 'run', warm_engine)
    if count == 21:
        module.prepare_boot(21)
    else:
        with pytest.raises(AssertionError, match=f'governed QLF inventory {count}; pinned 21'):
            module.prepare_boot(21)


@pytest.mark.parametrize('update', [False, True])
def test_c_inventory_failure_is_fatal_and_runtime_still_compares(
    update, monkeypatch, tmp_path, capsys,
):
    """A changed engine artifact set cannot silently decline or rewrite boot."""
    module = load_driver('c', monkeypatch)
    baseline_path = tmp_path / 'baseline.json'
    baseline_path.write_text(json.dumps({'schema': 1, 'benchmarks': {
        'boot': {'boot_qlf_count': 21, 'unit': 'boot', 'operations': 1,
                 'inferences': 100, 'instructions': 1000, 'instruction_noise_percent': 1,
                 'cpu_seconds': 0.01, 'cpu_noise_percent': 10},
        'work': {'unit': 'work', 'operations': 1, 'inferences': 100,
                 'instructions': 1000, 'instruction_noise_percent': 1,
                 'cpu_seconds': 0.01, 'cpu_noise_percent': 10},
    }}))
    baseline = module.BenchmarkBaseline(baseline_path, update=update)
    monkeypatch.setattr(module, 'ROOT', tmp_path)
    monkeypatch.setattr(module, 'time_is_measurable', lambda: True)
    monkeypatch.setattr(module.subprocess, 'run', lambda *_args, **_kwargs:
                        SimpleNamespace(stdout='boot-qlf-count 22\n'))
    sampled = []

    def sample(case, rounds):
        sampled.append(case.name)
        return ((1000,) * rounds, (0.01,) * rounds, (100,) * rounds)

    monkeypatch.setattr(module, 'sample', sample)
    failures, refused = module.observe_all(baseline, (
        module.Case('boot', 'boot', 1, whole_process=True),
        module.Case('work', 'work', 1),
    ), 3)
    baseline.finish()
    assert len(failures) == 1 and 'governed QLF inventory 22; pinned 21' in failures[0]
    assert not refused
    assert sampled == ['work']
    assert 'REFUSED' in capsys.readouterr().out
    assert json.loads(baseline_path.read_text())['benchmarks']['boot']['inferences'] == 100
