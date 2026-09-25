# Purpose: the integration distributions' own gate lanes, in the root gate's vocabulary.
# Assumes: the root check.sh supplies the runner when sourcing this file, so `run`,
#   `$HERE`, `$PY` and the shared summary table are already defined. Direct invocation
#   delegates to that runner with this component's lane names.
# Guarantees: every path a lane runs is written literally, because
#   tests/checks/evidence_runners.py models which files a lane covers by READING this
#   text and resolving `$HERE/`.
# Guarantees: these lanes are this component's, not the Python seat's. They used to be
#   three arguments on the seat's own ruff, bandit and interrogate lines, which made the
#   seat responsible for linting the distributions that exist to show a seat does not
#   need them [tested: sh check.sh ext-ruff ext-bandit ext-interrogate; commit=WORKTREE].
# Open Obligations:
#   To Do: None
#   Hacks: None
#   Future Enhancements: None

if ! command -v run >/dev/null 2>&1; then
    EXT_HERE=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
    exec sh "$EXT_HERE/../tools/check.sh" --component ext "$@"
fi

# Paths are resolved through $HERE rather than left relative, because a lane must not
# depend on which directory the gate happened to be in when it sourced this file, and
# because the evidence gate models a lane's coverage by resolving exactly that prefix.
# ruff walks up from each file for its configuration, so these read the repository's
# own pyproject.toml exactly as the seat's files do. bandit and interrogate do not
# walk, so each is HANDED that same file: there is one house style, and a component
# measured against a different one is reporting about the wrong rules rather than
# about itself.
#
# interrogate was the one left without it, and read its own defaults instead of
# `[tool.interrogate]`. Those defaults keep `tests/`, magic methods, private names,
# nested functions and overloads in scope, none of which the house style asks a
# docstring of, so the lane reported 64.9% against a threshold of 80 while the same
# tree under the same threshold reads 84.0% [measured 2026-09-20]. The 333 it called
# missing were mostly test functions, whose long declarative names are what this
# repository documents them with.
run GATE   ext-ruff        bounded "$PY" -m ruff check "$HERE/ext"
run GATE   ext-bandit      bounded "$PY" -m bandit -q -c "$HERE/pyproject.toml" -r "$HERE/ext"
run GATE   ext-interrogate bounded "$PY" -m interrogate -c "$HERE/pyproject.toml" "$HERE/ext"

# These lanes belong to this component even when their checkers live in tests/checks.
# The host wheel is built from a swipl-devel tree this repository fetches and
# patches itself, and two of the nineteen patches target a SUBMODULE of it.
# A router that tried only the superproject would apply seventeen and refuse
# two, which builds a host missing fixes rather than failing. This plants its
# own trees, so it needs no clone and no network.
# Owner: ext; this lane checks only this component.
run GATE fetch-source-selftest sh "$HERE/tools/pymetta-host/fetch_selftest.sh"

# The host build skips its two compile stages when OUT already holds their
# output for exactly this tree's inputs, which is how a release reuses a host
# built earlier the same day. A key missing one input would let a home built
# from other patches or other cmake arguments into a published wheel, and a
# record left by a half-finished build would vouch for a half-built OUT. This
# drives run.sh with docker and fetch-source.sh stubbed, so it needs neither.
# Owner: ext; this lane checks only this component.
run GATE host-reuse-selftest sh "$HERE/tools/pymetta-host/run_selftest.sh"

# And the lane above must be able to FAIL: each rule run.sh keeps is taken
# away in turn, one mutant per rule beside an unmutated control, and the
# self-test has to go red for every one, as evidence-mutations does for the
# evidence gate.
# Owner: ext; this lane checks only this component.
run GATE host-reuse-mutants "$PY" "$HERE/tests/checks/check_host_reuse_mutations.py"

# A wheel cannot carry a symlink, so every alias in a staged tree reaches the
# user as a full copy, and a copy of a position-dependent ELF carries an RPATH
# for a directory it is no longer in. That shipped once: the host wheel then
# built as its own distribution imported cleanly, answered 6*7=42 and reported
# SWI 10.1.14 in the same install whose bin/swipl could not start and whose
# libswipl borrowed libgmp from the host. An import-level test sees none of
# it, so this reads the dynamic section of every shipped ELF instead, in the
# pymetta manylinux wheels tools/pymetta-host/run.sh grafts the host into.
# They are a build artefact rather than a repository one, so with none present
# the lane is vacuously true and assemble.sh is what gates the ones it builds.
# Owner: ext; this lane checks only this component.
run GATE host-bundle "$PY" "$HERE/tests/checks/check_host_bundle.py" \
    $(ls "$HERE"/ai-tmp/host-build/dist/pymetta-*-manylinux*.whl 2>/dev/null)
# Owner: ext; this lane checks only this component.
run GATE host-bundle-selftest "$PY" "$HERE/tests/checks/check_host_bundle_selftest.py"

# A wheel that passes the lane above can still rewrite itself on its first
# boot: SWI-Prolog 10.1.14 recompiled a bundled library .qlf whenever its .pl
# read newer, and a wheel carries no times an installer honours, so a fresh uv
# install of pymetta 0.9.2 rewrote up to 51 of them inside site-packages. This
# installs each wheel at two paths with every bundled .qlf dated before its
# source, boots it, and requires every installed file untouched. pip runs with
# --no-index from the wheelhouse tools/pymetta-host/assemble.sh fills beside
# the wheels, so the lane reaches no network; with no wheel built it measured
# nothing and says so with 125.
# Owner: ext; this lane checks only this component.
run GATE wheel-first-boot "$PY" "$HERE/tests/checks/check_wheel_first_boot.py" \
    --find-links "$HERE/ai-tmp/host-build/wheelhouse" \
    $(ls "$HERE"/ai-tmp/host-build/dist/pymetta-*-manylinux*.whl 2>/dev/null)
# Owner: ext; this lane checks only this component.
run GATE wheel-first-boot-selftest "$PY" "$HERE/tests/checks/check_wheel_first_boot_selftest.py"
