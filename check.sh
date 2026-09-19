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
    if [ "$#" -eq 0 ]; then
        set -- ext-ruff ext-bandit ext-interrogate
    fi
    exec sh "$EXT_HERE/../check.sh" "$@"
fi

# Paths are resolved through $HERE rather than left relative, because a lane must not
# depend on which directory the gate happened to be in when it sourced this file, and
# because the evidence gate models a lane's coverage by resolving exactly that prefix.
# ruff walks up from each file for its configuration, so these read the repository's
# own pyproject.toml exactly as the seat's files do.
run GATE   ext-ruff        bounded "$PY" -m ruff check "$HERE/ext"
run GATE   ext-bandit      bounded "$PY" -m bandit -q -c "$HERE/pyproject.toml" -r "$HERE/ext"
run GATE   ext-interrogate bounded "$PY" -m interrogate "$HERE/ext"
