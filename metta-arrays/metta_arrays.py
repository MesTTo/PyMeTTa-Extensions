"""Purpose: arrays as atoms for every library speaking the standard
protocols, not one. Recognition is DLPack (__dlpack__), semantics are the
Python array API standard reached through array-api-compat, so one operation
set serves NumPy, PyTorch, CuPy, JAX, Dask and whatever conforms next, and a
mixed-library call converts through from_dlpack. Arrays cross the boundary
by reference with identity, DLTensor joins each array's own classes as a
type, and printing shows shape, dtype and device whatever the library.

This is a DISTRIBUTION of its own, `metta-arrays`, and it names no array
library: which one is the default comes from a row against the seat's `array`
point, so `metta-numpy` is one package among the ones a user may install and
`install(m, default=<module>)` already takes any module the standard covers.
It reaches pymetta through public names and the seam's published services and
through nothing private, which is what makes it the same kind of thing as a
stranger's package.
Guarantees:
  - _top_indices returns the highest finite scores first and resolves equal
    scores by insertion order [tested test_top_indices_match_full_order_and_stabilize_ties]
  - _top_indices uses 35.26% fewer instructions than the prior full sort for
    500 top-10 selections from 100,000 scores [measured 2026-08-14: minimum
    of three perf stat instructions:u runs]
  - the fixed public constructor vocabulary is marked Final to type
    checkers [tested test_policy_constants_are_final]
  - all 44 installed operation names own arity-accurate arrows, and
    broadcast-shape relates compatible dimensions before any array exists
    [tested: test_every_array_operation_is_typed_and_a_shape_is_a_constraint;
     commit=f88aa8be03cb64cb59d3307515ded8701f418321]
  - array transport and Atom-delivery choices use the same declaration
    surface as every registered operation [tested:
    test_no_decorator_flag_changes_the_return_shape_and_declarations_are_atoms;
    commit=f88aa8be03cb64cb59d3307515ded8701f418321]
  - operations returning mutable arrays are writesState, scalar inspections
    are readOnlyLookup, random construction is oracleIO, and embedding search
    is nondeterministicReadOnly [tested:
    test_every_array_operation_is_typed_and_a_shape_is_a_constraint,
    test_embedding_store_runs_on_numpy;
    commit=3cfbe0d7417b1c453c2dc12d47e2e47e7de461f7]
  - constructor registrations use the backend namespace's fully qualified
    module name, so installing ``jax.numpy`` and then ``numpy`` cannot
    retarget the first space; random construction never borrows a different
    backend's hidden random state [tested:
    test_nested_backend_names_do_not_retarget_an_earlier_space,
    test_randn_never_borrows_another_backends_random_state;
    commit=de15573db164c24b9dcaa3e5b783e66dcb05d4d1]
  - a mixed call converts on the LIBRARY the operand belongs to and not on
    its Python class, so two classes of one namespace, a JAX tracer beside a
    concrete array among them, cross untouched [tested:
    test_an_operand_of_the_same_library_is_not_converted_through_dlpack,
    test_a_jax_tracer_crosses_a_binary_op_and_a_gradient_reaches_it]
  - install() and EmbeddingStore take a context or a space and register into
    the space either way [tested: test_install_takes_a_context_as_well_as_a_space,
    test_embedding_store_takes_a_context_as_well_as_a_space; commit=f25ac80f93e7c3626b87e593117d09b9c9bc8c95]
  - Shape metadata survives Python ``Annotated`` reflection, shaped tensors
    remain valid ``DLTensor`` arguments, broadcast arithmetic infers its
    output through ``broadcast-shape``, and rank-two ``matmul`` unifies its
    shared dimension before materialisation [tested:
    test_annotated_tensor_shapes_flow_through_broadcast_and_matmul;
    commit=4eaefdd8d40e53b2613722287302a14b41704662]
  - declared and observed shapes use one Annotated type expression; every
    registered head names its shape behavior and preserving arrows share their
    input shape [tested: test_declared_shape_variables_derive_the_result_without_execution,
    test_every_preserving_unary_head_keeps_symbolic_and_live_shapes;
    commit=4eaefdd8d40e53b2613722287302a14b41704662]
  - what a space installed is a property of THAT space, one
    (array-backend <space> <library> (ops ...)) row in &metta that ops() and
    backend() read back, so two spaces on two libraries answer their own
    rosters and constructor types in either install order, a second install
    replaces the row with the aliases and the operations it alone named, and
    dropping the space retires the row through the catalog's space-ownership
    walk [tested: test_a_space_answers_its_own_roster_in_either_install_order,
    test_a_second_install_replaces_the_roster_and_its_operations,
    test_dropping_the_space_retires_its_installation_row;
    commit=76dbea9f4bc10804a5ca19493972dfb7975bc4b0]
  - uninstall() is install's inverse and keeps every operation another
    space's row still claims, the registry being process-wide by name; a
    space with no row, and a space with two, both refuse by name
    [tested: test_uninstall_retires_the_installation_and_keeps_shared_operations,
    test_the_roster_doors_refuse_an_uninstalled_space_and_a_doubled_row;
    commit=76dbea9f4bc10804a5ca19493972dfb7975bc4b0]
Guarded by:
  - _PROTOCOLS_LOCK serializes one-time protocol registration
    [tested test_array_protocol_registration_is_idempotent]
  - _ROSTER_KIND_LOCK serializes the once-per-catalog declaration of the
    (array-backend ...) kind and its ownership marker, which the engine
    refuses a second time for one head. It guards the one piece of install
    state two SPACES share; two threads installing into ONE space race on
    that space's own atoms and are a caller error either way
    [assumed: no test installs from two threads at once, so the race is
    reasoned from the engine's one-kind-row-per-head refusal rather than
    reproduced; commit=76dbea9f4bc10804a5ca19493972dfb7975bc4b0]
  - the Array API index backend is a FALLBACK row, so a library's own
    nearest-neighbour backend wins `backend="auto"` whatever order the two
    distributions loaded in [tested:
    ext/metta-faiss/tests/test_faiss.py::test_faiss_wins_auto_over_the_array_api_fallback;
    commit=WORKTREE]
Open Obligations:
  To Do: None
  Hacks: None
  Future Enhancements: None.
"""  # noqa: D205  -- the API contract is one continuous invariant, not summary-and-body prose

from __future__ import annotations

import importlib
import inspect
import itertools
import operator
import threading
from collections.abc import Iterable
from functools import wraps
from typing import Annotated, Any, Final, Literal, NewType, cast

from metta import Space, seam
from metta import integrate as _integrate
from metta import ops as _ops
from metta.atoms import Atom, Expression, Grounded, S, Symbol, V, Variable, ground
from metta.errors import MettaError
from metta.wire import decode as _decode

#: Bound once, because these are per-call on the hot paths and a service lookup
#: is two dictionary reads. `alpha-eq` is MeTTa's =alpha and `module` is the
#: optional import with the caller's own guidance, both published by the seam
#: for exactly this: a registrant calls a service instead of importing a
#: private module.
_alpha_eq = seam.at("alpha-eq").call()
require_module = seam.at("module").call()


def _expr(*children: Any) -> Expression:
    """A variadic Expression: `Expression` encodes its children identically."""
    return Expression(list(children))

__all__ = [
    "SHAPE_RULES",
    "DLTensor",
    "EmbeddingStore",
    "Shape",
    "backend",
    "data_of",
    "install",
    "is_array",
    "namespace_of",
    "ops",
    "uninstall",
]

#: The catalog head one install writes, and the shape of its row:
#: ``(array-backend <space> <library> (ops <name> ...))``. The roster is a
#: property of the SPACE because that is where the install happened: two
#: spaces on two libraries answer their own rosters in either order, and the
#: row dies with the space because ``install`` marks the head
#: ``(owned-by-space array-backend)`` in the same catalog.
_ROSTER_HEAD: Final[str] = "array-backend"
_ROSTER_PAYLOAD: Final[str] = "ops"
_CATALOG: Final[str] = "&metta"
# Shape behavior for every logical head; backend constructor aliases use the
# same entry. Only preserve, broadcast, and matmul derive unevaluated types.
# Other rules describe the runtime transformation whose result is observed by
# the grounded type protocol. A missing entry refuses registration.
SHAPE_RULES: Final[dict[str, str]] = {
    "tensor": "from-data",
    "zeros": "dimensions",
    "ones": "dimensions",
    "randn": "dimensions",
    "arange-t": "range-length",
    "eye": "square",
    "matmul": "matmul",
    "t+": "broadcast",
    "t-": "broadcast",
    "t*": "broadcast",
    "t/": "broadcast",
    "t-pow": "broadcast",
    "t-neg": "preserve",
    "t-exp": "preserve",
    "t-log": "preserve",
    "relu": "preserve",
    "sigmoid": "preserve",
    "softmax": "preserve",
    "tanh": "preserve",
    "t-as": "preserve",
    "reshape": "reshape",
    "t-transpose": "swap-axes",
    "unsqueeze": "insert-axis",
    "squeeze": "remove-unit-axis",
    "t-index": "index-leading-axis",
    "cat": "concatenate-axis",
    "stack": "stack-axis",
    "t-sum": "reduce-all",
    "t-mean": "reduce-all",
    "t-max": "reduce-all",
    "t-min": "reduce-all",
    "t-norm": "reduce-all",
    "t-argmax": "reduce-axis",
    "t-item": "scalar-observation",
    "t-tolist": "nested-observation",
    "t-shape": "shape-observation",
    "t-dtype": "dtype-observation",
    "t-device": "device-observation",
}
DLTensor = NewType("DLTensor", Any)  # type: ignore[valid-newtype]  # ty: ignore[invalid-newtype]
_PROTOCOLS_REGISTERED = threading.Event()
_PROTOCOLS_LOCK = threading.Lock()
_ROSTER_KIND_LOCK = threading.Lock()

# Constructor names aliased per space, each with the arities it registers:
# the operation registers once per backend under name--lib, and each installed
# space carries equations routing its own (tensor ...) to its own backend, so
# two spaces with two default libraries coexist and a later install never
# retargets an earlier space's constructors. Reinstalling a space with another
# default replaces ITS aliases, and the alias equations to withdraw follow
# from this table and the library the space's own roster row names.
_CONSTRUCTOR_ARITIES: Final[dict[str, tuple[int, ...]]] = {
    "tensor": (1,),
    "zeros": (1, 2, 3, 4),
    "ones": (1, 2, 3, 4),
    "randn": (1, 2, 3, 4),
    "arange-t": (1,),
    "eye": (1,),
}
_SPACE_STORES: dict[tuple[str, str], tuple[str, str]] = {}
_STORE_SERIAL = itertools.count(1)

_BROADCAST_SHAPE_SOURCE: Final[str] = r"""
:- use_module(library(clpfd)).
:- metta_extension(metta_arrays_shape, [version('0.1.0')]).
:- metta_export("(: broadcast-shape (-> Expression Expression Expression Bool))").

'broadcast-shape'(Left, Right, Shape, true) :-
    reverse(Left, LeftReversed),
    reverse(Right, RightReversed),
    metta_broadcast_reversed(LeftReversed, RightReversed, ShapeReversed),
    reverse(ShapeReversed, Shape).

metta_broadcast_reversed([], Shape, Shape) :- !.
metta_broadcast_reversed(Shape, [], Shape) :- !.
metta_broadcast_reversed([Left | Lefts], [Right | Rights], [Out | Outs]) :-
    metta_broadcast_dimension(Left, Right, Out),
    metta_broadcast_reversed(Lefts, Rights, Outs).

metta_broadcast_dimension(D2, D1, D) :-
    D1 #\= 1 #/\ D1 #= D #\ D1 #= 1 #/\ D2 #= D,
    D2 #\= 1 #/\ D2 #= D #\ D2 #= 1 #/\ D1 #= D.
"""

#: The typing rule install() declares and uninstall() withdraws, named once
#: so the two spellings cannot drift apart.
_SHAPED_TENSOR_RULE_NAME: Final[str] = "metta-arrays-shaped-dltensor-base"
_SHAPED_TENSOR_RULE: Final[str] = (
    f"!(add-typing-rule! {_SHAPED_TENSOR_RULE_NAME} ordinary "
    "(Annotated DLTensor (Shape $shape)) DLTensor accept)"
)


def Shape(*dimensions: Any) -> Expression:  # noqa: N802  -- type-metadata constructors follow Python's type spelling
    """Build tensor dimension metadata.

    ``Annotated[DLTensor, Shape(...)]`` accepts integers or MeTTa variables.
    The dimensions stay in one expression so the same metadata is both a
    Python annotation claim and an input to ``broadcast-shape``.
    """
    return _expr(S.Shape, _expr(*dimensions))


def _shape_metadata(shape: Atom) -> Expression:
    """A Shape claim around an already assembled dimension expression."""
    return _expr(S.Shape, shape)


def _shaped_tensor(shape: Atom) -> Expression:
    """The MeTTa type carried by one Shape claim."""
    return _expr(S.Annotated, S.DLTensor, _shape_metadata(shape))


#: The shape reader every inference equation goes through. It exists for its
#: FIRST line: `(get-type $x)` is asked with an unbound subject as a matter of
#: course, `!(get-type $subject)` is one of the engine's own pinned questions,
#: and an equation whose head is `(get-type (t+ $l $r))` unifies with that
#: subject and then asks `(get-type $l)` about a variable it has just invented.
#: That descends forever. Reading the operand's metatype first refuses the
#: variable and the whole equation fails, which is the answer a shape rule owes
#: a subject that has no shape yet.
_SHAPE_READER: Final[str] = "metta-arrays-tensor-shape"


def _tensor_shape_equation() -> Expression:
    """The guarded reader: a bound operand's Shape metadata, or no answer."""
    subject = Variable("__arrays_subject")
    shape = Variable("__arrays_shape")
    return _expr(
        S["="],
        _expr(S[_SHAPE_READER], subject),
        _expr(
            S.let,
            ground(value=False),
            _expr(S["=="], _expr(S["get-metatype"], subject), S.Variable),
            _expr(
                S.let,
                _shaped_tensor(shape),
                _expr(S["get-type"], subject),
                shape,
            ),
        ),
    )


def _broadcast_type_equation(name: str) -> Expression:
    """Infer the shaped result of one elementwise binary operation."""
    left = Variable("__arrays_left")
    right = Variable("__arrays_right")
    left_shape = Variable("__arrays_left_shape")
    right_shape = Variable("__arrays_right_shape")
    out_shape = Variable("__arrays_out_shape")
    return _expr(
        S["="],
        _expr(S["get-type"], _expr(S[name], left, right)),
        _expr(
            S.let,
            left_shape,
            _expr(S[_SHAPE_READER], left),
            _expr(
                S.let,
                right_shape,
                _expr(S[_SHAPE_READER], right),
                _expr(
                    S.let,
                    ground(value=True),
                    _expr(S["broadcast-shape"], left_shape, right_shape, out_shape),
                    _shaped_tensor(out_shape),
                ),
            ),
        ),
    )


def _matmul_type_equation() -> Expression:
    """Infer ``(rows, shared) x (shared, columns) -> (rows, columns)``."""
    left = Variable("__arrays_left")
    right = Variable("__arrays_right")
    rows = Variable("__arrays_rows")
    shared = Variable("__arrays_shared")
    columns = Variable("__arrays_columns")
    return _expr(
        S["="],
        _expr(S["get-type"], _expr(S.matmul, left, right)),
        _expr(
            S.let,
            _expr(rows, shared),
            _expr(S[_SHAPE_READER], left),
            _expr(
                S.let,
                _expr(shared, columns),
                _expr(S[_SHAPE_READER], right),
                _shaped_tensor(_expr(rows, columns)),
            ),
        ),
    )


def _type_equations() -> tuple[Expression, ...]:
    """Every get-type equation an install adds, derived from SHAPE_RULES.

    install() adds each one as it registers the head it belongs to and
    uninstall() withdraws the same set; both classify from SHAPE_RULES, so a
    head that gains a shape rule cannot leave its equation behind.
    """
    equations = [_tensor_shape_equation()]
    for head, rule in SHAPE_RULES.items():
        if rule == "broadcast":
            equations.append(_broadcast_type_equation(head))
        elif rule == "matmul":
            equations.append(_matmul_type_equation())
    return tuple(equations)


def _top_indices(xp: Any, scores: Any, count: int) -> list[int]:
    """Top score indexes, best first, without sorting the full NumPy array."""
    size = int(scores.shape[0])
    if count <= 0:
        return []
    if count >= size:
        candidates = list(range(size))
    elif hasattr(xp, "argpartition"):
        boundary = size - count
        partition = xp.argpartition(scores, boundary)[boundary:]
        threshold = min(float(scores[int(index)]) for index in partition)
        nonzero = getattr(xp, "nonzero", None)
        if nonzero is None:
            msg = "an argpartition namespace must also provide nonzero"
            raise RuntimeError(msg)
        better = [int(index) for index in nonzero(scores > threshold)[0]]
        needed = count - len(better)
        tied = [int(index) for index in nonzero(scores == threshold)[0][:needed]]
        candidates = [*better, *tied]
    else:
        order = xp.argsort(scores)
        candidates = [int(order[-(offset + 1)]) for offset in range(count)]
    return sorted(candidates, key=lambda index: (-float(scores[index]), index))


def _alias_types(name: str, library: str) -> list[Expression]:
    """The bare-name arrow declarations one constructor's alias carries.

    A constructor registers as `zeros--numpy` and the space routes `zeros`
    there, so the bare name needs the same arrows the namespaced one declared,
    one per accepted arity. install() adds whichever the space lacks and
    uninstall() withdraws them, both reading them from the registration rather
    than restating the arrow shapes.
    """
    operation = _ops.registered().get(f"{name}--{library}")
    if operation is None:
        return []
    return [
        _expr(S[":"], S[name], declaration.args[1])
        for declaration in operation.declarations
        if declaration.head == S[":"]
        and declaration.args[0] == S[f"{name}--{library}"]
        and isinstance(declaration.args[1], Expression)
        and declaration.args[1].head == S["->"]
    ]


def _alias_equation(name: str, library: str, arity: int) -> Expression:
    _variables = [Variable(f"a{i}") for i in range(1, arity + 1)]
    return Expression(
        [
            S["="],
            Expression([S[name], *_variables]),
            Expression([S[f"{name}--{library}"], *_variables]),
        ]
    )


def _route_equation(alias: str, target: str, arity: int) -> Expression:
    _variables = [Variable(f"a{i}") for i in range(1, arity + 1)]
    return Expression([S["="], Expression([S[alias], *_variables]), Expression([S[target], *_variables])])


def is_array(x: Any) -> bool:
    """Whether a value speaks DLPack, the exchange protocol array libraries share."""
    return hasattr(x, "__dlpack__")


def _compat():
    return require_module(
        "array_api_compat",
        "metta.arrays needs array-api-compat, the standard's compatibility "
        "layer over NumPy, PyTorch, CuPy, JAX and Dask; install pymetta[arrays]",
    )


def _default_library():
    """The registered default array library, imported.

    Which library that is comes from the `array` point's rows, so a second
    library becomes the default by registering with `default=True` ahead of
    the shipped row rather than by an edit here.
    """
    for row in seam.array.table().values():
        if row.default:
            return require_module(row.module, row.missing)
    raise MettaError(seam.array.refusal("the default array library"))


def _index_backends() -> dict[str, Any]:
    """Every registered nearest-neighbour backend, in registration order."""
    return seam.index.table()


def namespace_of(x: Any):
    """The array API namespace an array belongs to: its own library, wrapped."""
    return _compat().array_namespace(x)


def _into(namespace: Any, like: Any, value: Any):
    """``value`` as an array of ``namespace``, converted only when it is not.

    Which LIBRARY a value belongs to is the question, and Python type identity
    is not that question. JAX's tracer and its concrete array are two classes
    of ONE namespace, and a tracer carries ``__dlpack__`` without the
    ``__dlpack_device__`` half, so converting one was both unnecessary and
    impossible: ``from_dlpack`` refuses it with "The array passed to
    from_dlpack must have __dlpack__ and __dlpack_device__ methods", which
    took every traced call through ``jax.jit`` and ``jax.grad`` down with it
    [measured 2026-09-04 on jax 0.11.0: ``namespace_of(tracer)`` IS
    ``namespace_of(concrete)`` while ``type`` reads DynamicJaxprTracer against
    ArrayImpl].

    The type test stays in front of the namespace one as a fast path, because
    two values of one Python type are always of one library and
    ``array_namespace`` is a dispatch this saves on every same-library call.
    """
    if type(value) is type(like) or namespace_of(value) is namespace:
        return value
    return namespace.from_dlpack(value)


def _default_namespace(backend: Any):
    compat = _compat()
    if backend is None:
        library = _default_library()
        return compat.array_namespace(library.zeros(0))
    if isinstance(backend, str):
        backend = importlib.import_module(backend)
    probe = backend.zeros(0) if hasattr(backend, "zeros") else backend.asarray([0])
    return compat.array_namespace(probe)


def _backend_name(namespace: Any) -> str:
    """The stable, fully qualified module identity used in MeTTa aliases."""
    return str(getattr(namespace, "__name__", namespace))


def _describe(x: Any) -> str:
    compat = _compat()
    dtype = str(x.dtype)
    dtype = dtype[dtype.rfind(".") + 1 :]
    shape = "x".join(str(d) for d in x.shape) or "scalar"
    device = str(compat.device(x))
    grad = " grad" if getattr(x, "requires_grad", False) else ""
    return f"<{type(x).__name__} {shape} {dtype} {device}{grad}>"


def _array_type(value: Any) -> Expression:
    """Observe shape on each type query, including an array resized in place."""
    return _shaped_tensor(_expr(*value.shape))


def _register_protocols() -> None:
    """DLTensor typing and protocol printing, once per process."""
    if _PROTOCOLS_REGISTERED.is_set():
        return
    with _PROTOCOLS_LOCK:
        if _PROTOCOLS_REGISTERED.is_set():
            return
        _integrate.register_object_type(is_array, "DLTensor")
        _integrate.register_object_type(is_array, _array_type)
        _integrate.register_repr(is_array, _describe)
        _PROTOCOLS_REGISTERED.set()


def data_of(a: Any) -> Any:
    """Nested expression of numbers to nested lists; grounded values unwrap."""
    if isinstance(a, Expression):
        return [data_of(c) for c in a]
    if isinstance(a, Grounded):
        return _decode(a)
    if isinstance(a, Atom):
        msg = f"tensor data may not contain {a.metatype}: {a}"
        raise TypeError(msg)
    return a


# ------------------------------------------------------ the per-space roster
#
# What a space installed is a fact ABOUT that space, so it is stored where the
# engine keeps every other per-space declaration: one row in the catalog,
# retired by the same walk that retires (annotations ...) and (handles ...)
# when the space is dropped. It replaced a module-level list rewritten by the
# last install anywhere in the process, whose meaning therefore depended on
# call order: installing jax in one space made a numpy space answer
# %Undefined% for the type of a name it had never registered.


def _catalog(m: Any) -> Space:
    """The declaration space this space's runtime reads and writes."""
    return Space(_CATALOG, _runtime=m.runtime)


def _roster_pattern(m: Any) -> Expression:
    """The rows one space's array installations occupy."""
    return _expr(S[_ROSTER_HEAD], S[str(m.name)], V.library, V.ops)


def _installations(m: Any) -> list[tuple[str, tuple[str, ...]]]:
    """(library, names) for every installation row this space carries.

    The row is ordinary catalog data, so a program can write one itself and
    a malformed roster is refused here naming the row, rather than read as
    an empty install.
    """
    found: list[tuple[str, tuple[str, ...]]] = []
    for row in _catalog(m).match(_roster_pattern(m)):
        payload = row.ops
        if (
            not isinstance(payload, Expression)
            or payload.head != S[_ROSTER_PAYLOAD]
            or not all(isinstance(name, Symbol) for name in payload.args)
        ):
            msg = (
                f"({_ROSTER_HEAD} {m.name} {row.library} {payload}) declares no "
                f"array roster: the third field must be "
                f"({_ROSTER_PAYLOAD} <name> ...) naming symbols"
            )
            raise MettaError(msg)
        found.append((str(row.library), tuple(str(name) for name in payload.args)))
    return found


def _installed(m: Any, door: str) -> tuple[str, tuple[str, ...]]:
    """The one installation this space carries, or the refusal naming it."""
    standing = _installations(m)
    if not standing:
        msg = (
            f"no array backend is installed in {m.name}, so arrays.{door} has "
            f"nothing to answer; arrays.install({m.name}) registers one"
        )
        raise MettaError(msg)
    if len(standing) > 1:
        libraries = ", ".join(library for library, _ in standing)
        msg = (
            f"{m.name} carries {len(standing)} array installation rows "
            f"({libraries}), so arrays.{door} has no one answer; "
            f"arrays.install({m.name}, default=...) replaces them with one and "
            f"arrays.uninstall({m.name}) retires them all"
        )
        raise MettaError(msg)
    return standing[0]


def _claimed_ops(m: Any) -> set[str]:
    """Every operation name a standing installation row still claims.

    The operation registry is process-wide by NAME, so retiring one space's
    roster may unregister only what no other space's row names: two spaces on
    numpy share `zeros--numpy` and the whole backend-agnostic operation set.
    """
    claimed: set[str] = set()
    for row in _catalog(m).match(_expr(S[_ROSTER_HEAD], V.space, V.library, V.ops)):
        payload = row.ops
        if isinstance(payload, Expression) and payload.head == S[_ROSTER_PAYLOAD]:
            claimed.update(str(name) for name in payload.args)
    return claimed


def _retire_unclaimed(m: Any, names: Iterable[str]) -> list[str]:
    """Unregister the named operations no standing row claims, and say which.

    In the order given, which is the roster's, so the answer reads as the
    install's own list minus what stayed. The bare constructor names never
    reach the registry, being alias equations, and are skipped here.
    """
    claimed = _claimed_ops(m)
    known = _ops.registered()
    retired = [
        name
        for name in dict.fromkeys(names)
        if name not in claimed and name in known
    ]
    for name in retired:
        m.unregister_op(name)
    return retired


def _clear_installation(m: Any, incoming: str | None = None) -> list[str]:
    """Drop this space's rows and the constructor aliases they route through.

    Returns the names those rows claimed, for the caller to retire once the
    replacement row stands. The INCOMING library's aliases are removed too, so
    a repeated install is idempotent: an install that failed part way through
    leaves its aliases standing, and adding them again would give the space
    the same equation twice.
    """
    previous = _installations(m)
    if previous:
        del _catalog(m)[_roster_pattern(m)]
    libraries = {library for library, _ in previous}
    if incoming is not None:
        libraries.add(incoming)
    for library in libraries:
        for name, arities in _CONSTRUCTOR_ARITIES.items():
            for arity in arities:
                m.remove(_alias_equation(name, library, arity))
    return [name for _, names in previous for name in names]


def _declare_roster_kind(catalog: Space) -> None:
    """Declare the row's shape and its space ownership, once per catalog.

    The kind row makes the engine's own declaration checker refuse a
    malformed roster at the write, and (owned-by-space array-backend) puts
    the head in the retirement walk every space-owned declaration already
    leaves through, so dropping a space takes its roster with it
    [source: engine/spaces/catalog.pl, metta_retire_space_catalog/1;
    commit=76dbea9f4bc10804a5ca19493972dfb7975bc4b0]. A program that has removed this kind row and declared a
    wider one of its own owns the consequence: the next install meets the
    engine's one-kind-row-per-head refusal, which names the row to remove.
    """
    declarations = (
        _expr(S.kind, S[_ROSTER_HEAD], S.symbol, S.symbol, S.term),
        _expr(S["owned-by-space"], S[_ROSTER_HEAD]),
    )
    # One catalog, one kind row: the engine refuses a second for the same
    # head, so two threads installing into two spaces at once would have the
    # loser raise on a check that had already passed. The same reason
    # _PROTOCOLS_LOCK guards the other once-per-process registration here.
    with _ROSTER_KIND_LOCK:
        for declaration in declarations:
            if declaration not in catalog:
                catalog.add(declaration)


def _record_installation(m: Any, library: str, names: Iterable[str]) -> None:
    """Write the one row saying what this space installed."""
    catalog = _catalog(m)
    _declare_roster_kind(catalog)
    catalog.add(
        _expr(
            S[_ROSTER_HEAD],
            S[str(m.name)],
            S[library],
            _expr(S[_ROSTER_PAYLOAD], *(S[name] for name in names)),
        )
    )


def ops(m) -> list[str]:
    """The array operation names installed in this space, in install order.

    `install` returns the same list; this reads it back from the space long
    afterwards, so two spaces on two libraries answer their own rosters
    whatever order they were installed in:

        numpy_space, jax_space = m.space(), m.space()
        arrays.install(jax_space, default=jax.numpy)
        arrays.install(numpy_space, default=numpy)
        arrays.ops(numpy_space)      # ... 'zeros--numpy' ...
        arrays.backend(jax_space)    # 'jax.numpy'

    m may be a context or a space. The longhand is the row itself, which is
    ordinary matchable data: `!(match &metta (array-backend &s $lib $ops) $ops)`.
    A space with no install refuses, naming install as the remedy.
    """
    _library, names = _installed(_integrate.space_of(m), "ops")
    return list(names)


def backend(m) -> str:
    """The array library this space's constructors build in.

    The fully qualified module name install() recorded, `numpy` or
    `jax.numpy`, which is the same name its constructor registrations carry
    after the `--` in `zeros--numpy`. `ops` answers the roster beside it, and
    the row behind both is `(array-backend <space> <library> (ops ...))` in
    `&metta`.
    """
    library, _names = _installed(_integrate.space_of(m), "backend")
    return library


def install(m, default: Any = None) -> list[str]:  # noqa: C901  -- install keeps the array backend registration table together so its branches share one state
    """Register the array operation set on the shared engine.

    default names the library the constructors build in: a module, a module
    name, or None for NumPy. Every other operation dispatches on its
    argument's own namespace, so arrays from any conforming library flow
    through the same MeTTa functions, and a mixed binary call converts the
    right operand into the left's library through from_dlpack.

    Every installed name has one or more arrow declarations. Constructors
    with optional or variadic dimensions have one arrow per accepted arity.

    broadcast-shape is the CLP(FD) relation over shape expressions. It can
    compute a result before any tensor exists, infer an unknown input
    dimension from a required result, or reject incompatible shapes:

        !(let True (broadcast-shape (4 1) (3) $shape) $shape)  ; (4 3)
        !(let True (broadcast-shape ($d 1) (1 3) (4 3)) $d)   ; 4
        !(broadcast-shape (2 3) (4 3) (4 3))                  ; no answer

    t-shape remains observation of an existing tensor. Use broadcast-shape
    when compatibility or inference must happen before materialisation.

    ``Shape`` carries those expressions through Python ``Annotated`` claims.
    User operation arrows retain the claim: argument dimensions unify and
    bind shared result dimensions. Live values report the same type expression.
    ``SHAPE_RULES`` names every installed head's behavior; preserving heads
    share their entire input shape with the result. Other transformations
    expose their actual shape when their result value exists.
    A declared ``(Annotated DLTensor (Shape ...))`` remains a valid DLTensor
    argument, elementwise binary operations derive their output shape with
    ``broadcast-shape``, and rank-two matmul unifies the two inner dimensions:

        (: image (Annotated DLTensor (Shape (4 1))))
        (: bias  (Annotated DLTensor (Shape (3))))
        !(get-type (t+ image bias))  ; (Annotated DLTensor (Shape (4 3)))

    m may be a context or a space. The operations are registered into the
    space either way, which is the object whose storage and introspection
    doors this needs; `install(m)` on a context used to raise
    `MeTTa has no 'is_function'` and leave every operation unregistered.

    What this space installed becomes one catalog row,
    ``(array-backend <space> <library> (ops ...))`` in ``&metta``, which
    ``ops(m)`` and ``backend(m)`` read back and a MeTTa program can match for
    itself. Installing again REPLACES that row, the space's constructor
    aliases, and every operation of the outgoing roster that no other space's
    row still names; ``uninstall(m)`` retires the whole installation, and
    dropping the space retires the row with it. Two spaces may therefore hold
    two libraries at once, in either install order, each answering its own.
    """
    m = _integrate.space_of(m)
    _register_protocols()
    default_module = _default_library() if default is None else default
    if isinstance(default_module, str):
        default_module = importlib.import_module(default_module)
    xp_default = _default_namespace(default_module)
    library = _backend_name(default_module)
    # Before anything is registered, so a failure part way through leaves a
    # space with no roster row, which every reader refuses loudly, rather than
    # a standing row describing an install that did not finish.
    outgoing = _clear_installation(m, library)
    registered: list[str] = []

    m.register_prolog(_BROADCAST_SHAPE_SOURCE)
    typing_rule_result = m.run(_SHAPED_TENSOR_RULE)
    if typing_rule_result != [[True]]:
        msg = f"could not install the shaped DLTensor typing rule: {typing_rule_result!r}"
        raise MettaError(msg)

    def add_type_equation(equation: Expression) -> None:
        if not any(_alpha_eq(equation, atom) for atom in m.atoms()):
            # Definitions must pass through the source runner so the engine
            # compiles them as reduction clauses. Space.add stores an ``=``
            # atom as data, which looks identical in ``atoms()`` but cannot
            # participate in ``get-type`` reduction.
            m.run(str(equation))

    add_type_equation(_tensor_shape_equation())

    def op(
        fn,
        *,
        name: str,
        effect: str,
        # policy-inventory-exempt: mechanism-internal; reason=encoded and raw are the registration transport's two wire-crossing modes, decoded once into the (op ...) kind; evidence=extensions/python/metta/ops.py:_operation_kind
        transport: Literal["encoded", "raw"] = "raw",
        **kw,
    ):
        if (
            m.is_function(name)
            and name not in _known_ops()
            and name not in _CONSTRUCTOR_ARITIES
        ):
            msg = (
                f"refusing to register {name!r}: the engine already has a "
                f"function by that name and it is not an array operation"
            )
            raise MettaError(
                msg
            )
        rule = SHAPE_RULES[name.split("--", 1)[0]]
        if rule == "preserve":
            original = fn

            @wraps(original)
            def preserving(a, *args, **kwargs):
                # NumPy elementwise functions can return scalars for rank-zero arrays.
                result = original(a, *args, **kwargs)
                return result if is_array(result) else namespace_of(a).asarray(result)

            fn = preserving
            tensor = Annotated[DLTensor, _shape_metadata(V.shape)]
            parameter = next(iter(inspect.signature(fn).parameters))
            fn.__annotations__[parameter] = tensor
            fn.__annotations__["return"] = tensor
        elif rule == "broadcast":
            add_type_equation(_broadcast_type_equation(name))
        elif rule == "matmul":
            add_type_equation(_matmul_type_equation())
        m.op(fn, name=name, effect=effect, transport=transport, **kw)
        registered.append(name)
        return fn

    def constructor(
        fn,
        *,
        name: str,
        effect: str,
        # policy-inventory-exempt: mechanism-internal; reason=encoded and raw are the registration transport's two wire-crossing modes, decoded once into the (op ...) kind; evidence=extensions/python/metta/ops.py:_operation_kind
        transport: Literal["encoded", "raw"] = "raw",
        **kw,
    ):
        """A constructor registers per backend as name--library, and THIS
        space routes its bare name there through per-space equations, so
        the default is the space's, never the process's. Installing the
        space again with another default replaces its aliases, which
        _clear_installation withdrew before this ran from the arities in
        _CONSTRUCTOR_ARITIES and the library the outgoing row named.
        """  # noqa: D205  -- the API contract is one continuous invariant, not summary-and-body prose
        arities = list(_CONSTRUCTOR_ARITIES[name])
        namespaced = f"{name}--{library}"
        declarations = [
            _expr(declaration.head, S[namespaced], *declaration.args[1:])
            if (
                isinstance(declaration, Expression)
                and declaration.args
                and declaration.args[0] == S[name]
            )
            else declaration
            for declaration in kw.pop("declarations", ())
        ]
        op(
            fn,
            name=namespaced,
            effect=effect,
            transport=transport,
            arities=arities,
            declarations=declarations,
            **kw,
        )
        for arity in arities:
            m.add(_alias_equation(name, library, arity))
        for alias_type in _alias_types(name, library):
            if alias_type not in m:
                m.add(alias_type)
        registered.append(name)
        return fn

    def aligned(a, b):
        """Two operands in one library: the right converts into the left's,
        through DLPack when it is an array and by lifting when it is a bare
        number, since the standard's functions take arrays on both sides.
        """  # noqa: D205  -- the API contract is one continuous invariant, not summary-and-body prose
        xp = namespace_of(a)
        b = _into(xp, a, b) if is_array(b) else xp.asarray(b, dtype=a.dtype)
        return a, b, xp

    # ------------------------------------------------------------ constructors

    def make_tensor(data: Any) -> DLTensor:
        raw_data = data_of(data) if isinstance(data, Atom) else data
        if is_array(raw_data):
            return DLTensor(ground(raw_data))
        return DLTensor(ground(xp_default.asarray(raw_data, dtype=xp_default.float32)))

    constructor(
        make_tensor,
        name="tensor",
        effect="writesState",
        transport="encoded",
        declarations=[_expr(S.arguments, S.tensor, S.atoms)],
    )

    def zeros(*shape: int) -> DLTensor:
        return xp_default.zeros(tuple(int(dimension) for dimension in shape))

    def ones(*shape: int) -> DLTensor:
        return xp_default.ones(tuple(int(dimension) for dimension in shape))

    def arange_tensor(n: float) -> DLTensor:
        return xp_default.arange(float(n))

    def identity(n: int) -> DLTensor:
        return xp_default.eye(int(n))

    constructor(zeros, name="zeros", effect="writesState")
    constructor(ones, name="ones", effect="writesState")
    constructor(_randn(xp_default), name="randn", effect="oracleIO")
    constructor(arange_tensor, name="arange-t", effect="writesState")
    constructor(identity, name="eye", effect="writesState")

    # ---------------------------------------------------------------- algebra

    def binop(fn, name: str):
        def call(a: DLTensor, b: Any) -> DLTensor:
            a2, b2, xp = aligned(a, b)
            return fn(xp, a2, b2)

        op(call, name=name, effect="writesState")

    def matmul(
        a: DLTensor,
        b: DLTensor,
    ) -> DLTensor:
        a2, b2, xp = aligned(a, b)
        return xp.matmul(a2, b2)

    op(matmul, name="matmul", effect="writesState")
    binop(lambda xp, a, b: xp.add(a, b), "t+")
    binop(lambda xp, a, b: xp.subtract(a, b), "t-")
    binop(lambda xp, a, b: xp.multiply(a, b), "t*")
    binop(lambda xp, a, b: xp.divide(a, b), "t/")

    def negative(a: DLTensor) -> DLTensor:
        return namespace_of(a).negative(a)

    def exponential(a: DLTensor) -> DLTensor:
        return namespace_of(a).exp(a)

    def logarithm(a: DLTensor) -> DLTensor:
        return namespace_of(a).log(a)

    op(negative, name="t-neg", effect="writesState")
    op(exponential, name="t-exp", effect="writesState")
    op(logarithm, name="t-log", effect="writesState")

    def power(a: DLTensor, p: Any) -> DLTensor:
        # Through aligned() like every other binary operation, so a mixed
        # pair (a torch base, a NumPy exponent) converts before the call.
        a2, p2, xp = aligned(a, p)
        return xp.pow(a2, p2)

    op(power, name="t-pow", effect="writesState")

    # ------------------------------------------------------------------ shape

    def reshape(a: DLTensor, *dimensions: int) -> DLTensor:
        return namespace_of(a).reshape(
            a,
            tuple(int(dimension) for dimension in dimensions),
        )

    def transpose(a: DLTensor, d0: int, d1: int) -> DLTensor:
        return _swap(namespace_of(a), a, int(d0), int(d1))

    def unsqueeze(a: DLTensor, dimension: int) -> DLTensor:
        return namespace_of(a).expand_dims(a, axis=int(dimension))

    def squeeze(a: DLTensor, dimension: int) -> DLTensor:
        return namespace_of(a).squeeze(a, axis=int(dimension))

    def tensor_index(a: DLTensor, index: int) -> Any:
        return cast(Any, a)[int(index)]

    op(reshape, name="reshape", effect="writesState", arities=[2, 3, 4, 5])
    op(transpose, name="t-transpose", effect="writesState")
    op(unsqueeze, name="unsqueeze", effect="writesState")
    op(squeeze, name="squeeze", effect="writesState")
    op(tensor_index, name="t-index", effect="writesState")

    # The tensor list is a VALUE these two decode member by member, so the
    # parameter is annotated by what it iterates rather than by the MeTTa atom
    # kind. `Expression` in a parameter position is the evaluation mask: it
    # hands the operand over AS WRITTEN, and `(cat ((tensor ((1 2)))) 0)` then
    # reaches the operation as two unrun `(tensor ...)` calls
    # [source: engine/translator/typing.pl, non_evaluated_parameter_type/1].
    def cat_op(tensors: Iterable[Atom], dim: int = 0) -> DLTensor:
        parts = [_decode(c) for c in tensors]
        dimension = _decode(dim) if isinstance(dim, Atom) else dim
        return DLTensor(
            ground(namespace_of(parts[0]).concat(parts, axis=int(dimension)))
        )

    def stack_op(tensors: Iterable[Atom], dim: int = 0) -> DLTensor:
        parts = [_decode(c) for c in tensors]
        dimension = _decode(dim) if isinstance(dim, Atom) else dim
        return DLTensor(
            ground(namespace_of(parts[0]).stack(parts, axis=int(dimension)))
        )

    op(cat_op, name="cat", effect="writesState", transport="encoded")
    op(stack_op, name="stack", effect="writesState", transport="encoded")

    # ------------------------------------------------------------- reductions

    def tensor_sum(a: DLTensor) -> Any:
        return namespace_of(a).sum(a)

    def tensor_mean(a: DLTensor) -> Any:
        return namespace_of(a).mean(a)

    def tensor_max(a: DLTensor) -> Any:
        return namespace_of(a).max(a)

    def tensor_min(a: DLTensor) -> Any:
        return namespace_of(a).min(a)

    op(tensor_sum, name="t-sum", effect="writesState")
    op(tensor_mean, name="t-mean", effect="writesState")
    op(tensor_max, name="t-max", effect="writesState")
    op(tensor_min, name="t-min", effect="writesState")

    def argmax(a: DLTensor, dim: int = -1) -> Any:
        xp = namespace_of(a)
        out = xp.argmax(a, axis=int(dim))
        return int(out) if cast(Any, a).ndim <= 1 else out

    def tensor_norm(a: DLTensor) -> Any:
        return namespace_of(a).linalg.vector_norm(a)

    op(argmax, name="t-argmax", effect="writesState")
    op(tensor_norm, name="t-norm", effect="writesState")

    # ------------------------------------------------------------ activations

    def relu(a: DLTensor) -> DLTensor:
        xp = namespace_of(a)
        raw = cast(Any, a)
        return xp.where(raw > 0, raw, xp.zeros_like(raw))

    def sigmoid(a: DLTensor) -> DLTensor:
        xp = namespace_of(a)
        return 1.0 / (1.0 + xp.exp(-cast(Any, a)))

    def softmax(a: DLTensor, dim: int = -1) -> DLTensor:
        xp = namespace_of(a)
        shifted = xp.exp(a - xp.max(a, axis=int(dim), keepdims=True))
        return shifted / xp.sum(shifted, axis=int(dim), keepdims=True)

    def hyperbolic_tangent(a: DLTensor) -> DLTensor:
        return namespace_of(a).tanh(a)

    op(relu, name="relu", effect="writesState")
    op(sigmoid, name="sigmoid", effect="writesState")
    op(softmax, name="softmax", effect="writesState")
    op(hyperbolic_tangent, name="tanh", effect="writesState")

    # ------------------------------------------------------------------ exits

    # A reduction answers a zero-dimensional array in some libraries and a
    # scalar in others, so the readout accepts either shape honestly.
    def item(a: Any) -> float:
        return float(a)

    op(item, name="t-item", effect="readOnlyLookup")

    def tolist(a: DLTensor) -> Any:
        data: Any = _decode(a) if isinstance(a, Atom) else a
        listed = data.tolist() if hasattr(data, "tolist") else list(data)
        return _expr(*listed) if isinstance(listed, list) else listed

    op(tolist, name="t-tolist", effect="readOnlyLookup", transport="encoded")

    def shape(a: DLTensor) -> Expression:
        data: Any = _decode(a) if isinstance(a, Atom) else a
        return _expr(*[int(dimension) for dimension in data.shape])

    op(shape, name="t-shape", effect="readOnlyLookup", transport="encoded")

    def dtype(a: DLTensor) -> str:
        return str(cast(Any, a).dtype)

    def device(a: DLTensor) -> str:
        return str(_compat().device(a))

    op(dtype, name="t-dtype", effect="readOnlyLookup")
    op(device, name="t-device", effect="readOnlyLookup")

    def convert(a: DLTensor, lib: str) -> DLTensor:
        """(t-as $x numpy): the same values in another library, via DLPack."""
        target = importlib.import_module(str(lib))
        probe = target.zeros(0) if hasattr(target, "zeros") else target.asarray([0])
        return _compat().array_namespace(probe).from_dlpack(a)

    op(convert, name="t-as", effect="oracleIO")

    _record_installation(m, library, registered)
    # After the replacement row stands, so an operation the new roster also
    # names, and one another space's row names, both stay registered.
    _retire_unclaimed(m, outgoing)
    return registered


def uninstall(m) -> list[str]:
    """Retire this space's array installation; answers what it unregistered.

    The inverse of `install`. The space's roster row goes, with its
    constructor aliases and the bare-name arrows those carried, the `get-type`
    shape equations and the shaped-DLTensor typing rule; then every operation
    the roster named is unregistered UNLESS another space's row still names
    it, because the operation registry is process-wide by name and two spaces
    on numpy share `zeros--numpy` and the whole backend-agnostic set. The
    answer is therefore what actually left the registry, in roster order.

        arrays.install(space, default=numpy)
        arrays.uninstall(space)
        arrays.ops(space)            # refuses: nothing is installed here

    An operation another space claims stays registered, and this space stops
    DECLARING it: `ops.withdraw` releases the rows that would otherwise keep
    the space describing a function it no longer routes to.

    Dropping the space retires the row without this call, the catalog
    retiring a space's declarations with it, but the process-wide operations
    are the registry's and only this door hands them back.

    Two registrations deliberately survive, both keyed on the DLPack
    predicate rather than on a space, so one registration serves every space
    and withdrawing it here would change another space's answers: the
    DLTensor type and array printing hooks, whose own doors are
    `integrate.unregister_object_type` and `integrate.unregister_repr`, and
    the `broadcast-shape` CLP(FD) relation, which `register_prolog` has no
    withdrawal for.

    m may be a context or a space, as `install` takes either.
    """
    m = _integrate.space_of(m)
    standing = _installations(m)
    if not standing:
        msg = (
            f"no array backend is installed in {m.name}, so there is nothing "
            f"to uninstall; arrays.install({m.name}) registers one"
        )
        raise MettaError(msg)
    # The typing rule first: it is the one step that can refuse, and refusing
    # here leaves the installation whole rather than half retired.
    rule_result = m.run(f"!(remove-typing-rule! {_SHAPED_TENSOR_RULE_NAME})")
    if rule_result != [[True]]:
        msg = (
            f"could not retire the shaped DLTensor typing rule from {m.name}: "
            f"{rule_result!r}"
        )
        raise MettaError(msg)
    for equation in _type_equations():
        m.remove(equation)
    for library, _ in standing:
        for name in _CONSTRUCTOR_ARITIES:
            for alias_type in _alias_types(name, library):
                m.remove(alias_type)
    _clear_installation(m)
    names = [name for _, names in standing for name in names]
    retired = _retire_unclaimed(m, names)
    # An operation another space still claims stays registered, but it may not
    # go on being DECLARED here: its rows would keep the space describing a
    # function it no longer routes to, and answering calls on it.
    known = _ops.registered()
    for name in dict.fromkeys(names):
        if name not in retired and name in known:
            _ops.withdraw(m.runtime, name, str(m.name))
    return retired


def _randn(xp_default):
    module = _backend_name(xp_default)

    def randn(*shape: int) -> DLTensor:
        dims = tuple(int(d) for d in shape)
        if hasattr(xp_default, "randn"):
            return xp_default.randn(*dims)
        random = getattr(xp_default, "random", None)
        if random is not None and hasattr(random, "standard_normal"):
            return xp_default.asarray(
                random.standard_normal(dims), dtype=xp_default.float32
            )
        msg = f"{module} offers no normal sampler for randn"
        raise MettaError(msg)

    return randn


def _swap(xp, a, d0: int, d1: int):
    order = list(range(a.ndim))
    order[d0], order[d1] = order[d1], order[d0]
    return xp.permute_dims(a, tuple(order))


def _known_ops() -> set[str]:
    return set(_ops.registered())


class EmbeddingStore:
    """Vectors by key, searchable from MeTTa, in whichever library the
    vectors arrive from.

        store = metta.arrays.EmbeddingStore(m, name="emb")
        store.add(S.dog, numpy.array([1.0, 0.0]))
        m.run("!(collapse (emb-knn (tensor (1.0 0.0)) 1))")

    Cosine similarity uses the array API's own operations, and the matrix
    caches between writes. add() has map semantics: adding an existing key
    replaces its vector in its first-seen position. (name-knn $q $k) is
    nondeterministic retrieval, best first; (name-embed $key) answers the
    stored vector or nothing. Public operation names route through equations
    in this space to unique internal operations, so the same store name in a
    different space cannot retarget this store.

    m may be a context or a space, as ``install`` takes either.
    """  # noqa: D205  -- the API contract is one continuous invariant, not summary-and-body prose

    def __init__(  # noqa: D107  -- the enclosing class documents construction and the object invariants
        self, m, name: str = "emb", mirror: bool = True, backend: str = "auto"  # noqa: FBT001, FBT002  -- the boolean is established API data and positional compatibility is part of the call shape
    ) -> None:
        registered = _index_backends()
        if backend != "auto" and backend not in registered:
            known = ", ".join(registered) or "nothing"
            msg = f"backend is auto or one of {known}, not {backend!r}"
            raise MettaError(msg)
        if backend != "auto" and not registered[backend].available():
            raise ImportError(registered[backend].missing)
        m = _integrate.space_of(m)
        self._m = m
        self._name = name
        self._mirror = mirror
        self._backend = backend
        self._keys: list[Atom] = []
        self._vectors: list[Any] = []
        self._matrix = None
        self._index: tuple[Any, Any] | None = None
        self._width: int | None = None

        def knn(query, k):
            yield from self._search(_decode(query), _decode(k))

        def embed(key):
            atom = key if isinstance(key, Atom) else S[str(key)]
            for stored, vector in zip(self._keys, self._vectors, strict=True):
                if stored == atom:
                    return ground(vector)
            return None

        serial = next(_STORE_SERIAL)
        internal_knn = f"{name}--store-{serial}-knn"
        internal_embed = f"{name}--store-{serial}-embed"
        m.op(
            knn,
            name=internal_knn,
            effect="nondeterministicReadOnly",
            declarations=[_expr(S.arguments, S[internal_knn], S.atoms)],
        )
        m.op(
            embed,
            name=internal_embed,
            effect="readOnlyLookup",
            declarations=[_expr(S.arguments, S[internal_embed], S.atoms)],
        )

        key = (m.name, name)
        previous = _SPACE_STORES.get(key)
        if previous is not None:
            m.remove(_route_equation(f"{name}-knn", previous[0], 2))
            m.remove(_route_equation(f"{name}-embed", previous[1], 1))
        m.add(
            _route_equation(f"{name}-knn", internal_knn, 2),
            _route_equation(f"{name}-embed", internal_embed, 1),
        )
        _SPACE_STORES[key] = (internal_knn, internal_embed)

    def add(self, key: Any, vector: Any) -> None:  # noqa: D102  -- the enclosing type and implemented protocol supply this method contract
        atom = key if isinstance(key, Atom) else S[str(key)]
        vector = self._checked_vector(vector, copy=True)
        try:
            index = self._keys.index(atom)
        except ValueError:
            self._keys.append(atom)
            self._vectors.append(vector)
        else:
            old = self._vectors[index]
            if self._mirror:
                self._m.remove(Expression([S.embedding, atom, ground(old)]))
            self._vectors[index] = vector
        self._matrix = None
        self._index = None
        if self._mirror:
            self._m.add(Expression([S.embedding, atom, ground(vector)]))

    def _checked_vector(self, vector: Any, *, copy: bool = False) -> Any:
        if not is_array(vector):
            library = _default_library()
            vector = library.asarray(vector, dtype=library.float32)
        if vector.ndim != 1:
            msg = f"embedding vectors must be one-dimensional, got shape {tuple(vector.shape)}"
            raise ValueError(
                msg
            )
        width = int(vector.shape[0])
        if self._width is not None and width != self._width:
            msg = f"embedding vector width must be {self._width}, got {width}"
            raise ValueError(
                msg
            )
        xp = namespace_of(vector)
        if not bool(xp.all(xp.isfinite(vector))):
            msg = "embedding vectors must contain only finite values"
            raise ValueError(msg)
        as_float = xp.astype(vector, xp.float32)
        norm = float(xp.linalg.vector_norm(as_float))
        if norm == 0.0:
            msg = "embedding vectors must have a nonzero norm"
            raise ValueError(msg)
        if self._width is None:
            self._width = width
        return xp.asarray(vector, copy=True) if copy else vector

    def __len__(self) -> int:  # noqa: D105  -- the Python data-model hook is defined by its name and enclosing type contract
        return len(self._keys)

    def keys(self) -> list[Atom]:  # noqa: D102  -- the enclosing type and implemented protocol supply this method contract
        return self._keys.copy()

    def vector_for(self, key: Any) -> Any:  # noqa: D102  -- the enclosing type and implemented protocol supply this method contract
        atom = key if isinstance(key, Atom) else S[str(key)]
        for stored, vector in zip(self._keys, self._vectors, strict=True):
            if stored == atom:
                return vector
        msg = f"no embedding stored for {atom}"
        raise KeyError(msg)

    def _normalized_query(self, query: Any):
        xp = namespace_of(self._vectors[0])
        if self._matrix is None:
            rows = [_into(xp, self._vectors[0], v) for v in self._vectors]
            stacked = xp.stack([xp.astype(r, xp.float32) for r in rows])
            norms = xp.sqrt(xp.sum(stacked * stacked, axis=-1, keepdims=True))
            self._matrix = stacked / norms
        q = _into(xp, self._vectors[0], self._checked_vector(query))
        q = xp.reshape(xp.astype(q, xp.float32), (-1,))
        return xp, q / xp.sqrt(xp.sum(q * q))

    def _backend_row(self) -> Any:
        """The index backend this store searches through.

        `auto` takes the first registered row that can run here, which is the
        registration order of the `index` point, so a library installs itself
        into every auto store by registering ahead of the fallback.
        """
        registered = _index_backends()
        if self._backend != "auto":
            return registered[self._backend]
        for row in registered.values():
            if row.available():
                return row
        raise MettaError(seam.index.refusal("this store"))

    def ranked(self, query: Any, k: int):
        """(key atom, cosine) pairs best first: the raw retrieval every
        surface (knn, the matcher) formats its own way. The backend is the
        first available row of the `index` point, whose shipped rows are an
        exact inner-product faiss index and this seat's own Array API path,
        the two byte-agreeing by a differential test; what a row is built into
        is cached until the matrix changes.
        """  # noqa: D205  -- the API contract is one continuous invariant, not summary-and-body prose
        if isinstance(k, bool):
            msg = f"k must be a positive integer, got {k!r}"
            raise TypeError(msg)
        try:
            k = operator.index(k)
        except TypeError:
            msg = f"k must be a positive integer, got {k!r}"
            raise TypeError(msg) from None
        if k <= 0:
            msg = f"k must be a positive integer, got {k}"
            raise ValueError(msg)
        if not self._keys:
            return
        _, q = self._normalized_query(self._resolve(query))
        count = min(k, len(self._keys))
        row = self._backend_row()
        if self._index is None or self._index[0] is not row:
            self._index = (row, row.build(self._matrix))
        for position, score in row.search(self._index[1], q, count):
            yield self._keys[position], round(float(score), 6)

    def _search(self, query: Any, k: int):
        for key, score in self.ranked(query, k):
            yield _expr(key, score)

    def _resolve(self, query: Any) -> Any:
        """A query as a vector: an array or sequence stands as itself, a
        stored key answers its vector.
        """  # noqa: D205  -- the API contract is one continuous invariant, not summary-and-body prose
        if is_array(query) or isinstance(query, (list, tuple)):
            return query
        return self.vector_for(query)


# ---------------------------------------------------- the Array API index row
#
# This package's own nearest-neighbour backend, over the standard rather than
# over any library: it is available wherever the default array library is, and
# it is registered as a FALLBACK so that a specific backend a library ships
# wins `backend="auto"` whatever order the two distributions loaded in. That
# ordering used to be one file's reading order and cannot be any more.

def _argsort_available() -> bool:
    """Always: the Array API path is this package's own and needs no library."""
    return True


def _argsort_build(matrix: Any) -> tuple[Any, Any]:
    """The matrix beside its own namespace; there is nothing else to build."""
    return namespace_of(matrix), matrix


def _argsort_search(built: tuple[Any, Any], query: Any, count: int) -> list[tuple[int, float]]:
    """(row, score) pairs best first, over the normalized matrix.

    NumPy-like namespaces use argpartition for the candidate set; namespaces
    exposing only the Array API use argsort.
    """
    xp, matrix = built
    scores = matrix @ query
    return [(index, float(scores[index])) for index in _top_indices(xp, scores, count)]


seam.index.register(
    "argsort",
    source="package",
    fallback=True,
    available=_argsort_available,
    build=_argsort_build,
    search=_argsort_search,
)
