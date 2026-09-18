"""Purpose: the library-agnostic array layer: one operation set over the
array API standard, exercised with NumPy end to end, DLTensor as a protocol
type the engine checks, protocol printing, cross-library conversion through
DLPack, and the embedding store running on NumPy alone.
The install is a property of the space it happened in: this suite installs
NumPy into the module space, the tests that need a second library install it
into a space of their own, and neither can reach the other.
Guarantees:
  - each installed array operation has an arrow and a cache-safe effect rank;
    broadcast-shape works forwards and backwards as a CLP(FD) relation
    [tested: test_every_array_operation_is_typed_and_a_shape_is_a_constraint,
     test_embedding_store_runs_on_numpy;
     commit=3cfbe0d7417b1c453c2dc12d47e2e47e7de461f7]
  - the module fixture retires its process-global operation registrations, so
    later suites do not inherit array callables [tested: python -m pytest
    ext/metta-arrays/tests/test_arrays.py
    extensions/python/tests/repository/test_operator_documentation.py;
    commit=f88aa8be03cb64cb59d3307515ded8701f418321]
  - fully qualified backend names keep ``jax.numpy`` constructors separate
    from NumPy, and JAX random construction refuses instead of sampling from
    NumPy's hidden global state [tested:
    test_nested_backend_names_do_not_retarget_an_earlier_space,
    test_randn_never_borrows_another_backends_random_state;
    commit=de15573db164c24b9dcaa3e5b783e66dcb05d4d1]
  - Annotated DLTensor shapes participate in base-type checking, broadcast
    inference, nested inference, and rank-two matmul unification before an
    array is built [tested:
    test_annotated_tensor_shapes_flow_through_broadcast_and_matmul;
    commit=4eaefdd8d40e53b2613722287302a14b41704662]
  - shape claims refuse incompatible values, bind output dimensions, and
    preserve shapes across the complete operation roster [tested:
    test_declared_shape_refuses_an_incompatible_live_argument,
    test_a_live_tensor_type_carries_its_current_shape,
    test_every_preserving_unary_head_keeps_symbolic_and_live_shapes;
    commit=4eaefdd8d40e53b2613722287302a14b41704662]
  - a space answers its OWN roster, backend and constructor types whichever
    order the two libraries were installed in, a second install replaces the
    first, and dropping the space retires the row; `--randomly-seed=4` is the
    order that read `%Undefined%` for `tensor--jax.numpy` when the roster was
    one process-global list
    [tested: test_a_space_answers_its_own_roster_in_either_install_order,
    test_a_second_install_replaces_the_roster_and_its_operations,
    test_dropping_the_space_retires_its_installation_row;
    commit=76dbea9f4bc10804a5ca19493972dfb7975bc4b0]
  - uninstall is install's inverse and keeps what another space still claims
    [tested: test_uninstall_retires_the_installation_and_keeps_shared_operations,
    test_the_roster_doors_refuse_an_uninstalled_space_and_a_doubled_row;
    commit=76dbea9f4bc10804a5ca19493972dfb7975bc4b0]
  - a repeated install writes nothing new and uninstall leaves the space
    empty [tested: test_install_is_idempotent_and_uninstall_empties_the_space;
    commit=76dbea9f4bc10804a5ca19493972dfb7975bc4b0]
Open Obligations:
  To Do: None
  Hacks: None
  Future Enhancements: None.
"""  # noqa: D205  -- the scenario narrative is one continuous invariant, not summary-and-body prose

import inspect
import threading

import metta_arrays as arrays

# The default array library is a ROW, and metta-numpy is the package that
# registers it; importing it here is the same door the entry point opens.
import metta_numpy  # noqa: F401  -- imported for the row it registers
import pytest
from hypothesis import given
from hypothesis import strategies as st

from metta import Expression, MeTTa, MettaError, S, V, convert, ground, registered
from metta.vocabularies import EffectClass

numpy = pytest.importorskip("numpy")
pytest.importorskip("array_api_compat")


@pytest.fixture(scope="module")
def am(metta):
    """The module's own installed space, retired through the library's inverse.

    This suite drives the process home space, so anything the install leaves
    behind is left for every later test in this worker: the operations, the
    `get-type` shape equations, the typing rule and the constructor aliases.
    `arrays.uninstall` is the door that retires exactly those, so the teardown
    is one call rather than a hand-rolled reconstruction of what install did.
    """
    arrays.install(metta, default=numpy)
    try:
        yield metta
    finally:
        arrays.uninstall(metta)


def test_numpy_flows_through_the_same_ops(am):  # noqa: D103  -- pytest discovers or injects this callable; its descriptive name states the contract
    r = am.run(
        "!(t-tolist (matmul (tensor ((1.0 2.0 3.0) (4.0 5.0 6.0))) "
        "(tensor ((7.0 8.0) (9.0 10.0) (11.0 12.0)))))"
    )
    assert r == [[Expression(Expression(58.0, 64.0), Expression(139.0, 154.0))]]
    assert am.run("!(t-shape (zeros 2 3))") == [[Expression(2, 3)]]
    assert am.run("!(t-item (t-sum (tensor (1.0 2.0 3.0))))") == [[6.0]]


def test_every_array_operation_is_typed_and_a_shape_is_a_constraint(am):
    """Every installed array op carries an arrow type, and broadcast-shape solves both ways."""
    roster = arrays.ops(am)
    assert len(roster) == len(set(roster)) == 44
    assert arrays.backend(am) == "numpy"
    for name in roster:
        types = [atom for group in am.run(f"!(get-type {name})") for atom in group]
        assert types, name
        # An answer that is not an expression is a FAILURE of this claim, not
        # an error inside it: asking a leaf for its head raises, and raising
        # from inside the generator threw away the (name, types) pair that
        # says which op answered what. Classify first, then compare.
        arrows = [
            isinstance(type_, Expression) and type_.head == S["->"] for type_ in types
        ]
        assert all(arrows), (name, types)

    operations = {
        name: registered()[name] for name in roster if name in registered()
    }
    assert operations
    assert all(operation.effect in EffectClass for operation in operations.values())
    assert registered()["matmul"].effect is EffectClass.writesState
    assert registered()["t-item"].effect is EffectClass.readOnlyLookup
    random_constructor = next(
        name for name in operations if name.startswith("randn--")
    )
    assert registered()[random_constructor].effect is EffectClass.oracleIO

    assert am.run(
        "!(let True (broadcast-shape (4 1) (3) $shape) $shape)"
    ) == [[Expression(4, 3)]]
    assert am.run(
        "!(let True (broadcast-shape ($d 1) (1 3) (4 3)) $d)"
    ) == [[4]]
    assert am.run("!(broadcast-shape (2 3) (4 3) (4 3))") == [[]]

    assert am.run("!(t-shape (reshape (arange-t 4) 2 2))") == [[Expression(2, 2)]]
    tensors = "((tensor ((1 2))) (tensor ((3 4))))"
    assert am.run(f"!(t-tolist (cat {tensors} 0))") == [
        [Expression(Expression(1.0, 2.0), Expression(3.0, 4.0))]
    ]
    assert am.run(f"!(t-tolist (stack {tensors} 0))") == [
        [Expression(Expression(Expression(1.0, 2.0)), Expression(Expression(3.0, 4.0)))]
    ]


def test_a_shape_rule_never_claims_an_unbound_get_type_subject(am):
    """A shape equation must fail on a variable rather than invent one.

    `!(get-type $subject)` is a question the engine answers for an unbound
    subject, and a head spelled `(get-type (t+ $l $r))` unifies with exactly
    that. Without the metatype guard the body then asks `(get-type $l)` about
    a variable it has just invented, and each level invents two more: the
    engine ran out of its 7.5Gb stack at depth 44,071,263 rather than
    answering. Every shape equation reads its operand through the one guarded
    reader, so the equation fails and the engine's own answer stands.
    """
    answers = am.run("!(get-type $subject)")
    assert len(answers) == 1
    assert [str(atom).startswith("$") for atom in answers[0]] == [True]
    assert am.run("!(get-type ())") == [[S["->"]()]]


def test_annotated_tensor_shapes_flow_through_broadcast_and_matmul(am):
    """Shape metadata computes compatible results and rejects incompatible ones."""
    assert str(arrays.Shape(V.n, V.k)) == "(Shape ($n $k))"
    assert arrays.SHAPE_RULES["matmul"] == "matmul"
    assert arrays.SHAPE_RULES["t+"] == "broadcast"

    am.run(
        """
        (: f3-image (Annotated DLTensor (Shape (4 1))))
        (: f3-bias (Annotated DLTensor (Shape (3))))
        (: f3-incompatible (Annotated DLTensor (Shape (2 5))))
        (: f3-matrix (Annotated DLTensor (Shape (2 3))))
        (: f3-right (Annotated DLTensor (Shape (3 4))))
        (: f3-bad-right (Annotated DLTensor (Shape (5 4))))
        (: f3-needs-tensor (-> DLTensor Bool))
        (= (f3-needs-tensor $tensor) True)
        """
    )

    def types_of(call: str) -> set[str]:
        return {
            str(atom)
            for group in am.run(f"!(get-type {call})")
            for atom in group
        }

    broadcast = "(Annotated DLTensor (Shape (4 3)))"
    assert broadcast in types_of("(t+ f3-image f3-bias)")
    assert broadcast in types_of("(t* (t+ f3-image f3-bias) f3-bias)")
    assert not any(
        type_name.startswith("(Annotated DLTensor")
        for type_name in types_of("(t+ f3-image f3-incompatible)")
    )
    assert "(Annotated DLTensor (Shape (2 4)))" in types_of(
        "(matmul f3-matrix f3-right)"
    )
    assert not any(
        type_name.startswith("(Annotated DLTensor")
        for type_name in types_of("(matmul f3-matrix f3-bad-right)")
    )
    assert am.run("!(f3-needs-tensor f3-image)") == [[True]]


def test_the_constructor_builds_numpy_here(am):  # noqa: D103  -- pytest discovers or injects this callable; its descriptive name states the contract
    (group,) = am.run("!(tensor (1.0 2.0))")
    assert isinstance(convert.decode(group[0]), numpy.ndarray)


def test_nested_backend_names_do_not_retarget_an_earlier_space():
    """A later NumPy install leaves an existing JAX space routed to JAX."""
    jax_numpy = pytest.importorskip("jax.numpy")
    first_owner = MeTTa()
    second_owner = MeTTa()
    try:
        first = first_owner.space()
        arrays.install(first, default=jax_numpy)
        (before_second,) = first.run("!(zeros 2 2)")

        second = second_owner.space()
        arrays.install(second, default=numpy)
        (second_answer,) = second.run("!(zeros 2 2)")
        (after_second,) = first.run("!(zeros 2 2)")

        assert type(convert.decode(before_second[0])).__module__.startswith("jax")
        assert isinstance(convert.decode(second_answer[0]), numpy.ndarray)
        assert type(convert.decode(after_second[0])).__module__.startswith("jax")
        assert "zeros--jax.numpy" in registered()
        assert "zeros--numpy" in registered()
        assert arrays.backend(first) == "jax.numpy"
        assert arrays.backend(second) == "numpy"
    finally:
        arrays.uninstall(first)
        arrays.uninstall(second)
        first_owner.close()
        second_owner.close()


@pytest.mark.parametrize("jax_first", [True, False])
def test_a_space_answers_its_own_roster_in_either_install_order(jax_first):
    """Two spaces, two libraries, each answering its own, in either order.

    The roster was one module-level list that every install rewrote, so
    whichever library installed LAST described both spaces: the numpy space
    was asked for the type of `tensor--jax.numpy`, a name it had never
    registered, and answered `%Undefined%`. It is a row in the catalog now,
    keyed by the space that installed it, and order cannot reach it.
    """
    jax_numpy = pytest.importorskip("jax.numpy")
    libraries = (jax_numpy, numpy) if jax_first else (numpy, jax_numpy)
    owner = MeTTa()
    spaces = (owner.space(), owner.space())
    try:
        for space, library in zip(spaces, libraries, strict=True):
            arrays.install(space, default=library)
        for index, (space, library) in enumerate(zip(spaces, libraries, strict=True)):
            other = libraries[1 - index]
            assert arrays.backend(space) == library.__name__
            assert f"tensor--{library.__name__}" in arrays.ops(space)
            assert f"tensor--{other.__name__}" not in arrays.ops(space)
            (types,) = space.run(f"!(get-type tensor--{library.__name__})")
            assert types and all(
                isinstance(type_, Expression) and type_.head == S["->"]
                for type_ in types
            ), (library.__name__, types)
    finally:
        for space in spaces:
            arrays.uninstall(space)
        owner.close()


def test_a_second_install_replaces_the_roster_and_its_operations():
    """Installing again retargets the space and retires what it dropped."""
    jax_numpy = pytest.importorskip("jax.numpy")
    owner = MeTTa()
    space = owner.space()
    try:
        arrays.install(space, default=jax_numpy)
        assert "zeros--jax.numpy" in registered()

        arrays.install(space, default=numpy)
        assert arrays.backend(space) == "numpy"
        assert [name for name in arrays.ops(space) if "jax" in name] == []
        # No other space's row claimed the JAX constructors, so they left the
        # process-wide registry with the roster that named them.
        assert "zeros--jax.numpy" not in registered()
        (answer,) = space.run("!(zeros 2 2)")
        assert isinstance(convert.decode(answer[0]), numpy.ndarray)
        # One alias equation per accepted arity, not two installs' worth.
        aliases = [
            atom for atom in space.atoms() if str(atom).startswith("(= (zeros ")
        ]
        assert len(aliases) == 4, aliases
    finally:
        arrays.uninstall(space)
        owner.close()


def test_dropping_the_space_retires_its_installation_row():
    """The row is a space-owned catalog declaration, so the drop takes it.

    Only the row. The operations belong to the process-wide registry, which
    the engine's space lifetime knows nothing about, and that is the division
    the two doors draw: a drop retires the declaration saying which space had
    them, `uninstall` hands the operations themselves back. The keeper here
    still answers afterwards, and its own uninstall is what empties the
    registry.
    """
    owner = MeTTa()
    try:
        keeper, dying = owner.space(), owner.space()
        arrays.install(keeper, default=numpy)
        arrays.install(dying, default=numpy)
        standing = (
            f"!(match &metta (array-backend {dying.name} $library $ops) $library)"
        )
        assert owner.run(standing) == [[S.numpy]]

        dying.drop()
        assert owner.run(standing) == [[]]
        assert arrays.backend(keeper) == "numpy"
        assert keeper.run("!(t-item (t-sum (tensor (1.0 2.0))))") == [[3.0]]
        arrays.uninstall(keeper)
    finally:
        owner.close()


def test_uninstall_retires_the_installation_and_keeps_shared_operations():
    """Install's inverse, which may not take what another space still uses."""
    jax_numpy = pytest.importorskip("jax.numpy")
    owner = MeTTa()
    try:
        keeper, leaving = owner.space(), owner.space()
        arrays.install(keeper, default=jax_numpy)
        arrays.install(leaving, default=jax_numpy)

        (routed,) = leaving.run("!(zeros 2 2)")
        assert type(convert.decode(routed[0])).__module__.startswith("jax")

        assert "zeros--jax.numpy" not in arrays.uninstall(leaving)
        assert "zeros--jax.numpy" in registered()
        assert arrays.backend(keeper) == "jax.numpy"
        # The space that left routes nothing OF ITS OWN: its alias equations
        # went with the roster. What a bare `(zeros ...)` reaches there now is
        # whatever the inherited &self holds, which is another space's install.
        assert [
            atom for atom in leaving.atoms() if str(atom).startswith("(= (zeros ")
        ] == []

        assert "zeros--jax.numpy" in arrays.uninstall(keeper)
        assert "zeros--jax.numpy" not in registered()
        with pytest.raises(MettaError, match="nothing to uninstall"):
            arrays.uninstall(keeper)
    finally:
        owner.close()


def test_install_is_idempotent_and_uninstall_empties_the_space():
    """Installing again writes nothing new, and uninstalling leaves nothing.

    Every equation, alias and arrow lands only where the space lacks it, so a
    repeated install is a no-op on the space, and `uninstall` withdraws each
    of them by construction rather than by matching their text
    [measured 2026-09-07 on this tree: a fresh space holds 0 atoms, 197 after
    the first install, 197 after the third, and 0 after the uninstall].
    """
    owner = MeTTa()
    space = owner.space()
    try:
        assert space.atoms() == []
        first = arrays.install(space, default=numpy)
        try:
            settled = len(space.atoms())
            assert settled > len(first)
            for _ in range(2):
                assert arrays.install(space, default=numpy) == first
                assert len(space.atoms()) == settled
            assert space.run("!(t-item (t-sum (tensor (1.0 2.0))))") == [[3.0]]
        finally:
            arrays.uninstall(space)
        assert space.atoms() == []
    finally:
        owner.close()


def test_the_roster_doors_refuse_an_uninstalled_space_and_a_doubled_row():
    """Absence and ambiguity refuse by name, and install is the remedy.

    The row is ordinary catalog data, so a program can write one itself and
    both failures are reachable from MeTTa alone.
    """
    owner = MeTTa()
    space = owner.space()
    malformed = f"(array-backend {space.name} numpy nonsense)"
    try:
        for door in (arrays.ops, arrays.backend):
            with pytest.raises(MettaError, match="no array backend is installed"):
                door(space)

        owner.run(f"!(add-atom &metta {malformed})")
        with pytest.raises(MettaError, match="declares no array roster"):
            arrays.ops(space)
        owner.run(f"!(remove-atom &metta {malformed})")

        arrays.install(space, default=numpy)
        owner.run(f"!(add-atom &metta (array-backend {space.name} numpy (ops t+)))")
        with pytest.raises(MettaError, match="carries 2 array installation rows"):
            arrays.backend(space)

        arrays.install(space, default=numpy)
        assert arrays.backend(space) == "numpy"
        assert len(arrays.ops(space)) == 44
    finally:
        arrays.uninstall(space)
        owner.close()


def test_randn_never_borrows_another_backends_random_state():
    """A namespace without implicit randomness refuses rather than crossing backends."""
    jax_numpy = pytest.importorskip("jax.numpy")
    numpy.random.seed(1701)
    expected_next = numpy.random.standard_normal(4)
    numpy.random.seed(1701)
    with pytest.raises(MettaError, match="jax\\.numpy offers no normal sampler"):
        arrays._randn(jax_numpy)(3)
    assert numpy.array_equal(numpy.random.standard_normal(4), expected_next)


def test_activations_are_standard_not_torch(am):  # noqa: D103  -- pytest discovers or injects this callable; its descriptive name states the contract
    assert am.run("!(t-tolist (relu (tensor (-1.0 2.0))))") == [[Expression(0.0, 2.0)]]
    (group,) = am.run("!(t-tolist (softmax (tensor (0.0 0.0))))")
    assert [round(float(x), 3) for x in group[0]] == [0.5, 0.5]


def test_ndarray_identity_through_space(am):  # noqa: D103  -- pytest discovers or injects this callable; its descriptive name states the contract
    array = numpy.arange(4.0)
    space = am._new_space()
    space.add(S.holds(ground(array)))
    assert convert.decode(space.match(S.holds(V.a))[0].a) is array


def test_dltensor_is_a_protocol_type_the_engine_checks(am):  # noqa: D103  -- pytest discovers or injects this callable; its descriptive name states the contract
    # get-type answers the classes and the protocol of the BUILT tensor, so
    # the tensor is built first. get-type does not evaluate its argument, and
    # the classes are a property of the value rather than of the call that
    # would make it.
    (answers,) = am.run("!(collapse (let $t (tensor (1.0)) (get-type $t)))")
    names = {str(a) for a in answers[0]}
    assert "ndarray" in names and "DLTensor" in names
    # and a declared (-> DLTensor DLTensor DLTensor) holds for NumPy values,
    # which is the whole point of protocol typing:
    assert am.run("!(t-item (t-sum (matmul (eye 2) (eye 2))))") == [[2.0]]


def test_a_protocol_and_a_declaration_are_both_answered_once(am):
    """The two sources of extra types compose instead of hiding each other.

    A protocol name reaches the engine through the shim's bridge, which
    computes it in Python; a `(py-atom f Type)` declaration reaches it
    through seam:grounded_extra_type/2, in Prolog. The engine used to CHOOSE
    between the bridge and the branch the declaration hangs off, so with
    the library loaded the declaration was dropped
    [tested test_ops.py::test_a_declared_type_survives_the_library_being_loaded].
    Answering both raises the opposite question, whether a name can now
    arrive twice, so this pins the whole list rather than a membership:
    the shaped protocol, then the classes, the named protocol, and the
    declaration, each once.
    """
    source = "\"__import__('numpy').arange(3.0)\""
    (answers,) = am.run(
        f"!(let $a (py-atom {source} (-> Number Number)) (collapse (get-type $a)))"
    )
    names = [str(a) for a in answers[0]]
    assert names == ["(Annotated DLTensor (Shape (3)))", "ndarray", "DLTensor", "(-> Number Number)"], names


def test_protocol_printing_covers_any_library(am):  # noqa: D103  -- pytest discovers or injects this callable; its descriptive name states the contract
    (group,) = am.run("!(tensor ((1.0 2.0)))")
    printed = repr(group[0])
    assert "1x2" in printed and "float32" in printed and "ndarray" in printed


def test_cross_library_conversion_via_dlpack(am):  # noqa: D103  -- pytest discovers or injects this callable; its descriptive name states the contract
    pytest.importorskip("torch")
    space = am._new_space()
    space.add(S.np_vec(ground(numpy.array([1.0, 2.0], dtype=numpy.float32))))
    (group,) = space.run(
        "!(t-dtype (t-as (match (context-space) (np-vec $v) $v) torch))"
    )
    assert "float32" in str(group[0])


def test_mixed_library_binary_op_converts_rightward(am):  # noqa: D103  -- pytest discovers or injects this callable; its descriptive name states the contract
    torch = pytest.importorskip("torch")
    left = numpy.ones((2, 2), dtype=numpy.float32)
    right = torch.ones(2, 2)
    space = am._new_space()
    space.add(S.pairT(ground(left), ground(right)))
    (group,) = space.run(
        "!(t-item (t-sum (match (context-space) (pairT $a $b) (matmul $a $b))))"
    )
    assert float(group[0]) == 8.0


class _SameLibraryOtherType(numpy.ndarray):
    """One library, another Python class, and no device half of DLPack.

    This is JAX's tracer in miniature and it needs no JAX: `array_namespace`
    answers NumPy for a subclass, so the two operands belong to ONE library,
    while `type` reads two classes and `from_dlpack` refuses a value that
    exports `__dlpack__` without `__dlpack_device__`.
    """

    @property
    def __dlpack_device__(self):
        """Absent, the way a JAX tracer's is."""
        msg = "__dlpack_device__"
        raise AttributeError(msg)


def test_an_operand_of_the_same_library_is_not_converted_through_dlpack(am):
    """Which LIBRARY an operand belongs to is the question, not which class.

    Python type identity answered it for a while, and JAX is where that
    breaks: a traced value and a concrete array are two classes of one
    namespace, and a tracer carries `__dlpack__` without
    `__dlpack_device__`, so the conversion was both unnecessary and
    impossible. Measured 2026-09-04 on jax 0.11.0, the reported case
    `(t+ <concrete> <tracer>)` raised `Python TypeError ... The array passed
    to from_dlpack must have __dlpack__ and __dlpack_device__ methods`.
    """
    left = numpy.asarray([1.0, 2.0, 3.0], dtype=numpy.float32)
    right = numpy.asarray([1.0, 1.0, 1.0], dtype=numpy.float32).view(
        _SameLibraryOtherType
    )
    assert arrays.namespace_of(right) is arrays.namespace_of(left)
    assert type(right) is not type(left)
    assert not hasattr(right, "__dlpack_device__")

    answer = am.eval(S["t-item"](S["t-sum"](S["t+"](ground(left), ground(right)))))
    assert answer == [9.0]


def test_a_jax_tracer_crosses_a_binary_op_and_a_gradient_reaches_it():
    """The reported case, on the library it was reported against.

    A tracer on the RIGHT is what fails: the left operand names the
    namespace, so a concrete left and a traced right is the pair that used to
    reach from_dlpack. Both directions are checked here because only one of
    them ever broke.
    """
    jax = pytest.importorskip("jax")
    jax_numpy = pytest.importorskip("jax.numpy")
    context = MeTTa()
    arrays.install(context, default=jax_numpy)
    try:
        concrete = jax_numpy.asarray([1.0, 2.0, 3.0])

        def through_metta(traced):
            call = S["t+"](ground(concrete), ground(traced))
            return arrays.data_of(context.self.eval(call)[0]).sum()

        assert float(through_metta(concrete)) == 12.0
        assert float(jax.jit(through_metta)(concrete)) == 12.0
        assert list(jax.grad(through_metta)(concrete)) == [1.0, 1.0, 1.0]
    finally:
        arrays.uninstall(context)
        context.close()


def test_install_takes_a_context_as_well_as_a_space():
    """`install(m)` is the spelling every downstream wrote, and it raised.

    The operations are registered into a space, and MeTTa refuses a Space
    door rather than forwarding it, so the installer died on the first one it
    reached: `MeTTa has no 'is_function'`, with every array operation left
    unregistered.
    """
    context = MeTTa()
    names = arrays.install(context, default=numpy)
    try:
        assert "t+" in names
        assert context.self.is_function("t+")
        assert arrays.ops(context) == names
        assert context.run("!(t-item (t-sum (tensor (1.0 2.0 3.0))))") == [[6.0]]
    finally:
        arrays.uninstall(context)
        context.close()


def test_embedding_store_runs_on_numpy(am):  # noqa: D103  -- pytest discovers or injects this callable; its descriptive name states the contract
    space = am._new_space()
    store = arrays.EmbeddingStore(space, name="npk")
    routes = arrays._store_routes(space, "npk")
    assert routes is not None, "the store recorded no routes in its own space"
    internal_knn, internal_embed = routes
    assert (
        registered()[internal_knn].effect
        is EffectClass.nondeterministicReadOnly
    )
    assert registered()[internal_embed].effect is EffectClass.readOnlyLookup
    store.add(S.dog, numpy.array([1.0, 0.0, 0.0]))
    store.add(S.cat, numpy.array([0.9, 0.1, 0.0]))
    store.add(S.car, numpy.array([0.0, 0.0, 1.0]))
    (group,) = space.run("!(collapse (npk-knn (tensor (1.0 0.0 0.0)) 2))")
    (pairs,) = group
    assert [p[0] for p in pairs] == [S.dog, S.cat]
    scores = [float(p[1]) for p in pairs]
    assert scores == sorted(scores, reverse=True)


def test_a_stores_record_dies_with_its_space(am):
    """A dropped space takes its store record with it, and names are POOLED.

    The record was a module-global dict keyed by ``(space name, store name)``
    and nothing removed an entry when its space went, so the next space to be
    handed the recycled anonymous name read a closed store's internal
    operations. This asserts the recycling happens, because a test that never
    reuses the name proves nothing about the record that outlived it.
    """
    first = am._new_space()
    reused = first.name
    arrays.EmbeddingStore(first, name="dropk")
    assert arrays._store_routes(first, "dropk") is not None
    first.drop()

    second = am._new_space()
    try:
        assert second.name == reused, "the name pool did not recycle, so this proves nothing"
        assert arrays._store_routes(second, "dropk") is None
    finally:
        second.drop()


def test_embedding_store_takes_a_context_as_well_as_a_space():
    """install()'s sibling had the same hole: `MeTTa has no 'name'`.

    The store registers its two internal operations into a space and keys
    itself by that space's name, so a context died on the first Space door
    the constructor reached, one line after the backend check.
    """
    context = MeTTa()
    # The query is written `(tensor ...)`, so this context needs its OWN array
    # install to reduce it. It used to reach one another test had left in the
    # shared &self, which made it pass or fail on the shuffle.
    arrays.install(context, default=numpy)
    try:
        store = arrays.EmbeddingStore(context, name="ctxk")
        store.add(S.dog, numpy.array([1.0, 0.0]))
        # The store's routes are a row in the space's own catalog, so this
        # reads what the space holds rather than what a process-wide dict
        # remembers. A dict keyed by the space's NAME outlived the space, and
        # anonymous names are pooled.
        routes = arrays._store_routes(context.self, "ctxk")
        assert routes is not None
        (group,) = context.run("!(collapse (ctxk-knn (tensor (1.0 0.0)) 1))")
        assert [pair[0] for pair in group[0]] == [S.dog]
    finally:
        arrays.uninstall(context)
        for name in routes:
            context.self.unregister_op(name)
        context.close()


def test_top_indices_match_full_order_and_stabilize_ties():  # noqa: D103  -- pytest discovers or injects this callable; its descriptive name states the contract
    xp = arrays.namespace_of(numpy.array([0.0]))
    scores = numpy.random.default_rng(7).normal(size=10_000)
    expected = sorted(
        range(len(scores)), key=lambda index: (-float(scores[index]), index)
    )[:25]
    assert arrays._top_indices(xp, scores, 25) == expected

    ties = numpy.array([0.5, 1.0, 1.0, 1.0, 0.1])
    assert arrays._top_indices(xp, ties, 2) == [1, 2]
    assert arrays._top_indices(xp, ties, 0) == []
    assert arrays._top_indices(xp, ties, len(ties)) == [1, 2, 3, 0, 4]


def test_array_protocol_registration_is_idempotent(monkeypatch):  # noqa: D103  -- pytest discovers or injects this callable; its descriptive name states the contract
    calls = []
    monkeypatch.setattr(arrays, "_PROTOCOLS_REGISTERED", threading.Event())
    monkeypatch.setattr(arrays, "_PROTOCOLS_LOCK", threading.Lock())
    monkeypatch.setattr(
        arrays._integrate,
        "register_object_type",
        lambda *args: calls.append(("type", args)),
    )
    monkeypatch.setattr(
        arrays._integrate,
        "register_repr",
        lambda *args: calls.append(("repr", args)),
    )

    arrays._register_protocols()
    arrays._register_protocols()

    assert [kind for kind, _ in calls] == ["type", "type", "repr"]


def test_same_named_embedding_stores_route_per_space(metta):  # noqa: D103  -- pytest discovers or injects this callable; its descriptive name states the contract
    with metta._new_space() as left, metta._new_space() as right:
        left_store = arrays.EmbeddingStore(left, name="shared-emb")
        right_store = arrays.EmbeddingStore(right, name="shared-emb")
        left_store.add(S.dog, numpy.array([1.0, 0.0]))
        right_store.add(S.cat, numpy.array([0.0, 1.0]))

        assert left.run("!(shared-emb-embed dog)")
        assert left.run("!(shared-emb-embed cat)") == [[]]
        assert right.run("!(shared-emb-embed cat)")
        assert right.run("!(shared-emb-embed dog)") == [[]]


def test_embedding_store_replaces_duplicate_keys_and_owns_vectors(metta):  # noqa: D103  -- pytest discovers or injects this callable; its descriptive name states the contract
    with metta._new_space() as space:
        store = arrays.EmbeddingStore(space, name="replace-emb")
        original = numpy.array([1.0, 0.0])
        store.add(S.same, original)
        original[:] = [0.0, 1.0]
        assert store.vector_for(S.same).tolist() == [1.0, 0.0]

        replacement = numpy.array([0.0, 1.0])
        store.add(S.same, replacement)
        assert len(store) == 1
        assert store.keys() == [S.same]
        assert store.vector_for(S.same).tolist() == [0.0, 1.0]
        assert len(space.match(S.embedding(S.same, V.vector))) == 1


@pytest.mark.parametrize(
    ("vector", "message"),
    [
        (numpy.array([[1.0, 0.0]]), "one-dimensional"),
        (numpy.array([numpy.nan, 1.0]), "finite"),
        (numpy.array([numpy.inf, 1.0]), "finite"),
        (numpy.array([0.0, 0.0]), "nonzero"),
    ],
)
def test_embedding_store_validates_added_vectors(metta, vector, message):  # noqa: D103  -- pytest discovers or injects this callable; its descriptive name states the contract
    with metta._new_space() as space:
        store = arrays.EmbeddingStore(space, name="validated-emb")
        with pytest.raises(ValueError, match=message):
            store.add(S.bad, vector)


def test_embedding_store_requires_one_width_and_positive_integer_k(metta):  # noqa: D103  -- pytest discovers or injects this callable; its descriptive name states the contract
    with metta._new_space() as space:
        store = arrays.EmbeddingStore(space, name="bounded-emb")
        store.add(S.good, numpy.array([1.0, 0.0]))
        with pytest.raises(ValueError, match="width must be 2"):
            store.add(S.wide, numpy.array([1.0, 0.0, 0.0]))
        for invalid in (0, -1):
            with pytest.raises(ValueError, match="positive integer"):
                list(store.ranked([1.0, 0.0], invalid))
        for invalid in (True, 1.5, "1"):
            with pytest.raises(TypeError, match="positive integer"):
                list(store.ranked([1.0, 0.0], invalid))
        with pytest.raises(ValueError, match="width must be 2"):
            list(store.ranked([1.0, 0.0, 0.0], 1))


def test_arrays_layer_is_torch_free():
    """The module must not import torch anywhere, even lazily by name."""
    source = inspect.getsource(arrays)
    assert "import torch" not in source


def test_declared_shape_refuses_an_incompatible_live_argument(am):
    """A shape claim is checked at the operation door before Python executes."""
    from typing import Annotated

    calls = []

    def claimed(a: Annotated[arrays.DLTensor, arrays.Shape(2, 3)]) -> arrays.DLTensor:
        calls.append(a)
        return a

    name = "shape-claimed-argument"
    am.op(claimed, name=name, effect="writesState", transport="raw")
    try:
        answers = am.eval(S[name](ground(numpy.ones((4, 1)))))
        assert answers and all(isinstance(answer, Expression) and answer.head == S.Error for answer in answers)
        assert any(
            answer.args[1] == S.BadArgType(
                1, S.Annotated(S.DLTensor, arrays.Shape(2, 3)),
                S.Annotated(S.DLTensor, arrays.Shape(4, 1)),
            )
            for answer in answers
        )
        assert calls == []
        assert arrays.data_of(am.eval(S[name](ground(numpy.ones((2, 3)))))[0]).shape == (2, 3)
    finally:
        am.unregister_op(name)


def test_declared_shape_variables_derive_the_result_without_execution(am):
    """Input bindings project through the ordinary arrow to its result."""
    from typing import Annotated

    calls = []

    def rows(
        a: Annotated[arrays.DLTensor, arrays.Shape(V.rows, V.columns)],
    ) -> Annotated[arrays.DLTensor, arrays.Shape(V.rows)]:
        calls.append(a)
        return numpy.sum(a, axis=1)

    name = "shape-claimed-rows"
    am.op(rows, name=name, effect="writesState", transport="raw")
    try:
        am.run("(: shape-claimed-input (Annotated DLTensor (Shape (2 3))))")
        assert am.run(f"!(get-type ({name} shape-claimed-input))") == [
            [S.Annotated(S.DLTensor, arrays.Shape(2))]
        ]
        assert am.eval(S["get-type"](S[name](ground(numpy.ones((5, 7)))))) == [
            S.Annotated(S.DLTensor, arrays.Shape(5))
        ]
        assert calls == []
    finally:
        am.unregister_op(name)


def test_a_live_tensor_type_carries_its_current_shape(am):
    """Grounded types observe dimensions anew, including scalar and empty axes."""
    for dimensions in ((), (0,), (2, 3)):
        value = numpy.ones(dimensions)
        assert S.Annotated(S.DLTensor, arrays.Shape(*dimensions)) in am.eval(
            S["get-type"](ground(value))
        )
    value = numpy.ones((2, 3))
    atom = ground(value)
    value.resize((3, 2), refcheck=False)
    assert S.Annotated(S.DLTensor, arrays.Shape(3, 2)) in am.eval(S["get-type"](atom))


@pytest.mark.parametrize(
    "head", [head for head, rule in arrays.SHAPE_RULES.items() if rule == "preserve"]
)
@pytest.mark.parametrize("dimensions", [(), (0, 3), (2, 3)])
def test_every_preserving_unary_head_keeps_symbolic_and_live_shapes(am, head, dimensions):
    """Every preserving head has the same arrow-level shape behavior."""
    suffix = "x".join(map(str, dimensions)) or "scalar"
    subject = S[f"shape-unary-input-{head}-{suffix}"]
    expected = S.Annotated(S.DLTensor, arrays.Shape(*dimensions))
    am.run(f"(: {subject} {expected})")
    extra = (ground("numpy"),) if head == "t-as" else ()
    assert expected in am.eval(S["get-type"](S[head](subject, *extra)))
    argument = ground(numpy.ones(dimensions))
    assert expected in am.eval(S["get-type"](S[head](argument, *extra)))
    assert arrays.data_of(am.eval(S[head](argument, *extra))[0]).shape == dimensions


def test_every_registered_head_has_a_shape_rule(am):
    """A newly registered head cannot omit its shape behavior."""
    assert {name.split("--", 1)[0] for name in arrays.ops(am)} == set(arrays.SHAPE_RULES)


@pytest.mark.parametrize(
    "head", [head for head, rule in arrays.SHAPE_RULES.items() if rule == "broadcast"]
)
def test_broadcast_rules_retain_scalar_operands(am, head):
    """Refinement does not narrow the scalar-capable binary operation arrows."""
    tensor = ground(numpy.ones((2, 3)))
    (answer,) = am.eval(S[head](tensor, ground(2.0)))
    assert arrays.data_of(answer).shape == (2, 3)


@pytest.mark.parametrize(
    ("left", "right", "expected"),
    [((3,), (3, 2), (2,)), ((2, 3), (3,), (2,)), ((5, 2, 3), (5, 3, 4), (5, 2, 4))],
)
def test_rank_two_matmul_inference_does_not_narrow_backend_execution(am, left, right, expected):
    """The backend still handles vector and batched cases beyond static inference."""
    (answer,) = am.eval(S.matmul(ground(numpy.ones(left)), ground(numpy.ones(right))))
    assert arrays.data_of(answer).shape == expected


def test_declared_shape_checks_the_evaluated_constructor_result(am):
    """A bare constructor arrow defers the shape check until its value exists."""
    from typing import Annotated

    calls = []

    def claimed(a: Annotated[arrays.DLTensor, arrays.Shape(2, 3)]) -> arrays.DLTensor:
        calls.append(a)
        return a

    name = "shape-claimed-constructor"
    am.op(claimed, name=name, effect="writesState", transport="raw")
    try:
        (good,) = am.eval(S[name](S.zeros(2, 3)))
        assert arrays.data_of(good).shape == (2, 3)
        answers = am.eval(S[name](S.zeros(4, 1)))
        assert answers and all(isinstance(answer, Expression) and answer.head == S.Error for answer in answers)
        assert any("(Shape (4 1))" in str(answer) for answer in answers)
        assert len(calls) == 1
    finally:
        am.unregister_op(name)


@given(dimensions=st.lists(st.integers(min_value=0, max_value=3), max_size=4))
def test_live_and_inferred_shapes_agree_across_ranks_and_empty_axes(am, dimensions):
    """Backend shapes agree with inference for scalars, empty axes, and higher ranks."""
    argument = ground(numpy.ones(tuple(dimensions)))
    call = S["t-exp"](argument)
    (result,) = am.eval(call)
    expected = S.Annotated(S.DLTensor, arrays.Shape(*arrays.data_of(result).shape))
    assert expected in am.eval(S["get-type"](argument))
    assert am.eval(S["get-type"](call)) == [expected]
