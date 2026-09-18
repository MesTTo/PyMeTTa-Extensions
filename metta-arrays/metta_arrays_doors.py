"""Purpose: register the arrays accessor using deferred implementation references.

Guarantees: registration imports no implementation and accessor calls preserve
  its behavior [tested: test_array_namespace_preserves_installation_and_withdrawal; commit=b615b5a33b43252ef9826e5387da7c9bd7f6b543].
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

import metta.doors as _doors
from metta import seam

if TYPE_CHECKING:
    from metta import SpaceLike



@_doors.door(
    kind=_doors.Kind.provider,
    answers=_doors.AnswersAs.sequence,
    effect=_doors.EffectClass.oracleIO,
    determinism=_doors.Determinism.det,
    tiers=(_doors.Tier.sync, _doors.Tier.context),
    provider=_doors.Provider('metta-arrays', 'arrays'),
    evidence=('ext/metta-arrays/tests/test_arrays_doors.py::test_array_namespace_preserves_installation_and_withdrawal',),
)
def install(m: SpaceLike, default: Any=None) -> list[str]:
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
    return import_module('metta_arrays').install(m, default)

@_doors.door(
    kind=_doors.Kind.provider,
    answers=_doors.AnswersAs.sequence,
    effect=_doors.EffectClass.oracleIO,
    determinism=_doors.Determinism.det,
    tiers=(_doors.Tier.sync, _doors.Tier.context),
    provider=_doors.Provider('metta-arrays', 'arrays'),
    evidence=('ext/metta-arrays/tests/test_arrays_doors.py::test_array_namespace_preserves_installation_and_withdrawal',),
)
def uninstall(m: SpaceLike) -> list[str]:
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
    return import_module('metta_arrays').uninstall(m)

@_doors.door(
    kind=_doors.Kind.introspection,
    answers=_doors.AnswersAs.sequence,
    effect=_doors.EffectClass.readOnlyLookup,
    determinism=_doors.Determinism.det,
    tiers=(_doors.Tier.sync, _doors.Tier.context),
    provider=_doors.Provider('metta-arrays', 'arrays'),
    evidence=('ext/metta-arrays/tests/test_arrays_doors.py::test_array_namespace_preserves_installation_and_withdrawal',),
)
def ops(m: SpaceLike) -> list[str]:
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
    return import_module('metta_arrays').ops(m)

@_doors.door(
    kind=_doors.Kind.introspection,
    answers=_doors.AnswersAs.text,
    effect=_doors.EffectClass.readOnlyLookup,
    determinism=_doors.Determinism.det,
    tiers=(_doors.Tier.sync, _doors.Tier.context),
    provider=_doors.Provider('metta-arrays', 'arrays'),
    evidence=('ext/metta-arrays/tests/test_arrays_doors.py::test_array_namespace_preserves_installation_and_withdrawal',),
)
def backend(m: SpaceLike) -> str:
    """The array library this space's constructors build in.

    The fully qualified module name install() recorded, `numpy` or
    `jax.numpy`, which is the same name its constructor registrations carry
    after the `--` in `zeros--numpy`. `ops` answers the roster beside it, and
    the row behind both is `(array-backend <space> <library> (ops ...))` in
    `&metta`.
    """
    return import_module('metta_arrays').backend(m)


def register() -> None:
    """Publish this package's complete accessor declaration atomically."""
    seam.door.register('metta-arrays', doors=_doors.declarations(__name__))


register()
