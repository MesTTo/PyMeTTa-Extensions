"""Purpose: register the arrays accessor using deferred implementation references.

Guarantees: registration imports no implementation and accessor calls preserve
  its behavior [tested: test_array_namespace_preserves_installation_and_withdrawal; commit=WORKTREE].
"""

from __future__ import annotations

from metta import seam
from metta.doors import (
    AnswersAs,
    Body,
    Door,
    Kind,
    Owner,
    Provider,
    Receiver,
    Signature,
    Tier,
)
from metta.vocabularies import Determinism, EffectClass

# closed-set: decides; policy=this package owns these accessor contracts; reads=the named implementation signatures checked by tools/doorgen.py
DOORS: tuple[Door, ...] = (
    Door(
        owner=Owner.namespace, name='install', kind=Kind.provider,
        signatures=(Signature('m, default: Any=None', returns='list[str]'),), answers=AnswersAs.sequence,
        effect=EffectClass.oracleIO, determinism=Determinism.det,
        tiers=(Tier.sync, Tier.context), body=Body('metta_arrays', 'install', Receiver.space),
        provider=Provider('metta-arrays', 'arrays'),
        docs="Register the array operation set on the shared engine.\n\ndefault names the library the constructors build in: a module, a module\nname, or None for NumPy. Every other operation dispatches on its\nargument's own namespace, so arrays from any conforming library flow\nthrough the same MeTTa functions, and a mixed binary call converts the\nright operand into the left's library through from_dlpack.\n\nEvery installed name has one or more arrow declarations. Constructors\nwith optional or variadic dimensions have one arrow per accepted arity.\n\nbroadcast-shape is the CLP(FD) relation over shape expressions. It can\ncompute a result before any tensor exists, infer an unknown input\ndimension from a required result, or reject incompatible shapes:\n\n    !(let True (broadcast-shape (4 1) (3) $shape) $shape)  ; (4 3)\n    !(let True (broadcast-shape ($d 1) (1 3) (4 3)) $d)   ; 4\n    !(broadcast-shape (2 3) (4 3) (4 3))                  ; no answer\n\nt-shape remains observation of an existing tensor. Use broadcast-shape\nwhen compatibility or inference must happen before materialisation.\n\n``Shape`` carries those expressions through Python ``Annotated`` claims.\nUser operation arrows retain the claim: argument dimensions unify and\nbind shared result dimensions. Live values report the same type expression.\n``SHAPE_RULES`` names every installed head's behavior; preserving heads\nshare their entire input shape with the result. Other transformations\nexpose their actual shape when their result value exists.\nA declared ``(Annotated DLTensor (Shape ...))`` remains a valid DLTensor\nargument, elementwise binary operations derive their output shape with\n``broadcast-shape``, and rank-two matmul unifies the two inner dimensions:\n\n    (: image (Annotated DLTensor (Shape (4 1))))\n    (: bias  (Annotated DLTensor (Shape (3))))\n    !(get-type (t+ image bias))  ; (Annotated DLTensor (Shape (4 3)))\n\nm may be a context or a space. The operations are registered into the\nspace either way, which is the object whose storage and introspection\ndoors this needs; `install(m)` on a context used to raise\n`MeTTa has no 'is_function'` and leave every operation unregistered.\n\nWhat this space installed becomes one catalog row,\n``(array-backend <space> <library> (ops ...))`` in ``&metta``, which\n``ops(m)`` and ``backend(m)`` read back and a MeTTa program can match for\nitself. Installing again REPLACES that row, the space's constructor\naliases, and every operation of the outgoing roster that no other space's\nrow still names; ``uninstall(m)`` retires the whole installation, and\ndropping the space retires the row with it. Two spaces may therefore hold\ntwo libraries at once, in either install order, each answering its own.",
        evidence=('extensions/python/ext/metta-arrays/tests/test_arrays_doors.py::test_array_namespace_preserves_installation_and_withdrawal',),
    ),
    Door(
        owner=Owner.namespace, name='uninstall', kind=Kind.provider,
        signatures=(Signature('m', returns='list[str]'),), answers=AnswersAs.sequence,
        effect=EffectClass.oracleIO, determinism=Determinism.det,
        tiers=(Tier.sync, Tier.context), body=Body('metta_arrays', 'uninstall', Receiver.space),
        provider=Provider('metta-arrays', 'arrays'),
        docs="Retire this space's array installation; answers what it unregistered.\n\nThe inverse of `install`. The space's roster row goes, with its\nconstructor aliases and the bare-name arrows those carried, the `get-type`\nshape equations and the shaped-DLTensor typing rule; then every operation\nthe roster named is unregistered UNLESS another space's row still names\nit, because the operation registry is process-wide by name and two spaces\non numpy share `zeros--numpy` and the whole backend-agnostic set. The\nanswer is therefore what actually left the registry, in roster order.\n\n    arrays.install(space, default=numpy)\n    arrays.uninstall(space)\n    arrays.ops(space)            # refuses: nothing is installed here\n\nAn operation another space claims stays registered, and this space stops\nDECLARING it: `ops.withdraw` releases the rows that would otherwise keep\nthe space describing a function it no longer routes to.\n\nDropping the space retires the row without this call, the catalog\nretiring a space's declarations with it, but the process-wide operations\nare the registry's and only this door hands them back.\n\nTwo registrations deliberately survive, both keyed on the DLPack\npredicate rather than on a space, so one registration serves every space\nand withdrawing it here would change another space's answers: the\nDLTensor type and array printing hooks, whose own doors are\n`integrate.unregister_object_type` and `integrate.unregister_repr`, and\nthe `broadcast-shape` CLP(FD) relation, which `register_prolog` has no\nwithdrawal for.\n\nm may be a context or a space, as `install` takes either.",
        evidence=('extensions/python/ext/metta-arrays/tests/test_arrays_doors.py::test_array_namespace_preserves_installation_and_withdrawal',),
    ),
    Door(
        owner=Owner.namespace, name='ops', kind=Kind.introspection,
        signatures=(Signature('m', returns='list[str]'),), answers=AnswersAs.sequence,
        effect=EffectClass.readOnlyLookup, determinism=Determinism.det,
        tiers=(Tier.sync, Tier.context), body=Body('metta_arrays', 'ops', Receiver.space),
        provider=Provider('metta-arrays', 'arrays'),
        docs="The array operation names installed in this space, in install order.\n\n`install` returns the same list; this reads it back from the space long\nafterwards, so two spaces on two libraries answer their own rosters\nwhatever order they were installed in:\n\n    numpy_space, jax_space = m.space(), m.space()\n    arrays.install(jax_space, default=jax.numpy)\n    arrays.install(numpy_space, default=numpy)\n    arrays.ops(numpy_space)      # ... 'zeros--numpy' ...\n    arrays.backend(jax_space)    # 'jax.numpy'\n\nm may be a context or a space. The longhand is the row itself, which is\nordinary matchable data: `!(match &metta (array-backend &s $lib $ops) $ops)`.\nA space with no install refuses, naming install as the remedy.",
        evidence=('extensions/python/ext/metta-arrays/tests/test_arrays_doors.py::test_array_namespace_preserves_installation_and_withdrawal',),
    ),
    Door(
        owner=Owner.namespace, name='backend', kind=Kind.introspection,
        signatures=(Signature('m', returns='str'),), answers=AnswersAs.text,
        effect=EffectClass.readOnlyLookup, determinism=Determinism.det,
        tiers=(Tier.sync, Tier.context), body=Body('metta_arrays', 'backend', Receiver.space),
        provider=Provider('metta-arrays', 'arrays'),
        docs="The array library this space's constructors build in.\n\nThe fully qualified module name install() recorded, `numpy` or\n`jax.numpy`, which is the same name its constructor registrations carry\nafter the `--` in `zeros--numpy`. `ops` answers the roster beside it, and\nthe row behind both is `(array-backend <space> <library> (ops ...))` in\n`&metta`.",
        evidence=('extensions/python/ext/metta-arrays/tests/test_arrays_doors.py::test_array_namespace_preserves_installation_and_withdrawal',),
    ),
)


def register() -> None:
    """Publish this package's complete accessor declaration atomically."""
    seam.door.register('metta-arrays', doors=DOORS)


register()
