"""Purpose: a pydantic model as a constructor expression, in one row.

A model with no explicit conversion registered still has to cross: its fields
are the arguments of an expression named for the class, and rebuilding runs
pydantic's own validation. One row against the seat's `image` point, which is
an ownership point consulted before the four structural readings this seat
ships as fallbacks -- a validated model is also a class with `__match_args__`,
so the specific reading has to win, and once the two live in separate
distributions no load order can be relied on to arrange that.

Detected through sys.modules: if pydantic was never imported, no BaseModel
subclass can exist, so recognising a class costs a dictionary lookup and this
row imports nothing.

Assumes:
  - `model_fields` names the model's declared fields in declaration order,
    which is pydantic v2's own API
Guarantees:
  - a model projects as `(Model field ...)` and rebuilds through
    `model_validate(..., by_name=True)`, so a field declared with an alias
    validates from the attribute names the projection read
    [tested: tests/test_pydantic.py; commit=94057a0f073c0fab0a35c42beff2c324d8a0addd]
  - a model carrying pydantic EXTRA fields is refused naming them, rather than
    projected with them silently dropped [tested:
    tests/test_pydantic.py::test_extra_fields_refuse_rather_than_vanish;
    commit=94057a0f073c0fab0a35c42beff2c324d8a0addd]
  - this row is asked before the structural fallbacks whatever order the two
    registered in [tested:
    tests/test_pydantic.py::test_a_model_is_not_read_as_a_match_args_class;
    commit=94057a0f073c0fab0a35c42beff2c324d8a0addd]
Open Obligations:
  To Do: None
  Hacks: None
  Future Enhancements: None
"""

from __future__ import annotations

import sys
from typing import Any

from metta import seam

_field_types = seam.at("field-types").call()


def _image(cls: type) -> Any:
    """A pydantic model is a constructor expression like a dataclass.

    Its fields are read from model_fields and its rebuild goes through the
    class itself, so validation runs exactly where pydantic runs it.
    """
    pydantic = sys.modules.get("pydantic")
    if pydantic is None or not issubclass(cls, pydantic.BaseModel):
        return None
    model_cls: Any = cls
    names = tuple(model_cls.model_fields.keys())

    def parts(obj: Any) -> tuple[Any, ...]:
        extras = getattr(obj, "__pydantic_extra__", None)
        if extras:
            extra_names = ", ".join(sorted(map(str, extras)))
            msg = (
                f"cannot project {cls.__name__}: its Pydantic extra fields "
                f"would be lost ({extra_names}). Declare those fields on "
                f"the model or register an explicit conversion."
            )
            raise TypeError(msg)
        return tuple(getattr(obj, name) for name in names)

    return seam.image_of(
        "expression",
        parts,
        # model_validate with by_name, not cls(**...): a field declared with
        # an alias validates under the alias in the constructor, while
        # projection read attribute names, and by_name accepts them directly.
        lambda *values: model_cls.model_validate(
            dict(zip(names, values, strict=True)), by_name=True
        ),
        cls.__name__,
        fields=names,
        types=_field_types(cls, names),
    )


def register() -> None:
    """This package's one row, against the seat's `image` point."""
    seam.image.register("pydantic", claims=_image)


register()
