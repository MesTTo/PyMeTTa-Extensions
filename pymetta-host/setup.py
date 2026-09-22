"""Purpose: declare this wheel impure, so it carries an interpreter tag and a
platform tag.

The vendored bridge is an extension module built against one interpreter's ABI
and one platform's libc, and the SWI home is an ELF tree for one architecture,
so `py3-none-any` would claim to install where it cannot run. Worse than the
claim: all three interpreter builds would write that ONE filename and the last
would overwrite the other two, and auditwheel cannot repair a wheel with no
platform tag at all.

setuptools decides purity from whether the distribution declares extensions,
and these extensions arrive as package DATA rather than through `ext_modules`,
so the distribution has to say so itself. `distclass` is the only hook for it
and has no `pyproject.toml` equivalent, which is why this file exists beside a
complete `pyproject.toml` rather than instead of one.

Assumes: setuptools reads `pyproject.toml` for everything else; this file adds
    one behaviour and configures nothing.
Guarantees:
  - the built wheel names this interpreter and this platform
    [tested: tests/checks/check_host_bundle.py reads the tag of each wheel it
    is given; commit=WORKTREE]
Open Obligations:
  To Do: None
  Hacks: None
  Future Enhancements: None
"""

from setuptools import setup
from setuptools.dist import Distribution


class _Binary(Distribution):
    """A distribution that reports extensions, because it ships them as data."""

    def has_ext_modules(self) -> bool:  # setuptools' own hook name
        return True


setup(distclass=_Binary)
