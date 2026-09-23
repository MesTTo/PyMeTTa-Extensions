"""Purpose: carry the SWI-Prolog host PyMeTTa's engine needs.

A user does not have to build one.

PyMeTTa's engine runs on a PATCHED SWI. `docs/host-workarounds.md` in the
MeTTa repository records sixty host defects and the twenty-two we carry a
patch for, and a stock host shows fourteen of them against the patched host's
five. One of the nine a stock host adds does not misbehave quietly: the
thread-join window aborts the process on `pl-thread.c: stack_avail___LD:
Assertion failed` [measured 2026-09-22]. So this ships the patched host
rather than asking for one.

SWI is not a single shared library, which is why `pyswip`, `swiplserver` and
the official `janus-swi` all decline to bundle it and require a system
install: it needs its HOME, the boot files and the Prolog library tree and the
foreign object set, found at run time through `SWI_HOME_DIR`. `jdk4py` is the
precedent that does bundle a runtime with a home, shipping a whole JDK as
package data; this home is 25 MB against that one's 180.

The bridge is VENDORED under `_vendor/` rather than installed as the
top-level `janus_swi`, the way pip carries its own dependencies under
`pip._vendor`. Two distributions owning one import name is last-writer-wins
on install and a broken sibling on uninstall, and a rename is not the escape:
janus names `janus_swi` in its Prolog half too, and its extension exports
`PyInit__swipl`, so the name is part of the ABI rather than a label.

Assumes: the wheel for this interpreter and platform. The bridge is an
    extension module, so a wheel is per-interpreter by construction.
Guarantees:
  - `activate()` is idempotent, and selects the bridge and the home as one
    pair or refuses: a bundled bridge driving a foreign home is the same ABI
    mismatch as a foreign bridge driving this one
  - `SWI_HOME` and `SWIPL` resolve inside this package wherever pip unpacked
    it: every RPATH is `$ORIGIN`-relative, so nothing depends on the path it
    was built at
Open Obligations:
  To Do: None
  Hacks: None
  Future Enhancements: None
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

__version__ = "0.9.1"

_HERE = Path(__file__).resolve().parent

#: Where SWI looks for its boot files, library tree and foreign objects.
SWI_HOME = _HERE / "swipl" / "lib" / "swipl"

def _launcher() -> Path:
    """The real launcher inside the home, found rather than spelled.

    NOT `swipl/bin/swipl`: that is a symlink in an installed SWI, and building
    a wheel dereferences symlinks, so it arrives as a second copy of the
    binary at a depth its own RPATH does not match. One binary, at the path
    the home actually keeps it, and the architecture directory is globbed
    because naming it would be one more thing to be wrong about on the next
    platform.
    """
    found = next((SWI_HOME / "bin").glob("*/swipl"), None)
    if found is None:
        msg = (
            f"no launcher under {SWI_HOME / 'bin'}: this package's home is "
            f"incomplete, which a wheel should make impossible. Reinstall "
            f"pymetta-host."
        )
        raise RuntimeError(msg)
    return found


#: The launcher, for a caller who wants the command line rather than the bridge.
SWIPL = _launcher()

_VENDOR = _HERE / "_vendor"


def activate() -> Path:
    """Point this process at the bundled host, and answer its home.

    The bridge and the home are ONE build, so this selects both or refuses.
    A janus built against another SWI, driving this home, and this bridge
    driving another SWI are the same ABI mismatch wearing opposite halves,
    and an earlier version of this function refused the first and permitted
    the second through `setdefault`.

    The vendor directory goes on `sys.path` ahead of site-packages so the
    bridge built against THIS home wins over one built against another. It
    holds nothing but the bridge, so that precedence reaches nothing else.

    Both refusals are checked before the import rather than after, because
    `sys.path` cannot undo one: `sys.modules` is consulted first, so a path
    inserted afterwards changes nothing. What the refusal prevents is a crash
    later and elsewhere, with nothing pointing back here.
    """
    loaded = sys.modules.get("janus_swi")
    if loaded is not None:
        where = Path(getattr(loaded, "__file__", "") or "").resolve()
        if _VENDOR not in where.parents:
            msg = (
                f"janus_swi was already imported from {where or 'an unknown location'}, "
                f"and it was not the bridge this package carries. Pointing the bundled "
                f"home at another build's bridge is an ABI mismatch that fails later and "
                f"elsewhere, so it is refused here instead. Either import pymetta_host "
                f"before anything imports janus_swi, or uninstall janus_swi and let this "
                f"package supply it."
            )
            raise RuntimeError(msg)

    chosen = os.environ.get("SWI_HOME_DIR")
    if chosen is not None and Path(chosen).resolve() != SWI_HOME:
        msg = (
            f"SWI_HOME_DIR names {chosen}, and this package carries its own host at "
            f"{SWI_HOME}. The bridge and the home are one build: activate() would put "
            f"the bundled bridge on sys.path while that home answered its calls, which "
            f"is the same ABI mismatch as importing a foreign bridge, and it surfaces "
            f"as a crash with nothing pointing back here. Unset SWI_HOME_DIR to use the "
            f"bundled host, or do not call activate() and let your own host and bridge "
            f"pair up."
        )
        raise RuntimeError(msg)

    if loaded is None:
        os.environ["SWI_HOME_DIR"] = str(SWI_HOME)
        vendor = str(_VENDOR)
        if vendor not in sys.path:
            sys.path.insert(0, vendor)
    return SWI_HOME
