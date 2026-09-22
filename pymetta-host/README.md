# pymetta-host

The patched SWI-Prolog host [PyMeTTa](https://pypi.org/project/PyMeTTa/)'s
engine runs on, bundled as a wheel, with the Python bridge built against it.

```sh
pip install pymetta-host
```

You do not normally install this yourself: `pip install 'pymetta[engine]'`
brings it. Install it directly only to get the host without the library.

## Why it exists

PyMeTTa's engine needs a patched SWI. A stock host shows fourteen of the
defects the MeTTa repository documents where the patched host shows five,
and one of the nine a stock host adds aborts the process on thread join
rather than failing quietly. This ships the patched host so that is not
something you have to build.

## What you get

```python
import pymetta_host

pymetta_host.activate()      # point this process at the bundled host
pymetta_host.SWI_HOME        # its home, wherever pip unpacked it
pymetta_host.SWIPL           # the launcher, if you want the command line
```

`activate()` selects the bridge and the home as ONE pair, or refuses. A janus
built against another SWI driving this home, and this bridge driving another
SWI, are the same ABI mismatch wearing opposite halves, and both surface later
as a crash with nothing pointing back here. So with `SWI_HOME_DIR` already set
to something else it raises and names the remedy, rather than putting the
bundled bridge on `sys.path` while a foreign home answers its calls. To use
your own host, unset `SWI_HOME_DIR` or do not call `activate()`.

## What is in the wheel

The SWI home, 25 MB of it, and the bridge vendored under `_vendor/` the way pip
carries its own dependencies under `pip._vendor`: two distributions owning the
import name `janus_swi` is last-writer-wins on install and a broken sibling on
uninstall, and renaming is not the escape because janus names that package in
its Prolog half too and its extension exports `PyInit__swipl`.

Every library the host needs travels with it, twenty of them under
`pymetta_host.libs/` with content-hashed names, which is `auditwheel repair`'s
own layout. Without that the wheel borrows `libgmp`, `libncurses`, `libyaml`,
`libarchive` and eight more from whatever machine it lands on. SWI is built
against OpenSSL 3.5.5 rather than the 1.1.1k the manylinux image ships, since
that reached end of life in 2023 and `lib_http` reaches the TLS plugin.

A wheel is per-interpreter and per-platform here, because the bridge is an
extension module and the home is an ELF tree for one architecture.
