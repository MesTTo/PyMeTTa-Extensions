"""Purpose: prove an installed distribution owns and loads its generated face.

Owns resources: pytest removes the copied build tree, wheel and installation;
the child process owns its engine and import hooks.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tomllib
import zipfile
from pathlib import Path

from _workspace import ROOT

PACKAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests/checks"))

from bounded_spawn import bounded  # noqa: E402 -- repository process policy


def test_built_wheel_loads_its_generated_library(tmp_path):
    """Build the real package and execute the advertised file from that wheel."""
    source, wheels, site = (tmp_path / name for name in ("source", "wheels", "site"))
    source.mkdir()
    metadata = tomllib.loads((PACKAGE / "pyproject.toml").read_text(encoding="utf-8"))
    config = metadata["tool"]["setuptools"]
    for name in ("pyproject.toml", metadata["project"]["readme"],
                 *(f"{name}.py" for name in config["py-modules"])):
        shutil.copy2(PACKAGE / name, source / name)
    for name in config["packages"]:
        shutil.copytree(PACKAGE / name, source / name,
                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    built = subprocess.run(
        bounded([sys.executable, "-m", "build", "--wheel", "--no-isolation",
                 "--outdir", str(wheels), str(source)]),
        capture_output=True, text=True, check=False,
    )
    assert built.returncode == 0, built.stdout + built.stderr
    wheel, = wheels.glob("*.whl")
    with zipfile.ZipFile(wheel) as archive:
        archive.extractall(site)
    code = '''
import importlib
from pathlib import Path
import numpy as np
from metta import G, MeTTa, importing
from metta.integrate import LIBRARIES_GROUP, load_entry_point

directory = Path(load_entry_point("lib_arrays", group=LIBRARIES_GROUP))
assert directory.is_relative_to(Path(__import__("sys").argv[1]))
with MeTTa() as context:
    with importing.install(context.self, path=[]):
        module = importlib.import_module("lib_arrays")
        assert Path(module.__file__).parent == directory
        assert context.eval(module.arrays_is_array(G(np.arange(3)))) == [True]
        assert context.eval(module.arrays_is_array((1, 2, 3))) == [False]
print("wheel library: installed metadata, generated face and both answers verified")
'''
    environment = dict(os.environ, PYTHONPATH=os.pathsep.join((str(site), str(ROOT / "extensions/python"))))
    checked = subprocess.run(bounded([sys.executable, "-c", code, str(site)]),
                             cwd=tmp_path, env=environment,
                             capture_output=True, text=True, check=False)
    assert checked.returncode == 0, checked.stdout + checked.stderr
