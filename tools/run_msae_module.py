"""Launcher for msae/train.py and msae/sae_naming.py.

Those scripts mix package-relative imports (`from .utils import ...` inside
msae/sae.py) with top-level imports (`from sae import ...`, `from metrics import
...`), so neither `python msae/train.py` nor `python -m msae.train` works on its
own. This launcher registers the package submodules under their top-level names,
then runs the target module as __main__ so every import resolves.

No repo files are modified.

Usage:
    python tools/run_msae_module.py msae.train      <train.py args...>
    python tools/run_msae_module.py msae.sae_naming <sae_naming.py args...>
"""
import importlib
import os
import runpy
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "msae"))
os.chdir(REPO)                      # so relative result/embedding paths resolve

if len(sys.argv) < 2:
    sys.exit("usage: python tools/run_msae_module.py <module e.g. msae.train> <args...>")

module = sys.argv[1]
for name in ("utils", "sae", "metrics", "config"):
    try:
        sys.modules[name] = importlib.import_module(f"msae.{name}")
    except Exception:
        pass

sys.argv = [f"msae/{module.split('.')[-1]}.py"] + sys.argv[2:]
runpy.run_module(module, run_name="__main__", alter_sys=True)
