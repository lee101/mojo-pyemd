from __future__ import annotations

import importlib.util
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOCAL_PYTHON = os.path.join(ROOT, "python")
sys.path.insert(0, LOCAL_PYTHON)


@pytest.fixture(scope="session")
def upstream_pyemd():
    local_init = os.path.realpath(os.path.join(LOCAL_PYTHON, "pyemd", "__init__.py"))
    for entry in sys.path:
        candidate = os.path.join(entry, "pyemd", "__init__.py")
        if os.path.isfile(candidate) and os.path.realpath(candidate) != local_init:
            spec = importlib.util.spec_from_file_location(
                "_upstream_pyemd",
                candidate,
                submodule_search_locations=[os.path.dirname(candidate)],
            )
            if spec is None or spec.loader is None:
                continue
            module = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = module
            spec.loader.exec_module(module)
            return module
    pytest.fail("upstream pyemd is not installed")
