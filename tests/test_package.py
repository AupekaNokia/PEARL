"""Package skeleton: subpackages import without pulling heavy dependencies."""

from __future__ import annotations

import importlib
import sys

import pytest


@pytest.mark.parametrize("name", ["pearl", "pearl.sensor", "pearl.collect", "pearl.dataset"])
def test_import_subpackage(name: str) -> None:
    assert importlib.import_module(name).__doc__


def test_import_pearl_does_not_load_torch() -> None:
    for mod in [m for m in sys.modules if m == "pearl" or m.startswith("pearl.")]:
        del sys.modules[mod]
    sys.modules.pop("torch", None)
    importlib.import_module("pearl")
    assert "torch" not in sys.modules
