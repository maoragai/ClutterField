"""Smoke tests that all skeleton subpackages import cleanly."""

from __future__ import annotations

import importlib

import clutterfield

_SUBPACKAGES = [
    "clutterfield.encoders",
    "clutterfield.fields",
    "clutterfield.radar",
    "clutterfield.simulator",
    "clutterfield.detector",
    "clutterfield.datasets",
    "clutterfield.losses",
    "clutterfield.visualization",
]


def test_version_is_defined() -> None:
    """The package should expose a version string."""
    assert isinstance(clutterfield.__version__, str)


def test_all_subpackages_import() -> None:
    """Every architecture subpackage should be importable."""
    for module_name in _SUBPACKAGES:
        importlib.import_module(module_name)
