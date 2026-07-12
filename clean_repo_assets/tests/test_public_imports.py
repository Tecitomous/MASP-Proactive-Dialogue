"""Smoke tests for the public package surface."""

import importlib


def test_public_modules_import():
    for module_name in (
        "masp",
        "masp.mind.bdi_schema",
        "masp.data.bdi_dataset",
        "masp.eval.metrics",
    ):
        importlib.import_module(module_name)

