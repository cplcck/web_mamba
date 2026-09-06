"""Exercise real test imports and discovery in fresh Python processes."""
from pathlib import Path
import subprocess
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
PROBE = r"""
from collections import Counter
import importlib
import json
from pathlib import Path
import sys
import unittest

root = Path.cwd()
foreign = Path("/home/cplcck/llama.cpp-ssm/gguf-py")
sys.path.insert(0, "tests")
assert all(Path(entry).resolve() != foreign for entry in sys.path), sys.path

if sys.argv[1] == "discovery":
    expected = {path.stem: path.resolve()
                for path in (root / "tests").glob("test_*.py")}
    loader = unittest.TestLoader()
    suite = loader.discover("tests", pattern="test_*.py")
    assert not loader.errors, loader.errors

    def cases(suite):
        for item in suite:
            if isinstance(item, unittest.TestSuite):
                yield from cases(item)
            else:
                yield item

    collected = list(cases(suite))
    assert not any(isinstance(case, unittest.loader._FailedTest)
                   for case in collected), collected
    ids = [case.id() for case in collected]
    assert ids and all(ids), ids
    assert len(ids) == len(set(ids)), ids
    counts = Counter(case.__class__.__module__ for case in collected)
    assert set(counts) == set(expected), (counts, expected)
    assert {"test_audit_gguf", "test_export_web", "test_export_privacy"} <= set(counts)
    for name, path in expected.items():
        assert Path(sys.modules[name].__file__).resolve() == path, name
    print(json.dumps({"modules": dict(sorted(counts.items())), "total": len(ids)}))
else:
    for name in sys.argv[1:]:
        before = sys.path.copy()
        module = importlib.import_module(name)
        assert Path(module.__file__).resolve() == root / "tests" / (name + ".py")
        if name == "test_audit_gguf":
            assert sys.path == before, (before, sys.path)

import tests
from tests import validation_fixtures
import gguf

assert Path(tests.__file__).resolve() == root / "tests/__init__.py"
assert Path(validation_fixtures.__file__).resolve() == root / "tests/validation_fixtures.py"
assert Path(gguf.__file__).resolve().parent == foreign / "gguf"
assert all(Path(entry).resolve() != foreign for entry in sys.path), sys.path
"""


class ImportOrder(unittest.TestCase):
    def probe(self, *arguments: str) -> None:
        result = subprocess.run(
            [sys.executable, "-B", "-c", PROBE, *arguments],
            cwd=ROOT, capture_output=True, text=True, timeout=60,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        if result.stdout:
            print(result.stdout, end="")

    def test_imports_resolve_locally_in_both_orders(self) -> None:
        for order in (
            ("test_audit_gguf", "test_export_web", "test_export_privacy"),
            ("test_export_web", "test_export_privacy", "test_audit_gguf"),
        ):
            with self.subTest(order=order):
                self.probe(*order)

    def test_discovery_collects_every_current_test_without_foreign_path_leak(self) -> None:
        self.probe("discovery")
