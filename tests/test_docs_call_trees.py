"""Keep the explanatory docs honest: docs/CALL_TREES.md and the examples written into docstrings.

Two guards:
  * every `name()` written in docs/CALL_TREES.md is still a function or method somewhere in the package, so a
    rename or removal that leaves the map wrong fails here;
  * every `>>>` example in a docstring still runs and gives the output it shows.
"""
import ast
import doctest
import importlib
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
PACKAGE = ROOT / "cody_ai_shared_lib"
CALL_TREES = ROOT / "docs" / "CALL_TREES.md"

# Names written as a call in the docs, e.g. `_begin()` or `generator.generate()` (the part after the last dot)
CALLED_NAME = re.compile(r"\b([A-Za-z_][A-Za-z0-9_]*)\(\)")


def defined_names() -> set[str]:
    """Every function and method name defined anywhere in the package."""
    names: set[str] = set()
    for path in PACKAGE.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        names.update(
            node.name for node in ast.walk(tree) if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        )
    return names


def test_every_name_in_the_call_trees_exists():
    written = set(CALLED_NAME.findall(CALL_TREES.read_text(encoding="utf-8")))
    assert written, "CALL_TREES.md should name functions as `name()`"
    missing = sorted(written - defined_names())
    assert not missing, f"docs/CALL_TREES.md names functions that no longer exist: {missing}"


def modules_with_examples() -> list[str]:
    """Dotted names of the package modules whose source contains a `>>>` example."""
    found = []
    for path in sorted(PACKAGE.rglob("*.py")):
        if ">>>" in path.read_text(encoding="utf-8"):
            dotted = ".".join(path.relative_to(ROOT).with_suffix("").parts)
            found.append(dotted.removesuffix(".__init__"))
    return found


@pytest.mark.parametrize("module_name", modules_with_examples())
def test_docstring_examples_run(module_name):
    module = importlib.import_module(module_name)
    result = doctest.testmod(module, verbose=False)
    assert result.failed == 0, f"{result.failed} docstring example(s) in {module_name} no longer match"
