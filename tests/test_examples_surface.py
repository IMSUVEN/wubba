import ast
from dataclasses import fields
from pathlib import Path

import wubba
from wubba import inference as inference_module
from wubba.config import Config

EXAMPLES_DIR = Path(__file__).resolve().parent.parent / "examples"
CONFIG_FIELD_NAMES = {field.name for field in fields(Config)}
WUBBA_EXPORTS = set(getattr(wubba, "__all__", []))


def iter_example_trees() -> list[tuple[Path, ast.AST]]:
    trees: list[tuple[Path, ast.AST]] = []
    for path in sorted(EXAMPLES_DIR.glob("*.py")):
        trees.append((path, ast.parse(path.read_text(encoding="utf-8"))))
    return trees


def test_example_imports_match_public_api() -> None:
    for path, tree in iter_example_trees():
        for node in ast.walk(tree):
            if not isinstance(node, ast.ImportFrom):
                continue

            if node.module == "wubba":
                imported = {alias.name for alias in node.names}
                missing = imported - WUBBA_EXPORTS
                assert not missing, f"{path.name} imports missing wubba exports: {sorted(missing)}"

            if node.module == "wubba.inference":
                imported = {alias.name for alias in node.names}
                missing = {name for name in imported if not hasattr(inference_module, name)}
                assert not missing, f"{path.name} imports missing inference symbols: {sorted(missing)}"


def test_example_config_attribute_accesses_match_config_fields() -> None:
    allowed_non_fields = {"__dict__"}

    for path, tree in iter_example_trees():
        invalid_attrs: set[str] = set()
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Attribute)
                and isinstance(node.value, ast.Name)
                and node.value.id == "config"
                and node.attr not in CONFIG_FIELD_NAMES
                and node.attr not in allowed_non_fields
            ):
                invalid_attrs.add(node.attr)

        assert not invalid_attrs, (
            f"{path.name} references unknown Config attributes: {sorted(invalid_attrs)}"
        )
