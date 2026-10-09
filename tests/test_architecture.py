"""The dependency direction, as a rule rather than an intention.

The package's import graph is already the right shape, and nothing was holding it that way.
Everywhere else this project turns a declared rule into a test -- the configuration against the
selection's own record, the master scale against `docs/rules.md`, the give-up ladder against the
order the rules declare -- and the layering was the one declaration nobody checked.

What is checked is the direction, not the shape: a module may import whatever it is below, and
nothing may reach up. The edges are read at **both scopes**, because this package uses
function-level imports heavily and on purpose -- importing `data.panel` at module scope put
**0.85 s** on every `creditsurv --help` (`cli.py`, and `test_smoke.py` holds the consequence) --
so a graph built from module-level imports alone would miss most of `cli.py` and all of
`explore.py`'s reach into `models`.

`docs/architecture.md` is the prose this file enforces.
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

import creditsurv

if TYPE_CHECKING:
    from collections.abc import Iterator

PACKAGE = "creditsurv"
ROOT = Path(creditsurv.__file__).parent

#: The layers, deepest first. A module may import from its own layer or any deeper one, and
#: never from a shallower one. The names are the package's own and the order is the one the
#: data moves in: raw files become cells, cells become fits, fits become views, views become
#: a site, and the command line sits above all of it.
LAYERS: tuple[tuple[str, ...], ...] = (
    # The floor: no `creditsurv` imports at all, or only the floor.
    ("config", "names", "features"),
    ("data",),
    # Exploratory statistics over frames the data layer produces. Above `data` because it
    # speaks its vocabulary -- `loan_months` is the cell table's weight column -- and below
    # `models` because a description of the book must not depend on a fit of it.
    ("explore",),
    ("models",),
    ("backtest",),
    ("portfolio", "profiling"),
    ("views", "reporting"),
    ("site",),
    ("cli",),
)


def _modules() -> list[str]:
    """Every module of the package, as a dotted name relative to it."""
    found = []
    for path in sorted(ROOT.rglob("*.py")):
        parts = path.relative_to(ROOT).with_suffix("").parts
        if parts[-1] == "__init__":
            parts = parts[:-1]
        if parts:
            found.append(".".join(parts))
    return found


def _imported(source: str) -> Iterator[str]:
    """Every `creditsurv` module this source names, at any scope.

    Both `import creditsurv.x` and `from creditsurv.x import y`, and both at module level and
    inside a function -- the walk does not care, which is the point. A `from ... import` of a
    *module* rather than a name is counted as an edge to the module, because that is what it is.
    """
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == PACKAGE or alias.name.startswith(f"{PACKAGE}."):
                    yield alias.name
        elif isinstance(node, ast.ImportFrom) and node.module:
            if node.module == PACKAGE:
                for alias in node.names:
                    yield f"{PACKAGE}.{alias.name}"
            elif node.module.startswith(f"{PACKAGE}."):
                yield node.module
                for alias in node.names:
                    yield f"{node.module}.{alias.name}"


def _edges() -> dict[str, set[str]]:
    """Which modules each module imports, as dotted names inside the package.

    An import of a *name* from a module gives an edge to the module, not to the name, so
    `from creditsurv.data.panel import WEIGHT` and `from creditsurv.data import panel` are the
    same edge. Names that are not modules are dropped by the membership test.
    """
    known = set(_modules())
    graph: dict[str, set[str]] = {}
    for module in known:
        path = ROOT.joinpath(*module.split("."))
        source = path.with_suffix(".py") if path.is_dir() is False else path / "__init__.py"
        named = set(_imported(source.read_text()))
        reached = {
            candidate[len(PACKAGE) + 1 :]
            for candidate in named
            if candidate[len(PACKAGE) + 1 :] in known
        }
        graph[module] = reached - {module}
    return graph


def _layer(module: str) -> int:
    """Which layer a module belongs to, by its first path component."""
    head = module.split(".")[0]
    for depth, names in enumerate(LAYERS):
        if head in names:
            return depth
    message = f"{module} is in no declared layer; add it to LAYERS or move it."
    raise AssertionError(message)


def test_every_module_is_in_a_declared_layer() -> None:
    """A module nobody placed is a module nobody has to keep in its place."""
    for module in _modules():
        _layer(module)


def test_nothing_imports_from_a_shallower_layer() -> None:
    """The direction, which is the whole rule.

    `models` may read `data`; `data` may not read `models`. The failure names the offending
    edge, because the useful thing to know is which import to delete, not that the graph is
    wrong.
    """
    upward = [
        f"{module} -> {imported}"
        for module, imports in _edges().items()
        for imported in imports
        if _layer(imported) > _layer(module)
    ]
    assert not upward, "a module reaches up into a layer above it: " + ", ".join(sorted(upward))


def test_no_module_imports_in_a_circle() -> None:
    """There is no import cycle anywhere in the package, at either scope.

    There used to be exactly one, in `data/`: `data.ingest` reads the record layout from
    `data.freddiemac`, which read the event definition from `data.book`, which reads the
    manifest from `data.ingest` -- and that last edge was taken inside the function that needed
    it, by hand. What closed it was a pandas loader nothing in the package called, now
    `tests/freddiemac_sample.py`; `data.freddiemac` imports nothing at all and `data.book`
    imports the manifest at the top of the file like anything else.

    So the rule is the strong one now, and this test exists so that a cycle cannot be
    reintroduced quietly by a convenient function-level import.
    """
    graph = _edges()
    found: set[frozenset[str]] = set()
    path: list[str] = []
    seen: set[str] = set()

    def walk(module: str) -> None:
        if module in path:
            found.add(frozenset(path[path.index(module) :]))
            return
        if module in seen:
            return
        seen.add(module)
        path.append(module)
        for imported in sorted(graph.get(module, ())):
            walk(imported)
        path.pop()

    for module in sorted(graph):
        seen.clear()
        walk(module)

    assert not found, "import cycle(s): " + "; ".join(
        " <-> ".join(sorted(cycle)) for cycle in sorted(found, key=sorted)
    )


@pytest.mark.parametrize("deep", ["models.kernel"])
def test_the_numerical_core_imports_nothing_from_the_package(deep: str) -> None:
    """The written-out likelihood is the deepest leaf, and that is what makes it replaceable.

    It takes arrays and returns arrays: no configuration, no panel, no lifelines, no I/O. A
    compiled implementation of the same arithmetic has to satisfy the same contract, and it can
    only be a drop-in while this holds. The moment the core reads a declared constant it stops
    being a port and becomes part of the model.
    """
    graph = _edges()
    inside = {module for module in graph if module == deep or module.startswith(f"{deep}.")}
    assert inside, f"{deep} is not a module of the package"
    reaching = {
        f"{module} -> {imported}" for module in inside for imported in graph[module] - inside
    }
    assert not reaching, f"{deep} must take arrays and nothing else: " + ", ".join(sorted(reaching))
