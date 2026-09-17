"""AST-backed Python repository map used for planning and change-impact analysis."""
from __future__ import annotations

import ast
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Set

@dataclass
class SymbolInfo:
    name: str
    kind: str
    line: int
    parent: str = ""

@dataclass
class ModuleInfo:
    path: str
    module: str
    imports: List[str] = field(default_factory=list)
    symbols: List[SymbolInfo] = field(default_factory=list)
    calls: List[str] = field(default_factory=list)
    parse_error: str = ""

class RepositoryGraph:
    """Conservative structural index: syntax-proven edges, no guessed links."""
    def __init__(self, root: Path):
        self.root = Path(root)
        self.modules: Dict[str, ModuleInfo] = {}
        self.importers: Dict[str, Set[str]] = {}
        self.callers: Dict[str, Set[str]] = {}

    def build(self) -> "RepositoryGraph":
        for path in self.root.rglob("*.py"):
            if any(part in {".git", ".venv", "venv", "__pycache__", "node_modules"} for part in path.parts):
                continue
            self._index_file(path)
        return self

    def _index_file(self, path: Path) -> None:
        rel = path.relative_to(self.root).as_posix()
        module = rel[:-3].replace("/", ".").removesuffix(".__init__")
        info = ModuleInfo(path=rel, module=module)
        try:
            tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"), filename=rel)
        except (SyntaxError, OSError) as exc:
            info.parse_error = str(exc); self.modules[rel] = info; return
        visitor = _Visitor(); visitor.visit(tree)
        info.imports, info.symbols, info.calls = visitor.imports, visitor.symbols, visitor.calls
        self.modules[rel] = info
        for target in info.imports: self.importers.setdefault(target, set()).add(rel)
        for target in info.calls: self.callers.setdefault(target, set()).add(rel)

    def importers_of(self, module: str) -> List[str]:
        found: Set[str] = set()
        for target, files in self.importers.items():
            if target == module or target.startswith(module + "."): found.update(files)
        return sorted(found)

    def callers_of(self, symbol: str) -> List[str]:
        return sorted(self.callers.get(symbol, set()))

    def impacted_by(self, paths: Iterable[str]) -> List[str]:
        impacted = set(paths)
        for path in paths:
            info = self.modules.get(path)
            if not info: continue
            impacted.update(self.importers_of(info.module))
            for symbol in info.symbols: impacted.update(self.callers_of(symbol.name))
        return sorted(impacted)

    def summary(self, paths: Iterable[str]) -> Dict[str, dict]:
        return {path: {"module": info.module, "imports": info.imports,
                       "imported_by": self.importers_of(info.module), "calls": info.calls,
                       "callers": {s.name: self.callers_of(s.name) for s in info.symbols},
                       "parse_error": info.parse_error}
                for path in paths if (info := self.modules.get(path))}

class _Visitor(ast.NodeVisitor):
    def __init__(self): self.imports, self.symbols, self.calls, self.parents = [], [], [], []
    def visit_Import(self, node): self.imports.extend(a.name for a in node.names); self.generic_visit(node)
    def visit_ImportFrom(self, node):
        base = node.module or ""; self.imports.extend(f"{base}.{a.name}".strip(".") for a in node.names); self.generic_visit(node)
    def visit_ClassDef(self, node):
        self.symbols.append(SymbolInfo(node.name, "class", node.lineno, ".".join(self.parents)))
        self.parents.append(node.name); self.generic_visit(node); self.parents.pop()
    def visit_FunctionDef(self, node):
        self.symbols.append(SymbolInfo(node.name, "method" if self.parents else "function", node.lineno, ".".join(self.parents)))
        self.parents.append(node.name); self.generic_visit(node); self.parents.pop()
    visit_AsyncFunctionDef = visit_FunctionDef
    def visit_Call(self, node):
        name = _name(node.func)
        if name: self.calls.append(name)
        self.generic_visit(node)

def _name(node):
    if isinstance(node, ast.Name): return node.id
    if isinstance(node, ast.Attribute):
        base = _name(node.value)
        return f"{base}.{node.attr}" if base else node.attr
    return ""
