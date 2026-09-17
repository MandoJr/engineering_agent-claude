"""Structured, diff-first patch preparation and application.

This module is deliberately independent of model prompting.  It accepts only
small change operations, validates them against the current repository state,
produces a preview, then applies a prevalidated changeset with best-effort
rollback and explicit partial-state evidence.
"""
from __future__ import annotations

import ast
import difflib
import json
import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

from .models import ChangeOperation, ChangeSet, PatchResult, StructuredChange
from .tools import ToolBox

class PatchValidationError(ValueError):
    def __init__(self, classification: str, message: str):
        self.classification = classification
        super().__init__(message)

@dataclass
class PreparedChange:
    change: StructuredChange
    before: Optional[str]
    after: Optional[str]
    diff: str

@dataclass
class PreparedChangeSet:
    changes: List[PreparedChange]
    initial: Dict[str, Optional[str]]
    final: Dict[str, Optional[str]]

def parse_change_set(raw: str, task_id: str = "", default_file: str = "") -> ChangeSet:
    """Parse strictly shaped model JSON; no textual patch guessing or overwrite fallback."""
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip(), flags=re.MULTILINE)
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise PatchValidationError("MALFORMED_PATCH", f"Patch response is not JSON: {exc}") from exc
    rows = data.get("changes") if isinstance(data, dict) else None
    if not isinstance(rows, list) or not rows:
        raise PatchValidationError("MALFORMED_PATCH", "Patch response must contain a non-empty 'changes' list")
    allowed = {item.value for item in ChangeOperation}
    changes = []
    for row in rows:
        if not isinstance(row, dict):
            raise PatchValidationError("MALFORMED_PATCH", "Each patch change must be an object")
        operation = row.get("operation", ChangeOperation.REPLACE_TEXT.value)
        if operation not in allowed:
            raise PatchValidationError("MALFORMED_PATCH", f"Unsupported operation: {operation}")
        file = row.get("file", default_file)
        if not isinstance(file, str) or not file:
            raise PatchValidationError("MALFORMED_PATCH", "Every patch change needs a target file")
        change = StructuredChange(
            file=file, operation=operation,
            target_symbol=row.get("target_symbol"), expected_content=row.get("expected_content"),
            content=row.get("content", ""), target_file=row.get("target_file"),
            reason=row.get("reason", ""), task_id=row.get("task_id", task_id) or task_id,
            risk=row.get("risk", "LOW"), validation_requirements=list(row.get("validation_requirements", [])),
        )
        if row.get("change_id"):
            change.change_id = row["change_id"]
        changes.append(change)
    return ChangeSet(changes=changes, task_id=task_id, rationale=data.get("rationale", ""))

class PatchValidator:
    """Renders a patch against explicit expected state without writing files."""
    def validate(self, change: StructuredChange, state: Dict[str, Optional[str]], approved_files: Iterable[str]) -> Tuple[Optional[str], Optional[str]]:
        approved = set(approved_files)
        if change.file not in approved:
            raise PatchValidationError("SCOPE_VIOLATION", f"{change.file} is outside approved file scope")
        if change.target_file and change.target_file not in approved:
            raise PatchValidationError("SCOPE_VIOLATION", f"{change.target_file} is outside approved file scope")
        before = state.get(change.file)
        op = change.operation
        if op == ChangeOperation.CREATE_FILE.value:
            if before is not None: raise PatchValidationError("TARGET_EXISTS", f"Cannot create existing file {change.file}")
            after = change.content
        elif op == ChangeOperation.DELETE_FILE.value:
            if before is None: raise PatchValidationError("TARGET_MISSING", f"Cannot delete missing file {change.file}")
            after = None
        elif op == ChangeOperation.MOVE_FILE.value:
            if before is None or not change.target_file: raise PatchValidationError("MALFORMED_PATCH", "MOVE_FILE needs an existing file and target_file")
            if state.get(change.target_file) is not None: raise PatchValidationError("TARGET_EXISTS", f"Move target exists: {change.target_file}")
            if change.expected_content is not None and before != change.expected_content: raise PatchValidationError("STALE_CONTEXT", "Move source differs from expected content")
            return before, None
        else:
            if before is None: raise PatchValidationError("TARGET_MISSING", f"Cannot modify missing file {change.file}")
            after = self._render(change, before)
        if after is not None and change.file.endswith(".py"):
            try: ast.parse(after, filename=change.file)
            except SyntaxError as exc: raise PatchValidationError("SYNTAX_INVALID", f"Patch makes {change.file} invalid Python: {exc}") from exc
        return before, after

    def _render(self, change: StructuredChange, before: str) -> str:
        op, expected = change.operation, change.expected_content
        if op == ChangeOperation.REPLACE_TEXT.value:
            return self._unique_replace(before, expected, change.content)
        if op == ChangeOperation.DELETE_REGION.value:
            return self._unique_replace(before, expected, "")
        if op in (ChangeOperation.INSERT_BEFORE.value, ChangeOperation.INSERT_AFTER.value):
            if not expected: raise PatchValidationError("MALFORMED_PATCH", f"{op} requires expected_content anchor")
            count = before.count(expected)
            if count != 1: raise PatchValidationError("STALE_CONTEXT" if count == 0 else "AMBIGUOUS_CONTEXT", f"Anchor occurs {count} times")
            index = before.index(expected) + (len(expected) if op == ChangeOperation.INSERT_AFTER.value else 0)
            return before[:index] + change.content + before[index:]
        if op == ChangeOperation.REPLACE_SYMBOL.value:
            if not change.target_symbol: raise PatchValidationError("MALFORMED_PATCH", "REPLACE_SYMBOL requires target_symbol")
            start, end = self._symbol_bounds(before, change.file, change.target_symbol)
            region = before[start:end]
            if expected is not None and region != expected:
                raise PatchValidationError("STALE_CONTEXT", "Symbol body differs from expected content")
            return before[:start] + change.content + before[end:]
        raise PatchValidationError("MALFORMED_PATCH", f"Operation {op} cannot modify content")

    @staticmethod
    def _unique_replace(source: str, expected: Optional[str], replacement: str) -> str:
        if not expected: raise PatchValidationError("MALFORMED_PATCH", "Replacement requires expected_content")
        count = source.count(expected)
        if count != 1: raise PatchValidationError("STALE_CONTEXT" if count == 0 else "AMBIGUOUS_CONTEXT", f"Expected context occurs {count} times")
        return source.replace(expected, replacement, 1)

    @staticmethod
    def _symbol_bounds(source: str, filename: str, symbol: str) -> Tuple[int, int]:
        try: tree = ast.parse(source, filename=filename)
        except SyntaxError as exc: raise PatchValidationError("SYNTAX_INVALID", f"Cannot locate symbol in invalid Python: {exc}") from exc
        found = []
        def visit(nodes, parents=()):
            for node in nodes:
                if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                    qualified = ".".join((*parents, node.name))
                    if node.name == symbol or qualified == symbol: found.append(node)
                    visit(getattr(node, "body", ()), (*parents, node.name))
        visit(tree.body)
        if len(found) != 1: raise PatchValidationError("SYMBOL_NOT_FOUND" if not found else "AMBIGUOUS_SYMBOL", f"Symbol '{symbol}' resolved {len(found)} times")
        lines = source.splitlines(keepends=True); node = found[0]
        return sum(len(line) for line in lines[:node.lineno - 1]), sum(len(line) for line in lines[:node.end_lineno])

class PatchApplier:
    def __init__(self, tools: ToolBox, validator: Optional[PatchValidator] = None):
        self.tools, self.validator = tools, validator or PatchValidator()

    def preview(self, changeset: ChangeSet, approved_files: Iterable[str]) -> PreparedChangeSet:
        state: Dict[str, Optional[str]] = {}
        def load(path: str) -> None:
            if path in state: return
            read = self.tools.read_file(path)
            state[path] = read.data["content"] if read.ok else None
        for change in changeset.changes:
            load(change.file)
            if change.target_file: load(change.target_file)
        initial = dict(state); prepared = []; seen = set()
        for change in changeset.changes:
            if change.change_id in seen: raise PatchValidationError("DUPLICATE_PATCH", f"Duplicate change id {change.change_id}")
            seen.add(change.change_id)
            before, after = self.validator.validate(change, state, approved_files)
            if change.operation == ChangeOperation.MOVE_FILE.value:
                target_before = state.get(change.target_file)
                state[change.file], state[change.target_file] = None, before
                diff = _diff(change.file, before, None) + _diff(change.target_file, target_before, before)
            else:
                state[change.file] = after; diff = _diff(change.file, before, after)
            prepared.append(PreparedChange(change, before, after, diff))
        return PreparedChangeSet(prepared, initial, dict(state))

    def apply(self, changeset: ChangeSet, approved_files: Iterable[str]) -> Tuple[List[PatchResult], bool]:
        try: prepared = self.preview(changeset, approved_files)
        except PatchValidationError as exc:
            return [PatchResult(classification=exc.classification, message=str(exc))], False
        results: List[PatchResult] = []; written: List[str] = []
        for path in sorted(prepared.final):
            before, after = prepared.initial.get(path), prepared.final[path]
            if before == after: continue
            if before is None: result = self.tools.create_file(path, after or "")
            elif after is None: result = self.tools.delete_file(path)
            else: result = self.tools.edit_file(path, after)
            if not result.ok:
                rollback = self._rollback(written, prepared.initial)
                results.append(PatchResult(file=path, applied=False, classification="APPLY_FAILED", message=result.error or "write failed", rollback_performed=rollback))
                return results, False
            written.append(path)
        for item in prepared.changes:
            added, removed = _line_counts(item.diff)
            results.append(PatchResult(change_id=item.change.change_id, file=item.change.file,
                operation=item.change.operation, applied=True, classification="APPLIED", diff=item.diff,
                lines_added=added, lines_removed=removed, message=item.change.reason))
        return results, True

    def _rollback(self, paths: List[str], initial: Dict[str, Optional[str]]) -> bool:
        ok = True
        for path in reversed(paths):
            old = initial[path]
            current = self.tools.read_file(path)
            if old is None:
                result = self.tools.delete_file(path)
            elif current.ok:
                result = self.tools.edit_file(path, old)
            else:
                result = self.tools.create_file(path, old)
            ok = ok and result.ok
        return ok

def _diff(path: str, before: Optional[str], after: Optional[str]) -> str:
    return "".join(difflib.unified_diff((before or "").splitlines(keepends=True), (after or "").splitlines(keepends=True), fromfile=path if before is not None else "/dev/null", tofile=path if after is not None else "/dev/null"))

def _line_counts(diff: str) -> Tuple[int, int]:
    return (sum(1 for line in diff.splitlines() if line.startswith("+") and not line.startswith("+++")), sum(1 for line in diff.splitlines() if line.startswith("-") and not line.startswith("---")))
