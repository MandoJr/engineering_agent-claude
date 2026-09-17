"""Structured patch implementer: model intent -> validated preview -> atomic apply."""
from __future__ import annotations

from typing import List

from .backend import BackendError, ModelBackend
from .models import ChangeOperation, ChangeSet, EngineeringPlan, FileEdit, ImplementationResult, StructuredChange
from .patching import PatchApplier, PatchValidationError, parse_change_set
from .tools import Permission, ToolBox

IMPLEMENTER_SYSTEM_PROMPT = """You are the Coding Implementer. Return ONLY JSON, never a whole rewritten existing file.
For one approved task, return {"changes":[...], "rationale":"..."}. Each change must use one of:
CREATE_FILE (content is full new file), REPLACE_TEXT (expected_content and content),
INSERT_BEFORE/INSERT_AFTER (expected_content anchor and content), REPLACE_SYMBOL (target_symbol and content),
DELETE_REGION (expected_content), DELETE_FILE, MOVE_FILE (target_file). Expected context must be exact and unique.
Make the smallest patch possible. Do not edit files beyond the requested task file."""

class StructuredCodingImplementer:
    def __init__(self, backend: ModelBackend, tools: ToolBox):
        if tools.permission not in (Permission.IMPLEMENT, Permission.COMMIT):
            raise PermissionError("StructuredCodingImplementer requires IMPLEMENT permission")
        self.backend, self.tools = backend, tools
        self.applier = PatchApplier(tools)

    def implement(self, run_id: str, plan: EngineeringPlan) -> ImplementationResult:
        changes = []
        try:
            for task in plan.changes:
                if task.change_type == "DELETE":
                    changes.append(StructuredChange(file=task.file, operation=ChangeOperation.DELETE_FILE.value,
                                                     reason=task.description, task_id=plan.plan_id, risk=task.risk))
                    continue
                current = self.tools.read_file(task.file) if task.change_type != "CREATE" else None
                current_text = current.data["content"] if current and current.ok else ""
                if task.change_type != "CREATE" and (not current or not current.ok):
                    return ImplementationResult(run_id=run_id, success=False, error=f"Cannot patch '{task.file}': {current.error}")
                patch_set = self._generate_patch(task.file, task.description, task.objective, current_text, plan.plan_id)
                if any(change.file != task.file for change in patch_set.changes):
                    return ImplementationResult(run_id=run_id, success=False, error=f"Patch for '{task.file}' attempted another task file")
                changes.extend(patch_set.changes)
        except (BackendError, PatchValidationError) as exc:
            return ImplementationResult(run_id=run_id, success=False, error=f"Patch generation failed: {exc}")

        patch_results, success = self.applier.apply(ChangeSet(changes=changes, task_id=plan.plan_id), plan.files)
        edits = [FileEdit(file=result.file, change_type=result.operation, diff=result.diff)
                 for result in patch_results if result.applied]
        error = None if success else "; ".join(result.message for result in patch_results if result.message)
        return ImplementationResult(run_id=run_id, edits=edits, patch_results=patch_results,
            backend_used=self.backend.name, success=success, error=error,
            rollback_performed=any(result.rollback_performed for result in patch_results))

    def _generate_patch(self, file: str, description: str, objective: str, current: str, task_id: str) -> ChangeSet:
        prompt = f"""APPROVED TASK FILE: {file}
TASK ID: {task_id}
OBJECTIVE: {objective or description}
CHANGE DESCRIPTION: {description}
CURRENT FILE CONTENT (read-only context; do not return a rewritten file):
---
{current if current else '(new file)'}
---
Return the minimal structured patch JSON now."""
        raw = self.backend.complete(prompt, system=IMPLEMENTER_SYSTEM_PROMPT, json_mode=True)
        return parse_change_set(raw, task_id=task_id, default_file=file)
