"""
Coding Implementer.

Applies an APPROVED plan's changes one file at a time. For each planned
change it reads current file content (if any), asks the backend for the
new full content of just that file (targeted, not a whole-repo rewrite),
and writes it through the ToolBox so every edit is diffed and permission
checked. Stops and reports on the first hard failure rather than
plowing through remaining files in an inconsistent state.
"""

from __future__ import annotations

from typing import List

from .backend import ModelBackend, BackendError
from .models import EngineeringPlan, FileEdit, ImplementationResult
from .tools import ToolBox, ToolError, Permission

IMPLEMENTER_SYSTEM_PROMPT = """You are the Coding Implementer inside an engineering agent. \
You are given ONE file to change, the specific change description, and the file's current \
content (if it exists). Output ONLY the complete new file content -- no markdown fences, no \
commentary, no explanations before or after. Preserve everything in the file not related to \
the requested change. Make the smallest change that correctly accomplishes the description."""


class CodingImplementer:
    def __init__(self, backend: ModelBackend, tools: ToolBox):
        if tools.permission not in (Permission.IMPLEMENT, Permission.COMMIT):
            raise PermissionError("CodingImplementer requires a ToolBox with IMPLEMENT permission")
        self.backend = backend
        self.tools = tools

    def implement(self, run_id: str, plan: EngineeringPlan) -> ImplementationResult:
        edits: List[FileEdit] = []

        for change in plan.changes:
            try:
                if change.change_type == "DELETE":
                    # Deletion is deliberately NOT auto-applied -- it's the
                    # highest-risk action a coding agent can take silently.
                    # Surface it for manual handling instead.
                    edits.append(FileEdit(
                        file=change.file, change_type="DELETE",
                        diff="(deletion skipped -- requires manual action, not auto-applied)",
                    ))
                    continue

                current = ""
                if change.change_type == "MODIFY":
                    read = self.tools.read_file(change.file)
                    if not read.ok:
                        return ImplementationResult(
                            run_id=run_id, edits=edits, success=False,
                            error=f"Cannot modify '{change.file}': {read.error}",
                        )
                    current = read.data["content"]

                new_content = self._generate_content(change.file, change.description, current)

                if change.change_type == "CREATE":
                    result = self.tools.create_file(change.file, new_content)
                else:
                    result = self.tools.edit_file(change.file, new_content)

                if not result.ok:
                    return ImplementationResult(
                        run_id=run_id, edits=edits, success=False,
                        error=f"Failed writing '{change.file}': {result.error}",
                    )

                edits.append(FileEdit(
                    file=change.file,
                    change_type=change.change_type,
                    diff=result.data.get("diff", ""),
                    bytes_before=result.data.get("bytes_before", 0),
                    bytes_after=result.data.get("bytes_after", 0),
                ))

            except (BackendError, ToolError) as exc:
                return ImplementationResult(
                    run_id=run_id, edits=edits, success=False,
                    error=f"Error implementing '{change.file}': {exc}",
                )

        return ImplementationResult(
            run_id=run_id, edits=edits, backend_used=self.backend.name, success=True,
        )

    def _generate_content(self, file: str, description: str, current_content: str) -> str:
        prompt = f"""FILE: {file}

CHANGE DESCRIPTION:
{description}

CURRENT CONTENT:
---
{current_content if current_content else '(new file, currently empty)'}
---

Output the complete new content of {file} now."""
        content = self.backend.complete(prompt, system=IMPLEMENTER_SYSTEM_PROMPT)
        # Strip accidental markdown fences.
        stripped = content.strip()
        if stripped.startswith("```"):
            lines = stripped.splitlines()
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            stripped = "\n".join(lines)
        return stripped
