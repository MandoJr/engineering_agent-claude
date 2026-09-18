"""Bounded recovery that generates a new structured patch for fresh evidence."""
from __future__ import annotations

import json
from typing import List

from .backend import ModelBackend
from .config import MAX_RECOVERY_ATTEMPTS
from .models import ChangeSet, EngineeringPlan, FileEdit, RecoveryAttempt, TestResult
from .patching import PatchApplier, PatchValidationError, parse_change_set
from .tester import TestEngineer
from .tools import ToolBox

RECOVERY_SYSTEM_PROMPT = """You are a failure-recovery engineer. Return ONLY JSON:
{"root_cause":"short evidence-based cause", "changes":[structured patch operations]}.
Generate a minimal patch using exact current context. Never output a whole replacement file unless creating a new file.
Return {"root_cause":"...", "changes":[]} when no confident safe patch exists."""

class StructuredFailureRecoveryEngineer:
    def __init__(self, backend: ModelBackend, tools: ToolBox, tester: TestEngineer, max_attempts: int = MAX_RECOVERY_ATTEMPTS):
        self.backend, self.tools, self.tester, self.max_attempts = backend, tools, tester, max_attempts
        self.applier = PatchApplier(tools)

    def recover(self, plan: EngineeringPlan, failing_file: str, failed_test: TestResult) -> List[RecoveryAttempt]:
        attempts, seen = [], set()
        if failing_file not in set(plan.files):
            return [RecoveryAttempt(attempt_number=1, root_cause="Recovery target is outside approved scope.",
                recovery_plan="Stopped; revised plan and approval required.", succeeded=False)]
        for number in range(1, self.max_attempts + 1):
            read = self.tools.read_file(failing_file)
            if not read.ok:
                attempts.append(RecoveryAttempt(attempt_number=number, root_cause="Recovery target missing", recovery_plan="Stopped", succeeded=False)); break
            raw = self.backend.complete(self._prompt(failing_file, read.data["content"], failed_test, attempts), system=RECOVERY_SYSTEM_PROMPT, json_mode=True)
            try:
                data = json.loads(raw)
                root = str(data.get("root_cause", "Unclassified failure"))
                if not data.get("changes"):
                    attempts.append(RecoveryAttempt(attempt_number=number, root_cause=root, recovery_plan="No confident patch generated.", succeeded=False)); break
                patch = parse_change_set(raw, task_id=plan.plan_id, default_file=failing_file)
                if any(change.file != failing_file for change in patch.changes):
                    raise PatchValidationError("SCOPE_VIOLATION", "Recovery patch targets another file")
            except (ValueError, PatchValidationError) as exc:
                attempts.append(RecoveryAttempt(attempt_number=number, root_cause="Invalid recovery patch", recovery_plan=str(exc), succeeded=False, evidence=[getattr(exc, "classification", "MALFORMED_PATCH")])); break
            # Fingerprint model intent, not generated change IDs, so the same
            # failed patch cannot evade duplicate detection on a later retry.
            fingerprint = json.dumps(data, sort_keys=True)
            if fingerprint in seen:
                attempts.append(RecoveryAttempt(attempt_number=number, root_cause=root, recovery_plan="Duplicate recovery patch prevented.", succeeded=False, evidence=["DUPLICATE_PATCH"])); break
            seen.add(fingerprint)
            results, applied = self.applier.apply(patch, plan.files)
            if not applied:
                attempts.append(RecoveryAttempt(attempt_number=number, root_cause=root, recovery_plan="Patch rejected or failed: " + "; ".join(r.message for r in results), succeeded=False, evidence=[r.classification for r in results])); break
            retest = self.tester.run_tests([failed_test.command])[0]
            edits = [FileEdit(file=r.file, change_type=r.operation, diff=r.diff) for r in results if r.applied]
            attempts.append(
                RecoveryAttempt(
                    attempt_number=number,
                    root_cause=root,
                    recovery_plan="Applied validated recovery patch and re-ran failing test.",
                    edits=edits,
                    patch_results=results,
                    test_result=retest,
                    succeeded=retest.passed,
                    evidence=[retest.classification],
                )
            )
            if retest.passed: break
            failed_test = retest
        return attempts

    @staticmethod
    def _prompt(file: str, content: str, failed: TestResult, attempts: List[RecoveryAttempt]) -> str:
        return f"""FAILING FILE: {file}
CURRENT CONTENT:\n---\n{content}\n---
FAILED TEST: {failed.command}\nSTDERR: {failed.stderr_tail[-1500:]}
PRIOR ATTEMPTS: {[a.recovery_plan for a in attempts]}
Produce the recovery patch JSON."""
