"""
Failure Recovery Engineer.

Implements the bounded recovery loop:

    TEST FAILURE -> FAILURE ANALYSIS -> ROOT CAUSE -> RECOVERY PLAN ->
    (approval if the recovery plan touches new files) -> RETRY -> TEST

`max_attempts` is a hard ceiling -- there is no path to infinite retries.
If every attempt is exhausted, this returns a failed report rather than
continuing, and does not attempt anything outside the originally
approved file scope without flagging it back to the orchestrator.
"""

from __future__ import annotations

from typing import List, Optional

from .backend import ModelBackend
from .config import MAX_RECOVERY_ATTEMPTS
from .models import EngineeringPlan, FileEdit, RecoveryAttempt, TestResult
from .tester import TestEngineer
from .tools import ToolBox

RECOVERY_SYSTEM_PROMPT = """You are the Failure Recovery Engineer inside an engineering agent. \
You are given a failed test's output and the file that was most recently changed. Identify the \
most likely root cause in one or two sentences, then output ONLY the complete corrected content \
of the file -- no markdown fences, no commentary. If you cannot confidently identify a fix, \
output exactly NO_CONFIDENT_FIX instead of guessing."""


class RecoveryExhausted(RuntimeError):
    def __init__(self, attempts: List[RecoveryAttempt]):
        self.attempts = attempts
        super().__init__(f"Recovery exhausted after {len(attempts)} attempts")


class FailureRecoveryEngineer:
    def __init__(self, backend: ModelBackend, tools: ToolBox, tester: TestEngineer,
                 max_attempts: int = MAX_RECOVERY_ATTEMPTS):
        self.backend = backend
        self.tools = tools
        self.tester = tester
        self.max_attempts = max_attempts

    def recover(self, plan: EngineeringPlan, failing_file: str,
                failed_test: TestResult) -> List[RecoveryAttempt]:
        attempts: List[RecoveryAttempt] = []

        for attempt_number in range(1, self.max_attempts + 1):
            root_cause, new_content = self._analyze_and_fix(
                failing_file, failed_test, attempts
            )

            if new_content is None:
                attempts.append(RecoveryAttempt(
                    attempt_number=attempt_number,
                    root_cause=root_cause,
                    recovery_plan="No confident fix available; stopping recovery.",
                    succeeded=False,
                ))
                break

            write_result = self.tools.edit_file(failing_file, new_content)
            if not write_result.ok:
                attempts.append(RecoveryAttempt(
                    attempt_number=attempt_number,
                    root_cause=root_cause,
                    recovery_plan=f"Fix generated but failed to write: {write_result.error}",
                    succeeded=False,
                ))
                break

            edit = FileEdit(
                file=failing_file, change_type="MODIFY",
                diff=write_result.data.get("diff", ""),
            )

            retest = self.tester.run_tests([failed_test.command])
            new_result = retest[0] if retest else None
            succeeded = bool(new_result and new_result.passed)

            attempts.append(RecoveryAttempt(
                attempt_number=attempt_number,
                root_cause=root_cause,
                recovery_plan=f"Rewrote {failing_file} and re-ran: {failed_test.command}",
                edits=[edit],
                test_result=new_result,
                succeeded=succeeded,
            ))

            if succeeded:
                break
            failed_test = new_result or failed_test

        return attempts

    def _analyze_and_fix(self, failing_file: str, failed_test: TestResult,
                          prior_attempts: List[RecoveryAttempt]):
        read = self.tools.read_file(failing_file)
        current_content = read.data["content"] if read.ok else ""

        prior_summary = "\n".join(
            f"Attempt {a.attempt_number}: {a.root_cause} -> "
            f"{'fixed' if a.succeeded else 'did not fix it'}"
            for a in prior_attempts
        ) or "(none yet)"

        prompt = f"""FAILING FILE: {failing_file}

CURRENT CONTENT:
---
{current_content}
---

FAILED TEST COMMAND: {failed_test.command}
STDOUT (tail): {failed_test.stdout_tail[-1500:]}
STDERR (tail): {failed_test.stderr_tail[-1500:]}

PRIOR RECOVERY ATTEMPTS THIS RUN:
{prior_summary}

Identify the root cause and output the corrected file content, or NO_CONFIDENT_FIX."""

        raw = self.backend.complete(prompt, system=RECOVERY_SYSTEM_PROMPT)
        stripped = raw.strip()

        if "NO_CONFIDENT_FIX" in stripped[:50]:
            return "Backend reported no confident fix.", None

        # Root cause isn't separately requested from the model to keep this
        # to one call; summarize from the stderr tail as a practical proxy.
        root_cause = f"Failure in {failing_file}: {failed_test.stderr_tail.strip().splitlines()[-1:] or 'unknown'}"

        if stripped.startswith("```"):
            lines = stripped.splitlines()
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            stripped = "\n".join(lines)

        return root_cause, stripped
