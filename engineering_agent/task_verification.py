from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

from .models import TestResult
from .task_graph import EngineeringTask
from .tester import TestEngineer


@dataclass
class TaskVerificationOutcome:
    task_id: str
    passed: bool
    results: List[TestResult] = field(default_factory=list)
    evidence: Dict[str, Any] = field(default_factory=dict)


class TaskVerificationEngine:
    """
    Runs verification specifically attached to one engineering task.

    Verification is intentionally separate from implementation so the
    execution engine can treat test evidence as an independent observation.
    """

    def __init__(self, tester: TestEngineer):
        self.tester = tester

    def verify(self, task: EngineeringTask) -> TaskVerificationOutcome:
        commands = list(dict.fromkeys(
            command.strip()
            for command in task.verification_commands
            if command and command.strip()
        ))

        if not commands:
            return TaskVerificationOutcome(
                task_id=task.task_id,
                passed=False,
                results=[],
                evidence={
                    "verification_defined": False,
                    "reason": "No task-specific verification commands were defined.",
                    "commands": [],
                    "passed_count": 0,
                    "failed_count": 0,
                },
            )

        results = self.tester.run_tests(commands)

        passed_count = sum(result.passed for result in results)
        failed_results = [result for result in results if not result.passed]

        return TaskVerificationOutcome(
            task_id=task.task_id,
            passed=bool(results) and all(result.passed for result in results),
            results=results,
            evidence={
                "verification_defined": True,
                "commands": commands,
                "passed_count": passed_count,
                "failed_count": len(failed_results),
                "failed_commands": [
                    result.command for result in failed_results
                ],
                "failure_classifications": [
                    result.classification for result in failed_results
                    if result.classification
                ],
            },
        )
