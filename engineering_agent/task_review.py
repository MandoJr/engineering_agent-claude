from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Set

from .models import EngineeringPlan, FileEdit, PatchResult, ReviewResult, TestResult
from .task_graph import TaskGraph, TaskStatus


@dataclass
class TaskGraphReviewResult:
    passed: bool
    findings: List[str] = field(default_factory=list)
    evidence: List[str] = field(default_factory=list)
    task_statuses: Dict[str, str] = field(default_factory=dict)
    blocked_by: Dict[str, List[str]] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            "passed": self.passed,
            "findings": list(self.findings),
            "evidence": list(self.evidence),
            "task_statuses": dict(self.task_statuses),
            "blocked_by": {
                task_id: list(dependencies)
                for task_id, dependencies in self.blocked_by.items()
            },
        }


class IndependentTaskGraphReviewer:
    """
    Evidence-based review of the entire task execution graph.

    This reviewer does not trust the orchestrator's overall success flag.
    It independently inspects task states, dependency consistency,
    verification evidence, changed-file scope, patch results, and Git diff.
    """

    def review(
        self,
        graph: TaskGraph,
        plan: EngineeringPlan,
        edits: Sequence[FileEdit],
        tests: Sequence[TestResult],
        patch_results: Sequence[PatchResult],
        git_diff: str = "",
    ) -> TaskGraphReviewResult:
        findings: List[str] = []
        evidence: List[str] = []

        graph.validate()

        task_statuses = {
            task.task_id: task.status.value
            for task in sorted(
                graph.tasks.values(),
                key=lambda item: item.task_id,
            )
        }

        blocked_by = {
            task.task_id: graph.dependency_failures(task.task_id)
            for task in graph.blocked_tasks()
        }

        passed_tasks = [
            task for task in graph.tasks.values()
            if task.status == TaskStatus.PASSED
        ]
        failed_tasks = [
            task for task in graph.tasks.values()
            if task.status == TaskStatus.FAILED
        ]
        blocked_tasks = [
            task for task in graph.tasks.values()
            if task.status == TaskStatus.BLOCKED
        ]

        evidence.append(
            f"reviewed {len(graph.tasks)} task(s)"
        )
        evidence.append(
            f"{len(passed_tasks)}/{len(graph.tasks)} task(s) reached PASSED"
            if graph.tasks
            else "reviewed empty task graph"
        )

        for task in sorted(graph.tasks.values(), key=lambda item: item.task_id):
            if task.status != TaskStatus.PASSED:
                findings.append(
                    f"Task '{task.task_id}' ended in {task.status.value}"
                )

            verification_entries = [
                item for item in task.evidence
                if item.kind == "VERIFICATION"
            ]

            if task.status == TaskStatus.PASSED and not verification_entries:
                findings.append(
                    f"Task '{task.task_id}' is PASSED without recorded "
                    "task verification evidence"
                )

            for dependency in task.dependencies:
                dependency_status = graph.get(dependency).status

                if (
                    task.status == TaskStatus.PASSED
                    and dependency_status != TaskStatus.PASSED
                ):
                    findings.append(
                        f"Task '{task.task_id}' is PASSED while dependency "
                        f"'{dependency}' is {dependency_status.value}"
                    )

        if failed_tasks:
            findings.append(
                "Failed tasks remain in the execution graph: "
                + ", ".join(sorted(task.task_id for task in failed_tasks))
            )

        if blocked_tasks:
            findings.append(
                "Blocked tasks remain in the execution graph: "
                + ", ".join(sorted(task.task_id for task in blocked_tasks))
            )

        approved_files: Set[str] = set(plan.files)
        edited_files = {edit.file for edit in edits}

        unexpected_edits = sorted(edited_files - approved_files)
        if unexpected_edits:
            findings.append(
                "Actual edits include files outside approved scope: "
                + ", ".join(unexpected_edits)
            )

        evidence.append(
            f"reviewed {len(edited_files)} changed file(s)"
        )

        unapplied = [
            patch for patch in patch_results
            if not patch.applied
        ]

        if unapplied:
            findings.append(
                "One or more structured patches were not applied: "
                + ", ".join(
                    patch.change_id or "<unknown>"
                    for patch in unapplied
                )
            )

        evidence.append(
            f"{sum(1 for patch in patch_results if patch.applied)}/"
            f"{len(patch_results)} structured patch result(s) applied"
        )

        if not tests:
            findings.append(
                "No verification results were recorded for the run"
            )
        elif not all(test.passed for test in tests):
            findings.append(
                "At least one run-level verification command failed"
            )

        if git_diff:
            diff_files = {
                line[6:]
                for line in git_diff.splitlines()
                if line.startswith("+++ b/")
            }
            unexpected_diff = sorted(diff_files - approved_files)

            evidence.append(
                f"reviewed Git diff for {len(diff_files)} file(s)"
            )

            if unexpected_diff:
                findings.append(
                    "Git diff includes files outside approved scope: "
                    + ", ".join(unexpected_diff)
                )

        if graph.tasks and all(
            task.status == TaskStatus.PASSED
            for task in graph.tasks.values()
        ):
            evidence.append("all graph tasks reached PASSED")
        else:
            evidence.append("graph contains incomplete task state")

        return TaskGraphReviewResult(
            passed=not findings,
            findings=findings,
            evidence=evidence,
            task_statuses=task_statuses,
            blocked_by=blocked_by,
        )
