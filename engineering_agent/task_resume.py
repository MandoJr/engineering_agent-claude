from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

from .execution_state import TaskExecutionStateMachine
from .task_graph import EngineeringTask, TaskGraph, TaskStatus
from .task_verification import TaskVerificationEngine


class ResumeError(RuntimeError):
    """Raised when a run cannot be safely resumed."""


@dataclass
class ResumeReconciliation:
    reconciled_passed: List[str] = field(default_factory=list)
    reconciled_failed: List[str] = field(default_factory=list)
    still_blocked: List[str] = field(default_factory=list)
    reasons: Dict[str, str] = field(default_factory=dict)


class TaskResumeEngine:
    """
    Reconciles interrupted tasks before normal DAG execution resumes.

    The engine never assumes that an interrupted implementation did or did
    not reach disk. It uses task verification as the first source of truth.
    """

    def __init__(self, verifier: TaskVerificationEngine):
        self.verifier = verifier

    def reconcile(
        self,
        graph: TaskGraph,
        state_machine: TaskExecutionStateMachine,
    ) -> ResumeReconciliation:
        graph.validate()
        state_machine.validate()

        result = ResumeReconciliation()

        interrupted = sorted(
            (
                task
                for task in graph.tasks.values()
                if task.status == TaskStatus.INTERRUPTED
            ),
            key=lambda task: task.task_id,
        )

        for task in interrupted:
            verification = self.verifier.verify(task)

            if verification.passed:
                state_machine.transition(
                    task.task_id,
                    TaskStatus.PASSED,
                    reason=(
                        "Interrupted task reconciliation succeeded; "
                        "existing implementation satisfies verification."
                    ),
                    evidence=verification.evidence,
                )
                result.reconciled_passed.append(task.task_id)
                result.reasons[task.task_id] = (
                    "Existing implementation passed task verification."
                )
            else:
                state_machine.transition(
                    task.task_id,
                    TaskStatus.FAILED,
                    reason=(
                        "Interrupted task could not be reconciled as "
                        "complete; recovery is required."
                    ),
                    evidence=verification.evidence,
                )
                result.reconciled_failed.append(task.task_id)
                result.reasons[task.task_id] = (
                    "Interrupted task failed reconciliation verification."
                )

        result.still_blocked = [
            task.task_id
            for task in graph.blocked_tasks()
        ]

        return result

    @staticmethod
    def mark_active_tasks_interrupted(
        graph: TaskGraph,
        state_machine: TaskExecutionStateMachine,
    ) -> List[str]:
        """
        Convert in-flight tasks to INTERRUPTED before process shutdown or
        explicit run suspension.

        Terminal tasks are never modified.
        """
        interrupted: List[str] = []

        for task in graph.tasks.values():
            if task.status in {
                TaskStatus.RUNNING,
                TaskStatus.VERIFYING,
                TaskStatus.RECOVERING,
            }:
                state_machine.transition(
                    task.task_id,
                    TaskStatus.INTERRUPTED,
                    reason="Engineering run was interrupted while task was active.",
                )
                interrupted.append(task.task_id)

        return interrupted
