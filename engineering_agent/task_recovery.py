from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

from .execution_state import TaskExecutionStateMachine
from .models import EngineeringPlan, ImplementationResult, RecoveryAttempt, TestResult
from .structured_recovery import StructuredFailureRecoveryEngineer
from .task_execution import TaskExecutionEngine, TaskExecutionOutcome
from .task_graph import TaskGraph, TaskStatus
from .task_verification import TaskVerificationEngine
from .tester import TestEngineer
from .tools import ToolBox


CheckpointFn = Callable[[str, TaskStatus], None]


@dataclass
class TaskRecoveryOutcome:
    attempts: List[RecoveryAttempt] = field(default_factory=list)
    recovered_tasks: List[str] = field(default_factory=list)
    failed_tasks: List[str] = field(default_factory=list)
    blocked_tasks: List[str] = field(default_factory=list)
    tests: List[TestResult] = field(default_factory=list)
    task_tests: Dict[str, List[TestResult]] = field(default_factory=dict)
    task_plans: Dict[str, EngineeringPlan] = field(default_factory=dict)
    resumed_outcomes: List[TaskExecutionOutcome] = field(default_factory=list)
    failures: List[str] = field(default_factory=list)


class TaskRecoveryCoordinator:
    """
    Coordinates bounded recovery across an engineering task DAG.

    Recovery is task-local. A recovered task can unlock dependent work, while
    an unrecovered task continues to block its dependents. Approval is never
    altered by this class.
    """

    def __init__(
        self,
        backend,
        tools: ToolBox,
        tester: TestEngineer,
        execution_engine: TaskExecutionEngine,
        max_attempts: int,
        checkpoint: Optional[CheckpointFn] = None,
    ):
        self.backend = backend
        self.tools = tools
        self.tester = tester
        self.execution_engine = execution_engine
        self.max_attempts = max_attempts
        self.checkpoint = checkpoint
        self.verifier = TaskVerificationEngine(tester)

    def recover(
        self,
        run_id: str,
        plan: EngineeringPlan,
        graph: TaskGraph,
        state_machine: TaskExecutionStateMachine,
        initial_outcome: TaskExecutionOutcome,
    ) -> TaskRecoveryOutcome:
        outcome = TaskRecoveryOutcome(
            tests=list(initial_outcome.tests),
            task_tests={
                task_id: list(results)
                for task_id, results in initial_outcome.task_tests.items()
            },
            task_plans=dict(initial_outcome.task_plans),
        )

        attempted: set[str] = set()
        pending = self._ordered_failed_tasks(graph)

        while pending:
            task_id = pending.pop(0)

            if task_id in attempted:
                continue

            task = graph.get(task_id)

            if task.status != TaskStatus.FAILED:
                continue

            attempted.add(task_id)

            task_results = outcome.task_tests.get(task_id, [])
            failing_test = next(
                (result for result in task_results if not result.passed),
                None,
            )

            if failing_test is None:
                message = (
                    f"{task_id}: recovery skipped because no failed "
                    "verification result is available."
                )
                outcome.failures.append(message)
                outcome.failed_tasks.append(task_id)
                task.add_evidence(
                    "RECOVERY_SKIPPED",
                    message,
                )
                continue

            task_plan = outcome.task_plans.get(task_id)
            if task_plan is None:
                message = (
                    f"{task_id}: recovery skipped because no task plan "
                    "is available."
                )
                outcome.failures.append(message)
                outcome.failed_tasks.append(task_id)
                task.add_evidence(
                    "RECOVERY_SKIPPED",
                    message,
                )
                continue

            if not task.affected_files:
                message = f"{task_id}: recovery skipped because no affected file is defined."
                outcome.failures.append(message)
                outcome.failed_tasks.append(task_id)
                continue

            failing_file = task.affected_files[0]

            state_machine.transition(
                task_id,
                TaskStatus.RECOVERING,
                reason="Beginning bounded recovery for failed task.",
                evidence={
                    "failed_test": failing_test.command,
                },
            )
            self._checkpoint(task_id, TaskStatus.RECOVERING)

            recovery_engineer = StructuredFailureRecoveryEngineer(
                self.backend,
                self.tools,
                self.tester,
                max_attempts=self.max_attempts,
            )

            attempts = recovery_engineer.recover(
                task_plan,
                failing_file,
                failing_test,
            )
            outcome.attempts.extend(attempts)

            task.add_evidence(
                "RECOVERY",
                f"Performed {len(attempts)} bounded recovery attempt(s).",
                {
                    "attempt_count": len(attempts),
                    "succeeded": bool(
                        attempts and attempts[-1].succeeded
                    ),
                    "failed_test": failing_test.command,
                },
            )

            if not attempts or not attempts[-1].succeeded:
                state_machine.transition(
                    task_id,
                    TaskStatus.FAILED,
                    reason="Recovery attempts did not resolve the task.",
                    evidence={
                        "attempt_count": len(attempts),
                    },
                )
                self._checkpoint(task_id, TaskStatus.FAILED)

                outcome.failed_tasks.append(task_id)
                outcome.failures.append(
                    f"{task_id}: recovery exhausted or produced no successful attempt."
                )
                continue

            state_machine.transition(
                task_id,
                TaskStatus.VERIFYING,
                reason="Recovery patch applied; running complete task verification.",
            )
            self._checkpoint(task_id, TaskStatus.VERIFYING)

            verification = self.verifier.verify(task)
            outcome.task_tests[task_id] = verification.results
            outcome.tests.extend(verification.results)

            if verification.passed:
                state_machine.transition(
                    task_id,
                    TaskStatus.PASSED,
                    reason="Recovery verification passed.",
                    evidence=verification.evidence,
                )
                self._checkpoint(task_id, TaskStatus.PASSED)

                outcome.recovered_tasks.append(task_id)

                # A recovered task may unlock one or many dependents.
                resumed = self.execution_engine.resume(
                    run_id,
                    plan,
                    graph,
                    state_machine,
                )
                outcome.resumed_outcomes.append(resumed)

                outcome.tests.extend(resumed.tests)
                outcome.task_tests.update(resumed.task_tests)
                outcome.task_plans.update(resumed.task_plans)

                task.add_evidence(
                    "RECOVERY_RESUME",
                    "Recovery succeeded and DAG execution resumed.",
                    {
                        "completed_tasks": resumed.completed_tasks,
                        "failed_tasks": resumed.failed_tasks,
                        "blocked_tasks": resumed.blocked_tasks,
                    },
                )

                for resumed_failed in resumed.failed_tasks:
                    if resumed_failed not in attempted:
                        pending.append(resumed_failed)

                pending = self._sort_pending(graph, pending)

            else:
                state_machine.transition(
                    task_id,
                    TaskStatus.FAILED,
                    reason="Recovery patch applied but task verification still fails.",
                    evidence=verification.evidence,
                )
                self._checkpoint(task_id, TaskStatus.FAILED)

                outcome.failed_tasks.append(task_id)
                outcome.failures.append(
                    f"{task_id}: recovery patch did not satisfy task verification."
                )

        outcome.blocked_tasks = [
            task.task_id
            for task in graph.blocked_tasks()
        ]

        outcome.failed_tasks = list(dict.fromkeys(outcome.failed_tasks))
        outcome.recovered_tasks = list(
            dict.fromkeys(outcome.recovered_tasks)
        )

        return outcome

    @staticmethod
    def _ordered_failed_tasks(graph: TaskGraph) -> List[str]:
        return sorted(
            (
                task.task_id
                for task in graph.tasks.values()
                if task.status == TaskStatus.FAILED
            ),
            key=lambda task_id: task_id,
        )

    @staticmethod
    def _sort_pending(
        graph: TaskGraph,
        task_ids: List[str],
    ) -> List[str]:
        unique = list(dict.fromkeys(task_ids))
        order = graph.topological_order()
        position = {
            task_id: index
            for index, task_id in enumerate(order)
        }
        return sorted(
            unique,
            key=lambda task_id: position.get(task_id, 10**9),
        )

    def _checkpoint(
        self,
        task_id: str,
        status: TaskStatus,
    ) -> None:
        if self.checkpoint is not None:
            self.checkpoint(task_id, status)
