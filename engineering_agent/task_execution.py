from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

from .execution_state import TaskExecutionStateMachine
from .models import (
    EngineeringPlan,
    FileEdit,
    ImplementationResult,
    PatchResult,
    TestResult,
)
from .structured_implementer import StructuredCodingImplementer
from .task_graph import EngineeringTask, TaskGraph, TaskStatus
from .tester import TestEngineer


CheckpointFn = Callable[[str, TaskStatus], None]


@dataclass
class TaskExecutionOutcome:
    implementation: ImplementationResult
    tests: List[TestResult] = field(default_factory=list)
    task_tests: Dict[str, List[TestResult]] = field(default_factory=dict)
    task_plans: Dict[str, EngineeringPlan] = field(default_factory=dict)
    completed_tasks: List[str] = field(default_factory=list)
    failed_tasks: List[str] = field(default_factory=list)
    blocked_tasks: List[str] = field(default_factory=list)
    failures: List[str] = field(default_factory=list)


class TaskExecutionEngine:
    """
    Executes an EngineeringPlan one DAG task at a time.

    The engine deliberately does not own approval, Git commit, independent
    review, or learning. Those remain orchestrator-level responsibilities.
    """

    def __init__(
        self,
        implementer: StructuredCodingImplementer,
        tester: TestEngineer,
        checkpoint: Optional[CheckpointFn] = None,
    ):
        self.implementer = implementer
        self.tester = tester
        self.checkpoint = checkpoint

    def execute(
        self,
        run_id: str,
        plan: EngineeringPlan,
        graph: TaskGraph,
        state_machine: TaskExecutionStateMachine,
    ) -> TaskExecutionOutcome:
        graph.validate()
        state_machine.validate()

        edits: List[FileEdit] = []
        patch_results: List[PatchResult] = []
        tests: List[TestResult] = []
        task_tests: Dict[str, List[TestResult]] = {}
        task_plans: Dict[str, EngineeringPlan] = {}
        completed_tasks: List[str] = []
        failed_tasks: List[str] = []
        failures: List[str] = []
        backend_used = self.implementer.backend.name

        while True:
            ready = graph.ready_tasks()
            if not ready:
                break

            # ready_tasks() is deterministic, so execution order remains
            # deterministic without pretending list order is dependency.
            progress = False

            for task in ready:
                if task.status != TaskStatus.READY:
                    continue

                progress = True

                state_machine.transition(
                    task.task_id,
                    TaskStatus.RUNNING,
                    reason="Task prerequisites are satisfied; execution started.",
                )
                self._checkpoint(task.task_id, TaskStatus.RUNNING)

                task_plan = self._task_plan(plan, task)
                task_plans[task.task_id] = task_plan

                implementation = self.implementer.implement(
                    f"{run_id}:{task.task_id}",
                    task_plan,
                )

                edits.extend(implementation.edits)
                patch_results.extend(implementation.patch_results)

                if not implementation.success:
                    message = (
                        implementation.error
                        or f"Implementation failed for task {task.task_id}"
                    )
                    failures.append(f"{task.task_id}: {message}")
                    failed_tasks.append(task.task_id)

                    state_machine.transition(
                        task.task_id,
                        TaskStatus.FAILED,
                        reason="Structured implementation failed.",
                        evidence={"error": message},
                    )
                    self._checkpoint(task.task_id, TaskStatus.FAILED)
                    continue

                state_machine.transition(
                    task.task_id,
                    TaskStatus.VERIFYING,
                    reason="Implementation applied; task verification started.",
                )
                self._checkpoint(task.task_id, TaskStatus.VERIFYING)

                task_result = self.tester.run_tests(
                    task.verification_commands
                )
                task_tests[task.task_id] = task_result
                tests.extend(task_result)

                passed = bool(task_result) and all(
                    result.passed for result in task_result
                )

                task.add_evidence(
                    "VERIFICATION",
                    (
                        f"{sum(result.passed for result in task_result)}/"
                        f"{len(task_result)} verification command(s) passed"
                    ),
                    {
                        "commands": [
                            result.command for result in task_result
                        ],
                        "passed": all(
                            result.passed for result in task_result
                        ) if task_result else False,
                    },
                )

                if passed:
                    completed_tasks.append(task.task_id)

                    state_machine.transition(
                        task.task_id,
                        TaskStatus.PASSED,
                        reason="All task verification commands passed.",
                        evidence={
                            "verification_count": len(task_result),
                        },
                    )
                    self._checkpoint(task.task_id, TaskStatus.PASSED)
                else:
                    failed_tasks.append(task.task_id)
                    failures.append(
                        f"{task.task_id}: one or more verification commands failed"
                    )

                    state_machine.transition(
                        task.task_id,
                        TaskStatus.FAILED,
                        reason="Task verification failed.",
                        evidence={
                            "verification_count": len(task_result),
                            "passed_count": sum(
                                result.passed for result in task_result
                            ),
                        },
                    )
                    self._checkpoint(task.task_id, TaskStatus.FAILED)

            if not progress:
                break

        blocked_tasks = [
            task.task_id
            for task in graph.blocked_tasks()
        ]

        overall_success = (
            bool(graph.tasks)
            and all(
                task.status == TaskStatus.PASSED
                for task in graph.tasks.values()
            )
        )

        implementation = ImplementationResult(
            run_id=run_id,
            edits=edits,
            patch_results=patch_results,
            backend_used=backend_used,
            success=overall_success,
            error="; ".join(failures) if failures else None,
            rollback_performed=any(
                result.rollback_performed
                for result in patch_results
            ),
        )

        return TaskExecutionOutcome(
            implementation=implementation,
            tests=tests,
            task_tests=task_tests,
            task_plans=task_plans,
            completed_tasks=completed_tasks,
            failed_tasks=list(dict.fromkeys(failed_tasks)),
            blocked_tasks=blocked_tasks,
            failures=failures,
        )

    def _task_plan(
        self,
        plan: EngineeringPlan,
        task: EngineeringTask,
    ) -> EngineeringPlan:
        index = task.metadata.get("source_change_index")
        if not isinstance(index, int):
            raise ValueError(
                f"Task {task.task_id} is missing source_change_index metadata"
            )

        change = plan.changes[index]

        return EngineeringPlan(
            plan_id=f"{plan.plan_id}:{task.task_id}",
            goal_id=plan.goal_id,
            affected_systems=list(plan.affected_systems),
            files=list(task.affected_files),
            changes=[change],
            risks=list(plan.risks),
            tests=list(task.verification_commands),
            benchmarks=[],
            expected_result=(
                task.objective or plan.expected_result
            ),
            risk_level=task.risk_level,
            created_at=plan.created_at,
            raw_backend_output=plan.raw_backend_output,
            revision=plan.revision,
            assumptions=list(plan.assumptions),
            verification_requirements=list(
                task.verification_commands
            ),
        )

    def _checkpoint(
        self,
        task_id: str,
        status: TaskStatus,
    ) -> None:
        if self.checkpoint is not None:
            self.checkpoint(task_id, status)
