from __future__ import annotations

from typing import List

from .models import EngineeringPlan, PlannedChange
from .task_graph import EngineeringTask, TaskGraph


class TaskDecompositionError(ValueError):
    """Raised when a plan cannot be represented as executable tasks."""


class TaskDecomposer:
    """
    Converts an approved EngineeringPlan into a deterministic task DAG.

    Dependencies come from explicit prerequisites and from genuine file
    overlap between planned changes. Missing dependency information does not
    get invented merely from list position.
    """

    @staticmethod
    def from_plan(plan: EngineeringPlan) -> TaskGraph:
        if not plan.changes:
            return TaskGraph()

        tasks: List[EngineeringTask] = []

        for index, change in enumerate(plan.changes):
            task_id = f"{plan.plan_id}:task-{index + 1:03d}"

            affected_files = [change.file]
            target_file = getattr(change, 'target_file', None)
            if target_file and target_file not in affected_files:
                affected_files.append(target_file)

            verification = (
                list(change.verification)
                or list(plan.verification_requirements)
                or list(plan.tests)
            )

            title = (
                change.description.strip()
                if change.description.strip()
                else f"{change.change_type.title()} {change.file}"
            )

            task = EngineeringTask(
                task_id=task_id,
                title=title,
                objective=change.objective or change.description,
                affected_files=affected_files,
                required_capabilities=["code_generation"],
                verification_commands=verification,
                completion_criteria=(
                    list(change.completion_criteria)
                    or [plan.expected_result]
                    if plan.expected_result
                    else []
                ),
                risk_level=change.risk or plan.risk_level,
                priority=index * 10 + 100,
                metadata={
                    "source_change_index": index,
                    "source_file": change.file,
                    "change_type": change.change_type,
                },
            )
            tasks.append(task)

        for index, task in enumerate(tasks):
            change = plan.changes[index]
            dependencies = set()

            for prior_index, prior_task in enumerate(tasks[:index]):
                prior_change = plan.changes[prior_index]

                # Two changes to the same file must be serialized.
                if set(task.affected_files) & set(prior_task.affected_files):
                    dependencies.add(prior_task.task_id)

                # Explicit prerequisites may reference a task id, file,
                # target file, or change description.
                for prerequisite in change.prerequisites:
                    if TaskDecomposer._matches_prerequisite(
                        prerequisite,
                        prior_task,
                        prior_change,
                    ):
                        dependencies.add(prior_task.task_id)

            task.dependencies = sorted(dependencies)

        graph = TaskGraph(tasks)
        graph.refresh_states()
        return graph

    @staticmethod
    def _matches_prerequisite(
        prerequisite: str,
        prior_task: EngineeringTask,
        prior_change: PlannedChange,
    ) -> bool:
        needle = prerequisite.strip().lower()
        if not needle:
            return False

        candidates = {
            prior_task.task_id.lower(),
            prior_task.title.lower(),
            prior_change.description.strip().lower(),
            prior_change.file.lower(),
        }

        target_file = getattr(prior_change, 'target_file', None)
        if target_file:
            candidates.add(target_file.lower())

        return needle in candidates or any(
            needle in candidate or candidate in needle
            for candidate in candidates
            if candidate
        )




