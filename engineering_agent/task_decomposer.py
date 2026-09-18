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

            # Changes to the same file are serialized in plan order so one
            # patch cannot race another patch against the same file.
            for prior_task in tasks[:index]:
                if set(task.affected_files) & set(prior_task.affected_files):
                    dependencies.add(prior_task.task_id)

            # Explicit prerequisites are resolved against the ENTIRE task set,
            # not just earlier tasks. Planner list order must never determine
            # whether a declared dependency exists.
            for prerequisite in change.prerequisites:
                matches = []

                for other_index, other_task in enumerate(tasks):
                    if other_index == index:
                        continue

                    other_change = plan.changes[other_index]

                    if TaskDecomposer._matches_prerequisite(
                        prerequisite,
                        other_task,
                        other_change,
                    ):
                        matches.append(other_task.task_id)

                if not matches:
                    raise TaskDecompositionError(
                        f"Task '{task.task_id}' declares unresolved "
                        f"prerequisite '{prerequisite}'"
                    )

                if len(matches) > 1:
                    raise TaskDecompositionError(
                        f"Task '{task.task_id}' declares ambiguous "
                        f"prerequisite '{prerequisite}'; matches {matches}"
                    )

                dependencies.add(matches[0])

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




