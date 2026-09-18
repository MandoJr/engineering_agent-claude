from __future__ import annotations

import heapq
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, Iterable, List, Optional


class TaskGraphError(Exception):
    """Base error for task-graph operations."""


class DuplicateTaskError(TaskGraphError):
    pass


class UnknownTaskError(TaskGraphError):
    pass


class DependencyCycleError(TaskGraphError):
    def __init__(self, cycle: List[str]):
        self.cycle = cycle
        super().__init__(f"Task dependency cycle detected: {' -> '.join(cycle)}")


class TaskStatus(str, Enum):
    PENDING = "PENDING"
    BLOCKED = "BLOCKED"
    READY = "READY"
    RUNNING = "RUNNING"
    VERIFYING = "VERIFYING"
    PASSED = "PASSED"
    FAILED = "FAILED"
    RECOVERING = "RECOVERING"
    INTERRUPTED = "INTERRUPTED"
    SKIPPED = "SKIPPED"
    CANCELLED = "CANCELLED"


@dataclass
class TaskEvidence:
    kind: str
    message: str
    data: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


@dataclass
class EngineeringTask:
    task_id: str
    title: str
    objective: str
    affected_files: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    required_capabilities: List[str] = field(default_factory=list)
    verification_commands: List[str] = field(default_factory=list)
    completion_criteria: List[str] = field(default_factory=list)
    risk_level: str = "LOW"
    priority: int = 100
    status: TaskStatus = TaskStatus.PENDING
    retry_count: int = 0
    evidence: List[TaskEvidence] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def add_evidence(
        self,
        kind: str,
        message: str,
        data: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.evidence.append(
            TaskEvidence(
                kind=kind,
                message=message,
                data=dict(data or {}),
            )
        )


class TaskGraph:
    """
    Dependency-aware DAG for an engineering run.

    The graph is deliberately independent from the orchestrator so that
    JARVIS can later expose it to a dashboard, specialist system, or
    resumable execution engine.
    """

    _TERMINAL_SUCCESS = {TaskStatus.PASSED}
    _BLOCKING_FAILURES = {TaskStatus.FAILED, TaskStatus.CANCELLED}

    def __init__(self, tasks: Optional[Iterable[EngineeringTask]] = None):
        self._tasks: Dict[str, EngineeringTask] = {}

        for task in tasks or []:
            self.add_task(task)

        self.validate()

    @property
    def tasks(self) -> Dict[str, EngineeringTask]:
        return dict(self._tasks)

    def add_task(self, task: EngineeringTask) -> None:
        if task.task_id in self._tasks:
            raise DuplicateTaskError(f"Task already exists: {task.task_id}")

        self._tasks[task.task_id] = task

    def get(self, task_id: str) -> EngineeringTask:
        try:
            return self._tasks[task_id]
        except KeyError as exc:
            raise UnknownTaskError(f"Unknown task: {task_id}") from exc

    def add_dependency(self, task_id: str, depends_on: str) -> None:
        task = self.get(task_id)
        self.get(depends_on)

        if task_id == depends_on:
            raise DependencyCycleError([task_id, task_id])

        if depends_on not in task.dependencies:
            task.dependencies.append(depends_on)

        self.validate()

    def dependencies_of(self, task_id: str) -> List[str]:
        return sorted(self.get(task_id).dependencies)

    def dependents_of(self, task_id: str) -> List[str]:
        self.get(task_id)

        return sorted(
            task.task_id
            for task in self._tasks.values()
            if task_id in task.dependencies
        )

    def validate(self) -> None:
        unknown: List[str] = []

        for task in self._tasks.values():
            for dependency in task.dependencies:
                if dependency not in self._tasks:
                    unknown.append(f"{task.task_id}->{dependency}")

        if unknown:
            raise UnknownTaskError(
                "Unknown task dependencies: " + ", ".join(sorted(unknown))
            )

        cycle = self._find_cycle()
        if cycle:
            raise DependencyCycleError(cycle)

    def _find_cycle(self) -> Optional[List[str]]:
        visiting = set()
        visited = set()
        stack: List[str] = []

        def visit(task_id: str) -> Optional[List[str]]:
            if task_id in visiting:
                try:
                    index = stack.index(task_id)
                except ValueError:
                    return [task_id, task_id]
                return stack[index:] + [task_id]

            if task_id in visited:
                return None

            visiting.add(task_id)
            stack.append(task_id)

            for dependency in self._tasks[task_id].dependencies:
                cycle = visit(dependency)
                if cycle:
                    return cycle

            stack.pop()
            visiting.remove(task_id)
            visited.add(task_id)
            return None

        for task_id in sorted(self._tasks):
            cycle = visit(task_id)
            if cycle:
                return cycle

        return None

    def topological_order(self) -> List[str]:
        self.validate()

        indegree = {
            task_id: len(task.dependencies)
            for task_id, task in self._tasks.items()
        }

        dependents = {
            task_id: self.dependents_of(task_id)
            for task_id in self._tasks
        }

        heap: List[tuple[int, str]] = [
            (self._tasks[task_id].priority, task_id)
            for task_id, degree in indegree.items()
            if degree == 0
        ]
        heapq.heapify(heap)

        result: List[str] = []

        while heap:
            _, task_id = heapq.heappop(heap)
            result.append(task_id)

            for dependent in dependents[task_id]:
                indegree[dependent] -= 1
                if indegree[dependent] == 0:
                    heapq.heappush(
                        heap,
                        (self._tasks[dependent].priority, dependent),
                    )

        if len(result) != len(self._tasks):
            cycle = self._find_cycle() or []
            raise DependencyCycleError(cycle)

        return result

    def refresh_states(self) -> None:
        """
        Recalculate dependency-derived states for tasks that have not
        started execution yet.

        Running/verification/terminal states are preserved.
        """
        self.validate()

        for task in self._tasks.values():
            if task.status not in {
                TaskStatus.PENDING,
                TaskStatus.BLOCKED,
            }:
                continue

            if not task.dependencies:
                task.status = TaskStatus.READY
                continue

            dependency_statuses = [
                self.get(dep).status for dep in task.dependencies
            ]

            if any(
                status in self._BLOCKING_FAILURES
                for status in dependency_statuses
            ):
                task.status = TaskStatus.BLOCKED
            elif all(
                status in self._TERMINAL_SUCCESS
                for status in dependency_statuses
            ):
                task.status = TaskStatus.READY
            else:
                task.status = TaskStatus.BLOCKED

    def ready_tasks(self) -> List[EngineeringTask]:
        self.refresh_states()

        return sorted(
            (
                task
                for task in self._tasks.values()
                if task.status == TaskStatus.READY
            ),
            key=lambda task: (task.priority, task.task_id),
        )

    def blocked_tasks(self) -> List[EngineeringTask]:
        self.refresh_states()

        return sorted(
            (
                task
                for task in self._tasks.values()
                if task.status == TaskStatus.BLOCKED
            ),
            key=lambda task: task.task_id,
        )

    def mark_status(self, task_id: str, status: TaskStatus) -> None:
        task = self.get(task_id)

        if not isinstance(status, TaskStatus):
            status = TaskStatus(status)

        task.status = status
        self.refresh_states()

    def record_evidence(
        self,
        task_id: str,
        kind: str,
        message: str,
        data: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.get(task_id).add_evidence(kind, message, data)

    def dependency_failures(self, task_id: str) -> List[str]:
        task = self.get(task_id)

        return sorted(
            dependency
            for dependency in task.dependencies
            if self.get(dependency).status in self._BLOCKING_FAILURES
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tasks": {
                task_id: {
                    **asdict(task),
                    "status": task.status.value,
                    "evidence": [
                        asdict(evidence) for evidence in task.evidence
                    ],
                }
                for task_id, task in self._tasks.items()
            }
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "TaskGraph":
        raw_tasks = payload.get("tasks", {})
        tasks: List[EngineeringTask] = []

        for task_id, raw in raw_tasks.items():
            evidence = [
                TaskEvidence(**item)
                for item in raw.pop("evidence", [])
            ]

            task = EngineeringTask(
                task_id=task_id,
                title=raw["title"],
                objective=raw["objective"],
                affected_files=list(raw.get("affected_files", [])),
                dependencies=list(raw.get("dependencies", [])),
                required_capabilities=list(
                    raw.get("required_capabilities", [])
                ),
                verification_commands=list(
                    raw.get("verification_commands", [])
                ),
                completion_criteria=list(
                    raw.get("completion_criteria", [])
                ),
                risk_level=raw.get("risk_level", "LOW"),
                priority=int(raw.get("priority", 100)),
                status=TaskStatus(raw.get("status", TaskStatus.PENDING.value)),
                retry_count=int(raw.get("retry_count", 0)),
                evidence=evidence,
                metadata=dict(raw.get("metadata", {})),
            )
            tasks.append(task)

        graph = cls(tasks)
        graph.refresh_states()
        return graph

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)

    @classmethod
    def from_json(cls, payload: str) -> "TaskGraph":
        return cls.from_dict(json.loads(payload))
