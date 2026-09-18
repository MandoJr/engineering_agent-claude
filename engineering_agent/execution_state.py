from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from .task_graph import TaskGraph, TaskStatus, UnknownTaskError


class ExecutionStateError(Exception):
    """Base error for execution-state failures."""


class InvalidTaskTransitionError(ExecutionStateError):
    """Raised when a task attempts an invalid state transition."""

    def __init__(
        self,
        task_id: str,
        current: TaskStatus,
        requested: TaskStatus,
    ):
        self.task_id = task_id
        self.current = current
        self.requested = requested
        super().__init__(
            f"Invalid transition for task {task_id}: "
            f"{current.value} -> {requested.value}"
        )


@dataclass
class StateTransition:
    task_id: str
    previous_status: TaskStatus
    new_status: TaskStatus
    reason: str = ""
    evidence: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["previous_status"] = self.previous_status.value
        payload["new_status"] = self.new_status.value
        return payload

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "StateTransition":
        return cls(
            task_id=payload["task_id"],
            previous_status=TaskStatus(payload["previous_status"]),
            new_status=TaskStatus(payload["new_status"]),
            reason=payload.get("reason", ""),
            evidence=dict(payload.get("evidence", {})),
            timestamp=payload.get(
                "timestamp",
                datetime.now(timezone.utc).isoformat(),
            ),
        )


class TaskExecutionStateMachine:
    """
    Authoritative state-transition layer for engineering-task execution.

    TaskGraph remains responsible for dependency relationships and derived
    READY/BLOCKED state. This class is responsible for explicit execution
    transitions and their audit history.
    """

    ALLOWED_TRANSITIONS = {
        TaskStatus.PENDING: {
            TaskStatus.READY,
            TaskStatus.BLOCKED,
            TaskStatus.SKIPPED,
            TaskStatus.CANCELLED,
        },
        TaskStatus.BLOCKED: {
            TaskStatus.READY,
            TaskStatus.SKIPPED,
            TaskStatus.CANCELLED,
        },
        TaskStatus.READY: {
            TaskStatus.RUNNING,
            TaskStatus.SKIPPED,
            TaskStatus.CANCELLED,
            TaskStatus.FAILED,
        },
        TaskStatus.RUNNING: {
            TaskStatus.VERIFYING,
            TaskStatus.FAILED,
            TaskStatus.RECOVERING,
            TaskStatus.INTERRUPTED,
            TaskStatus.CANCELLED,
        },
        TaskStatus.VERIFYING: {
            TaskStatus.PASSED,
            TaskStatus.FAILED,
            TaskStatus.RECOVERING,
            TaskStatus.INTERRUPTED,
            TaskStatus.CANCELLED,
        },
        TaskStatus.FAILED: {
            TaskStatus.RECOVERING,
            TaskStatus.PENDING,
            TaskStatus.CANCELLED,
        },
        TaskStatus.RECOVERING: {
            TaskStatus.RUNNING,
            TaskStatus.VERIFYING,
            TaskStatus.FAILED,
            TaskStatus.INTERRUPTED,
            TaskStatus.CANCELLED,
        },
        TaskStatus.PASSED: set(),
        TaskStatus.SKIPPED: set(),
        TaskStatus.CANCELLED: set(),
        TaskStatus.INTERRUPTED: {
            TaskStatus.PASSED,
            TaskStatus.FAILED,
            TaskStatus.RECOVERING,
            TaskStatus.CANCELLED,
        },
    }

    TERMINAL_STATES = {
        TaskStatus.PASSED,
        TaskStatus.SKIPPED,
        TaskStatus.CANCELLED,
    }

    RESUMABLE_STATES = {
        TaskStatus.INTERRUPTED,
    }

    def __init__(
        self,
        graph: TaskGraph,
        transitions: Optional[List[StateTransition]] = None,
    ):
        self.graph = graph
        self._transitions: List[StateTransition] = list(transitions or [])

    @property
    def transitions(self) -> List[StateTransition]:
        return list(self._transitions)

    def allowed_transitions(self, task_id: str) -> List[TaskStatus]:
        task = self.graph.get(task_id)
        return sorted(
            self.ALLOWED_TRANSITIONS[task.status],
            key=lambda status: status.value,
        )

    def can_transition(
        self,
        task_id: str,
        new_status: TaskStatus,
    ) -> bool:
        task = self.graph.get(task_id)

        if not isinstance(new_status, TaskStatus):
            new_status = TaskStatus(new_status)

        return new_status in self.ALLOWED_TRANSITIONS[task.status]

    def transition(
        self,
        task_id: str,
        new_status: TaskStatus,
        reason: str = "",
        evidence: Optional[Dict[str, Any]] = None,
    ) -> StateTransition:
        task = self.graph.get(task_id)

        if not isinstance(new_status, TaskStatus):
            new_status = TaskStatus(new_status)

        previous_status = task.status

        if not self.can_transition(task_id, new_status):
            raise InvalidTaskTransitionError(
                task_id,
                previous_status,
                new_status,
            )

        task.status = new_status

        transition = StateTransition(
            task_id=task_id,
            previous_status=previous_status,
            new_status=new_status,
            reason=reason,
            evidence=dict(evidence or {}),
        )
        self._transitions.append(transition)

        task.add_evidence(
            "STATE_TRANSITION",
            reason or f"{previous_status.value} -> {new_status.value}",
            {
                "previous_status": previous_status.value,
                "new_status": new_status.value,
                **dict(evidence or {}),
            },
        )

        # Dependency-derived states are refreshed after the explicit
        # transition so newly unblocked/dependent tasks are reflected.
        self.graph.refresh_states()

        return transition

    def mark_ready_tasks(self) -> List[str]:
        """
        Refresh dependency state and return tasks that are now READY.

        READY is dependency-derived, so this method intentionally does not
        create artificial transition-history entries for every refresh.
        """
        self.graph.refresh_states()
        return [
            task.task_id
            for task in self.graph.ready_tasks()
            if task.status == TaskStatus.READY
        ]

    def assert_can_execute(self, task_id: str) -> None:
        task = self.graph.get(task_id)

        if task.status != TaskStatus.READY:
            raise ExecutionStateError(
                f"Task {task_id} is not executable; "
                f"current state is {task.status.value}"
            )

    def assert_not_terminal(self, task_id: str) -> None:
        task = self.graph.get(task_id)

        if task.status in self.TERMINAL_STATES:
            raise ExecutionStateError(
                f"Task {task_id} is terminal in state {task.status.value}"
            )

    def validate(self) -> None:
        """
        Validate graph consistency and recorded transition history.

        This does not reconstruct history by replaying it; it confirms that
        every recorded transition itself was legal according to the state
        machine.
        """
        self.graph.validate()

        for transition in self._transitions:
            allowed = self.ALLOWED_TRANSITIONS[transition.previous_status]
            if transition.new_status not in allowed:
                raise InvalidTaskTransitionError(
                    transition.task_id,
                    transition.previous_status,
                    transition.new_status,
                )

            # Every transition must reference a real task.
            self.graph.get(transition.task_id)

    def to_dict(self) -> Dict[str, Any]:
        self.validate()
        return {
            "transitions": [
                transition.to_dict()
                for transition in self._transitions
            ]
        }

    @classmethod
    def from_dict(
        cls,
        graph: TaskGraph,
        payload: Dict[str, Any],
    ) -> "TaskExecutionStateMachine":
        transitions = [
            StateTransition.from_dict(item)
            for item in payload.get("transitions", [])
        ]

        machine = cls(graph, transitions=transitions)
        machine.validate()
        return machine
