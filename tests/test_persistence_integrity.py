import unittest

from engineering_agent.execution_state import (
    StateIntegrityError,
    StateTransition,
    TaskExecutionStateMachine,
)
from engineering_agent.task_graph import (
    EngineeringTask,
    TaskGraph,
    TaskStatus,
)


class PersistenceIntegrityTests(unittest.TestCase):

    def _graph(self):
        graph = TaskGraph()
        graph.add_task(
            EngineeringTask(
                task_id="task-001",
                title="Example task",
                objective="Validate persisted execution state.",
            )
        )
        return graph

    def test_transition_history_must_be_continuous(self):
        graph = self._graph()
        graph.get("task-001").status = TaskStatus.FAILED

        transitions = [
            StateTransition(
                task_id="task-001",
                previous_status=TaskStatus.READY,
                new_status=TaskStatus.RUNNING,
            ).to_dict(),
            StateTransition(
                task_id="task-001",
                previous_status=TaskStatus.READY,
                new_status=TaskStatus.FAILED,
            ).to_dict(),
        ]

        with self.assertRaises(StateIntegrityError):
            TaskExecutionStateMachine.from_dict(
                graph,
                {"transitions": transitions},
            )

    def test_graph_state_must_match_transition_history(self):
        graph = self._graph()
        graph.get("task-001").status = TaskStatus.PASSED

        transitions = [
            StateTransition(
                task_id="task-001",
                previous_status=TaskStatus.READY,
                new_status=TaskStatus.RUNNING,
            ).to_dict(),
        ]

        with self.assertRaises(StateIntegrityError):
            TaskExecutionStateMachine.from_dict(
                graph,
                {"transitions": transitions},
            )

    def test_non_derived_state_requires_transition_history(self):
        graph = self._graph()
        graph.get("task-001").status = TaskStatus.INTERRUPTED

        with self.assertRaises(StateIntegrityError):
            TaskExecutionStateMachine.from_dict(
                graph,
                {"transitions": []},
            )

    def test_valid_persisted_history_round_trips(self):
        graph = self._graph()
        graph.get("task-001").status = TaskStatus.PASSED

        transitions = [
            StateTransition(
                task_id="task-001",
                previous_status=TaskStatus.READY,
                new_status=TaskStatus.RUNNING,
            ).to_dict(),
            StateTransition(
                task_id="task-001",
                previous_status=TaskStatus.RUNNING,
                new_status=TaskStatus.VERIFYING,
            ).to_dict(),
            StateTransition(
                task_id="task-001",
                previous_status=TaskStatus.VERIFYING,
                new_status=TaskStatus.PASSED,
            ).to_dict(),
        ]

        machine = TaskExecutionStateMachine.from_dict(
            graph,
            {"transitions": transitions},
        )

        self.assertEqual(
            len(machine.transitions),
            3,
        )
        self.assertEqual(
            graph.get("task-001").status,
            TaskStatus.PASSED,
        )


if __name__ == "__main__":
    unittest.main()

