import unittest

from engineering_agent.execution_state import (
    ExecutionStateError,
    InvalidTaskTransitionError,
    StateTransition,
    TaskExecutionStateMachine,
)
from engineering_agent.task_graph import (
    EngineeringTask,
    TaskGraph,
    TaskStatus,
)


class ExecutionStateTests(unittest.TestCase):
    def make_machine(self):
        graph = TaskGraph(
            [
                EngineeringTask(
                    task_id="build",
                    title="Build",
                    objective="Build component",
                ),
                EngineeringTask(
                    task_id="tests",
                    title="Tests",
                    objective="Run tests",
                    dependencies=["build"],
                ),
            ]
        )
        machine = TaskExecutionStateMachine(graph)
        return graph, machine

    def test_ready_task_can_enter_running(self):
        graph, machine = self.make_machine()

        graph.refresh_states()

        transition = machine.transition(
            "build",
            TaskStatus.RUNNING,
            reason="worker started",
        )

        self.assertEqual(
            transition.previous_status,
            TaskStatus.READY,
        )
        self.assertEqual(
            transition.new_status,
            TaskStatus.RUNNING,
        )
        self.assertEqual(
            graph.get("build").status,
            TaskStatus.RUNNING,
        )

    def test_running_task_can_verify_and_pass(self):
        graph, machine = self.make_machine()
        graph.refresh_states()

        machine.transition("build", TaskStatus.RUNNING)
        machine.transition("build", TaskStatus.VERIFYING)

        transition = machine.transition(
            "build",
            TaskStatus.PASSED,
            reason="verification succeeded",
            evidence={"tests_passed": 5},
        )

        self.assertEqual(transition.new_status, TaskStatus.PASSED)
        self.assertEqual(graph.get("build").status, TaskStatus.PASSED)

        self.assertEqual(
            graph.get("tests").status,
            TaskStatus.READY,
        )

    def test_invalid_transition_is_rejected(self):
        _, machine = self.make_machine()

        with self.assertRaises(InvalidTaskTransitionError):
            machine.transition("build", TaskStatus.PASSED)

    def test_terminal_task_cannot_restart(self):
        graph, machine = self.make_machine()
        graph.refresh_states()

        machine.transition("build", TaskStatus.RUNNING)
        machine.transition("build", TaskStatus.VERIFYING)
        machine.transition("build", TaskStatus.PASSED)

        with self.assertRaises(InvalidTaskTransitionError):
            machine.transition("build", TaskStatus.RUNNING)

    def test_blocked_task_cannot_verify(self):
        _, machine = self.make_machine()

        with self.assertRaises(InvalidTaskTransitionError):
            machine.transition("tests", TaskStatus.VERIFYING)

    def test_failure_can_enter_recovery(self):
        graph, machine = self.make_machine()
        graph.refresh_states()

        machine.transition("build", TaskStatus.RUNNING)
        machine.transition(
            "build",
            TaskStatus.FAILED,
            reason="tests failed",
        )
        transition = machine.transition(
            "build",
            TaskStatus.RECOVERING,
            reason="attempt structured recovery",
        )

        self.assertEqual(transition.previous_status, TaskStatus.FAILED)
        self.assertEqual(transition.new_status, TaskStatus.RECOVERING)

    def test_recovery_can_return_to_verification(self):
        graph, machine = self.make_machine()
        graph.refresh_states()

        machine.transition("build", TaskStatus.RUNNING)
        machine.transition("build", TaskStatus.FAILED)
        machine.transition("build", TaskStatus.RECOVERING)

        transition = machine.transition(
            "build",
            TaskStatus.VERIFYING,
            reason="recovery patch applied",
        )

        self.assertEqual(
            transition.new_status,
            TaskStatus.VERIFYING,
        )

    def test_non_ready_execution_is_rejected(self):
        _, machine = self.make_machine()

        with self.assertRaises(ExecutionStateError):
            machine.assert_can_execute("tests")

    def test_transition_history_and_evidence_are_recorded(self):
        graph, machine = self.make_machine()
        graph.refresh_states()

        machine.transition(
            "build",
            TaskStatus.RUNNING,
            reason="start implementation",
            evidence={"backend": "fake"},
        )

        self.assertEqual(len(machine.transitions), 1)

        transition = machine.transitions[0]
        self.assertIsInstance(transition, StateTransition)
        self.assertEqual(transition.evidence["backend"], "fake")

        task_evidence = graph.get("build").evidence[-1]
        self.assertEqual(task_evidence.kind, "STATE_TRANSITION")
        self.assertEqual(task_evidence.data["backend"], "fake")

    def test_allowed_transitions_are_explicit(self):
        _, machine = self.make_machine()

        self.assertEqual(
            machine.allowed_transitions("build"),
            sorted(
                [
                    TaskStatus.READY,
                    TaskStatus.BLOCKED,
                    TaskStatus.SKIPPED,
                    TaskStatus.CANCELLED,
                ],
                key=lambda status: status.value,
            ),
        )

    def test_state_machine_round_trip(self):
        graph, machine = self.make_machine()
        graph.refresh_states()

        machine.transition(
            "build",
            TaskStatus.RUNNING,
            reason="started",
        )

        payload = machine.to_dict()
        restored = TaskExecutionStateMachine.from_dict(
            graph,
            payload,
        )

        self.assertEqual(len(restored.transitions), 1)
        self.assertEqual(
            restored.transitions[0].new_status,
            TaskStatus.RUNNING,
        )

    def test_unknown_task_fails(self):
        graph, machine = self.make_machine()

        with self.assertRaises(Exception):
            machine.transition("missing", TaskStatus.RUNNING)


if __name__ == "__main__":
    unittest.main()
