import unittest

from engineering_agent.task_graph import (
    DependencyCycleError,
    DuplicateTaskError,
    EngineeringTask,
    TaskGraph,
    TaskStatus,
    UnknownTaskError,
)


class TaskGraphTests(unittest.TestCase):
    def make_task(self, task_id, priority=100):
        return EngineeringTask(
            task_id=task_id,
            title=f"Task {task_id}",
            objective=f"Objective {task_id}",
            affected_files=[f"{task_id}.py"],
            priority=priority,
        )

    def test_dependency_graph_and_ready_tasks(self):
        graph = TaskGraph(
            [
                self.make_task("A"),
                self.make_task("B"),
                self.make_task("C"),
            ]
        )

        graph.add_dependency("B", "A")
        graph.add_dependency("C", "B")

        ready = [task.task_id for task in graph.ready_tasks()]

        self.assertEqual(ready, ["A"])
        self.assertEqual(graph.dependencies_of("C"), ["B"])
        self.assertEqual(graph.dependents_of("A"), ["B"])

    def test_deterministic_topological_order(self):
        graph = TaskGraph(
            [
                self.make_task("build", 20),
                self.make_task("tests", 30),
                self.make_task("config", 10),
            ]
        )

        graph.add_dependency("build", "config")
        graph.add_dependency("tests", "build")

        self.assertEqual(
            graph.topological_order(),
            ["config", "build", "tests"],
        )

    def test_cycle_detection(self):
        graph = TaskGraph(
            [
                self.make_task("A"),
                self.make_task("B"),
                self.make_task("C"),
            ]
        )

        graph.add_dependency("B", "A")
        graph.add_dependency("C", "B")

        with self.assertRaises(DependencyCycleError):
            graph.add_dependency("A", "C")

    def test_duplicate_task_is_rejected(self):
        graph = TaskGraph([self.make_task("A")])

        with self.assertRaises(DuplicateTaskError):
            graph.add_task(self.make_task("A"))

    def test_unknown_dependency_is_rejected(self):
        graph = TaskGraph([self.make_task("A")])

        with self.assertRaises(UnknownTaskError):
            graph.add_dependency("A", "missing")

    def test_failure_propagates_to_dependents(self):
        graph = TaskGraph(
            [
                self.make_task("A"),
                self.make_task("B"),
                self.make_task("C"),
            ]
        )

        graph.add_dependency("B", "A")
        graph.add_dependency("C", "B")

        graph.mark_status("A", TaskStatus.FAILED)

        self.assertEqual(graph.get("B").status, TaskStatus.BLOCKED)
        self.assertEqual(graph.get("C").status, TaskStatus.BLOCKED)
        self.assertEqual(graph.dependency_failures("B"), ["A"])

    def test_success_unlocks_next_task(self):
        graph = TaskGraph(
            [
                self.make_task("A"),
                self.make_task("B"),
            ]
        )

        graph.add_dependency("B", "A")

        self.assertEqual(
            [task.task_id for task in graph.ready_tasks()],
            ["A"],
        )

        graph.mark_status("A", TaskStatus.PASSED)

        self.assertEqual(
            [task.task_id for task in graph.ready_tasks()],
            ["B"],
        )

    def test_evidence_is_structured(self):
        graph = TaskGraph([self.make_task("A")])

        graph.record_evidence(
            "A",
            "TEST_RESULT",
            "Unit tests passed",
            {"passed": 4},
        )

        evidence = graph.get("A").evidence[0]

        self.assertEqual(evidence.kind, "TEST_RESULT")
        self.assertEqual(evidence.message, "Unit tests passed")
        self.assertEqual(evidence.data["passed"], 4)
        self.assertTrue(evidence.timestamp)

    def test_graph_round_trip(self):
        task = self.make_task("A")
        task.required_capabilities = ["code_generation"]
        task.verification_commands = ["python -m pytest -q"]
        task.completion_criteria = ["tests pass"]
        task.add_evidence("TEST", "passed", {"count": 5})

        graph = TaskGraph([task])

        restored = TaskGraph.from_json(graph.to_json())

        self.assertEqual(restored.get("A").title, "Task A")
        self.assertEqual(
            restored.get("A").required_capabilities,
            ["code_generation"],
        )
        self.assertEqual(
            restored.get("A").evidence[0].data["count"],
            5,
        )


if __name__ == "__main__":
    unittest.main()
