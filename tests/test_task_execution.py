import shutil
import tempfile
import unittest
from pathlib import Path

from engineering_agent.backend import CallableBackend
from engineering_agent.config import AgentConfig
from engineering_agent.execution_state import TaskExecutionStateMachine
from engineering_agent.models import EngineeringPlan, PlannedChange
from engineering_agent.patching import PatchApplier
from engineering_agent.structured_implementer import StructuredCodingImplementer
from engineering_agent.task_decomposer import TaskDecomposer
from engineering_agent.task_execution import TaskExecutionEngine
from engineering_agent.task_graph import TaskStatus, TaskGraph
from engineering_agent.tester import TestEngineer
from engineering_agent.tools import Permission, ToolBox

TestEngineer.__test__ = False


class TaskDecomposerTests(unittest.TestCase):
    def test_same_file_changes_become_dependency_chain(self):
        plan = EngineeringPlan(
            plan_id="plan",
            files=["app.py"],
            changes=[
                PlannedChange(
                    file="app.py",
                    description="Add function",
                    change_type="MODIFY",
                ),
                PlannedChange(
                    file="app.py",
                    description="Update function",
                    change_type="MODIFY",
                ),
            ],
            tests=["python -c \"print(1)\""],
        )

        graph = TaskDecomposer.from_plan(plan)

        tasks = list(graph.tasks.values())

        self.assertEqual(len(tasks), 2)
        self.assertEqual(
            tasks[1].dependencies,
            [tasks[0].task_id],
        )
        self.assertEqual(tasks[0].status, TaskStatus.READY)
        self.assertEqual(tasks[1].status, TaskStatus.BLOCKED)

    def test_explicit_prerequisite_becomes_dependency(self):
        plan = EngineeringPlan(
            plan_id="plan",
            changes=[
                PlannedChange(
                    file="base.py",
                    description="Create base layer",
                    change_type="CREATE",
                ),
                PlannedChange(
                    file="consumer.py",
                    description="Use base layer",
                    change_type="MODIFY",
                    prerequisites=["base.py"],
                ),
            ],
        )

        graph = TaskDecomposer.from_plan(plan)

        tasks = list(graph.tasks.values())

        self.assertEqual(
            tasks[1].dependencies,
            [tasks[0].task_id],
        )


class TaskExecutionEngineTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp())
        (self.root / "app.py").write_text(
            "VALUE = 1\n",
            encoding="utf-8",
        )
        self.config = AgentConfig(project_root=self.root)

    def tearDown(self):
        shutil.rmtree(self.root, ignore_errors=True)

    def test_multi_task_execution_follows_dependencies(self):
        responses = [
            '{"changes":[{"operation":"REPLACE_TEXT","expected_content":"VALUE = 1","content":"VALUE = 2","reason":"first"}]}',
            '{"changes":[{"operation":"REPLACE_TEXT","expected_content":"VALUE = 2","content":"VALUE = 3","reason":"second"}]}',
        ]
        calls = []

        def fake_complete(prompt, system=None, json_mode=False, timeout=120.0):
            calls.append(prompt)
            return responses.pop(0)

        backend = CallableBackend(
            "fake",
            fake_complete,
            ["code_generation", "debugging", "code_review",
             "repository_analysis", "deep_reasoning"],
        )

        tools = ToolBox(self.config, Permission.IMPLEMENT)
        implementer = StructuredCodingImplementer(backend, tools)
        tester = TestEngineer(ToolBox(self.config, Permission.READ_ONLY))

        plan = EngineeringPlan(
            plan_id="plan",
            files=["app.py"],
            changes=[
                PlannedChange(
                    file="app.py",
                    description="First change",
                    change_type="MODIFY",
                    verification=['python -c "from pathlib import Path; assert Path(\'app.py\').read_text().strip() == \'VALUE = 2\'"'],
                ),
                PlannedChange(
                    file="app.py",
                    description="Second change",
                    change_type="MODIFY",
                    verification=['python -c "from pathlib import Path; assert Path(\'app.py\').read_text().strip() == \'VALUE = 3\'"'],
                ),
            ],
        )

        graph = TaskDecomposer.from_plan(plan)
        machine = TaskExecutionStateMachine(graph)

        outcome = TaskExecutionEngine(
            implementer,
            tester,
        ).execute("run", plan, graph, machine)

        self.assertTrue(outcome.implementation.success)
        self.assertEqual(len(outcome.completed_tasks), 2)
        self.assertEqual(outcome.failed_tasks, [])
        self.assertEqual(outcome.blocked_tasks, [])
        self.assertEqual(len(calls), 2)
        self.assertEqual(
            (self.root / "app.py").read_text(encoding="utf-8").strip(),
            "VALUE = 3",
        )

    def test_failed_task_blocks_dependent_task(self):
        backend = CallableBackend(
            "fake",
            lambda *args, **kwargs:
                '{"changes":[{"operation":"REPLACE_TEXT","expected_content":"missing","content":"VALUE = 2","reason":"fail"}]}',
            ["code_generation"],
        )

        tools = ToolBox(self.config, Permission.IMPLEMENT)
        implementer = StructuredCodingImplementer(backend, tools)
        tester = TestEngineer(ToolBox(self.config, Permission.READ_ONLY))

        plan = EngineeringPlan(
            plan_id="plan",
            files=["app.py"],
            changes=[
                PlannedChange(
                    file="app.py",
                    description="Failing change",
                    change_type="MODIFY",
                    verification=['python -c "print(1)"'],
                ),
                PlannedChange(
                    file="app.py",
                    description="Dependent change",
                    change_type="MODIFY",
                    verification=['python -c "print(1)"'],
                ),
            ],
        )

        graph = TaskDecomposer.from_plan(plan)
        machine = TaskExecutionStateMachine(graph)

        outcome = TaskExecutionEngine(
            implementer,
            tester,
        ).execute("run", plan, graph, machine)

        self.assertFalse(outcome.implementation.success)
        self.assertEqual(len(outcome.failed_tasks), 1)
        self.assertEqual(len(outcome.blocked_tasks), 1)

        statuses = [
            task.status
            for task in graph.tasks.values()
        ]
        self.assertIn(TaskStatus.FAILED, statuses)
        self.assertIn(TaskStatus.BLOCKED, statuses)

    def test_checkpoint_receives_each_task_state_change(self):
        events = []

        backend = CallableBackend(
            "fake",
            lambda *args, **kwargs:
                '{"changes":[{"operation":"REPLACE_TEXT","expected_content":"VALUE = 1","content":"VALUE = 2","reason":"change"}]}',
            ["code_generation"],
        )

        tools = ToolBox(self.config, Permission.IMPLEMENT)
        implementer = StructuredCodingImplementer(backend, tools)
        tester = TestEngineer(ToolBox(self.config, Permission.READ_ONLY))

        plan = EngineeringPlan(
            plan_id="plan",
            files=["app.py"],
            changes=[
                PlannedChange(
                    file="app.py",
                    description="Change value",
                    verification=['python -c "assert open(\'app.py\').read().strip() == \'VALUE = 2\'"'],
                )
            ],
        )

        graph = TaskDecomposer.from_plan(plan)
        machine = TaskExecutionStateMachine(graph)

        TaskExecutionEngine(
            implementer,
            tester,
            checkpoint=lambda task_id, status: events.append(
                (task_id, status)
            ),
        ).execute("run", plan, graph, machine)

        self.assertEqual(
            [status for _, status in events],
            [
                TaskStatus.RUNNING,
                TaskStatus.VERIFYING,
                TaskStatus.PASSED,
            ],
        )



    def test_prerequisite_can_reference_a_later_change(self):
        plan = EngineeringPlan(
            plan_id="plan",
            changes=[
                PlannedChange(
                    file="consumer.py",
                    description="Use base layer",
                    change_type="MODIFY",
                    prerequisites=["base.py"],
                ),
                PlannedChange(
                    file="base.py",
                    description="Create base layer",
                    change_type="CREATE",
                ),
            ],
        )

        graph = TaskDecomposer.from_plan(plan)

        consumer = graph.get("plan:task-001")
        base = graph.get("plan:task-002")

        self.assertEqual(
            consumer.dependencies,
            [base.task_id],
        )
        self.assertEqual(base.status, TaskStatus.READY)
        self.assertEqual(consumer.status, TaskStatus.BLOCKED)

    def test_unresolved_prerequisite_is_rejected(self):
        plan = EngineeringPlan(
            plan_id="plan",
            changes=[
                PlannedChange(
                    file="app.py",
                    description="Change app",
                    prerequisites=["missing.py"],
                ),
            ],
        )

        with self.assertRaises(Exception):
            TaskDecomposer.from_plan(plan)

    def test_ambiguous_file_prerequisite_is_rejected(self):
        plan = EngineeringPlan(
            plan_id="plan",
            changes=[
                PlannedChange(
                    file="base.py",
                    description="First base change",
                ),
                PlannedChange(
                    file="base.py",
                    description="Second base change",
                ),
                PlannedChange(
                    file="consumer.py",
                    description="Use base",
                    prerequisites=["base.py"],
                ),
            ],
        )

        with self.assertRaises(Exception):
            TaskDecomposer.from_plan(plan)


if __name__ == "__main__":
    unittest.main()


