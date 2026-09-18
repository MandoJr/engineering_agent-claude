import shutil
import tempfile
import unittest
from pathlib import Path

from engineering_agent.backend import CallableBackend
from engineering_agent.config import AgentConfig
from engineering_agent.execution_state import TaskExecutionStateMachine
from engineering_agent.models import EngineeringPlan, PlannedChange
from engineering_agent.structured_implementer import StructuredCodingImplementer
from engineering_agent.task_decomposer import TaskDecomposer
from engineering_agent.task_execution import TaskExecutionEngine
from engineering_agent.task_graph import TaskStatus
from engineering_agent.task_recovery import TaskRecoveryCoordinator
from engineering_agent.tester import TestEngineer as EngineeringTestRunner
from engineering_agent.tools import Permission, ToolBox


class TaskRecoveryCoordinatorTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp())
        (self.root / "first.py").write_text(
            "VALUE = 1\n",
            encoding="utf-8",
        )
        (self.root / "second.py").write_text(
            "VALUE = 10\n",
            encoding="utf-8",
        )
        self.config = AgentConfig(project_root=self.root)

    def tearDown(self):
        shutil.rmtree(self.root, ignore_errors=True)

    def _verify_first(self):
        return [
            'python -c "from pathlib import Path; assert Path(\'first.py\').read_text().strip() == \'VALUE = 2\'"'
        ]

    def _verify_second(self):
        return [
            'python -c "from pathlib import Path; assert Path(\'second.py\').read_text().strip() == \'VALUE = 20\'"'
        ]

    def _build_engine(self, responses):
        def fake_complete(prompt, system=None, json_mode=False, timeout=120.0):
            return responses.pop(0)

        backend = CallableBackend(
            "fake",
            fake_complete,
            [
                "code_generation",
                "debugging",
                "code_review",
                "repository_analysis",
                "deep_reasoning",
            ],
        )

        implementer = StructuredCodingImplementer(
            backend,
            ToolBox(self.config, Permission.IMPLEMENT),
        )
        tester = EngineeringTestRunner(
            ToolBox(self.config, Permission.READ_ONLY)
        )
        engine = TaskExecutionEngine(
            implementer,
            tester,
        )

        return backend, tester, engine

    def test_successful_recovery_unlocks_dependent_task(self):
        # First implementation intentionally creates a failed verification.
        # Recovery fixes it. The dependent task then executes.
        responses = [
            '{"changes":[{"operation":"REPLACE_TEXT","expected_content":"VALUE = 1","content":"VALUE = 0","reason":"introduce failure"}]}',
            '{"root_cause":"wrong value","changes":[{"operation":"REPLACE_TEXT","expected_content":"VALUE = 0","content":"VALUE = 2","reason":"repair value"}]}',
            '{"changes":[{"operation":"REPLACE_TEXT","expected_content":"VALUE = 10","content":"VALUE = 20","reason":"dependent update"}]}',
        ]

        backend, tester, engine = self._build_engine(responses)

        plan = EngineeringPlan(
            plan_id="plan",
            files=["first.py", "second.py"],
            changes=[
                PlannedChange(
                    file="first.py",
                    description="Fix first file",
                    verification=self._verify_first(),
                ),
                PlannedChange(
                    file="second.py",
                    description="Update dependent file",
                    verification=self._verify_second(),
                    prerequisites=["first.py"],
                ),
            ],
        )

        graph = TaskDecomposer.from_plan(plan)
        machine = TaskExecutionStateMachine(graph)

        initial = engine.execute(
            "run",
            plan,
            graph,
            machine,
        )

        self.assertEqual(initial.failed_tasks, ["plan:task-001"])
        self.assertEqual(initial.blocked_tasks, ["plan:task-002"])

        coordinator = TaskRecoveryCoordinator(
            backend=backend,
            tools=ToolBox(self.config, Permission.IMPLEMENT),
            tester=tester,
            execution_engine=engine,
            max_attempts=2,
        )

        recovered = coordinator.recover(
            "run",
            plan,
            graph,
            machine,
            initial,
        )

        self.assertEqual(
            recovered.recovered_tasks,
            ["plan:task-001"],
        )
        self.assertEqual(
            graph.get("plan:task-001").status,
            TaskStatus.PASSED,
        )
        self.assertEqual(
            graph.get("plan:task-002").status,
            TaskStatus.PASSED,
        )
        self.assertEqual(recovered.blocked_tasks, [])

        self.assertEqual(
            (self.root / "first.py").read_text(encoding="utf-8").strip(),
            "VALUE = 2",
        )
        self.assertEqual(
            (self.root / "second.py").read_text(encoding="utf-8").strip(),
            "VALUE = 20",
        )

    def test_failed_recovery_keeps_dependents_blocked(self):
        responses = [
            '{"changes":[{"operation":"REPLACE_TEXT","expected_content":"VALUE = 1","content":"VALUE = 0","reason":"introduce failure"}]}',
            '{"root_cause":"not confident","changes":[]}',
        ]

        backend, tester, engine = self._build_engine(responses)

        plan = EngineeringPlan(
            plan_id="plan",
            files=["first.py", "second.py"],
            changes=[
                PlannedChange(
                    file="first.py",
                    description="Fail first task",
                    verification=self._verify_first(),
                ),
                PlannedChange(
                    file="second.py",
                    description="Dependent task",
                    verification=self._verify_second(),
                    prerequisites=["first.py"],
                ),
            ],
        )

        graph = TaskDecomposer.from_plan(plan)
        machine = TaskExecutionStateMachine(graph)

        initial = engine.execute(
            "run",
            plan,
            graph,
            machine,
        )

        coordinator = TaskRecoveryCoordinator(
            backend=backend,
            tools=ToolBox(self.config, Permission.IMPLEMENT),
            tester=tester,
            execution_engine=engine,
            max_attempts=2,
        )

        recovered = coordinator.recover(
            "run",
            plan,
            graph,
            machine,
            initial,
        )

        self.assertEqual(recovered.recovered_tasks, [])
        self.assertIn("plan:task-001", recovered.failed_tasks)
        self.assertEqual(
            graph.get("plan:task-001").status,
            TaskStatus.FAILED,
        )
        self.assertEqual(
            graph.get("plan:task-002").status,
            TaskStatus.BLOCKED,
        )
        self.assertEqual(
            recovered.blocked_tasks,
            ["plan:task-002"],
        )

    def test_recovery_attempt_preserves_patch_evidence(self):
        responses = [
            '{"changes":[{"operation":"REPLACE_TEXT","expected_content":"VALUE = 1","content":"VALUE = 0","reason":"introduce failure"}]}',
            '{"root_cause":"repair","changes":[{"operation":"REPLACE_TEXT","expected_content":"VALUE = 0","content":"VALUE = 2","reason":"repair"}]}',
        ]

        backend, tester, engine = self._build_engine(responses)

        plan = EngineeringPlan(
            plan_id="plan",
            files=["first.py"],
            changes=[
                PlannedChange(
                    file="first.py",
                    description="Fix",
                    verification=self._verify_first(),
                )
            ],
        )

        graph = TaskDecomposer.from_plan(plan)
        machine = TaskExecutionStateMachine(graph)

        initial = engine.execute(
            "run",
            plan,
            graph,
            machine,
        )

        coordinator = TaskRecoveryCoordinator(
            backend=backend,
            tools=ToolBox(self.config, Permission.IMPLEMENT),
            tester=tester,
            execution_engine=engine,
            max_attempts=2,
        )

        recovered = coordinator.recover(
            "run",
            plan,
            graph,
            machine,
            initial,
        )

        self.assertEqual(len(recovered.attempts), 1)
        self.assertTrue(recovered.attempts[0].succeeded)
        self.assertTrue(recovered.attempts[0].patch_results)
        self.assertTrue(
            recovered.attempts[0].patch_results[0].applied
        )


if __name__ == "__main__":
    unittest.main()

