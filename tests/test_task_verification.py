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
from engineering_agent.task_verification import TaskVerificationEngine
from engineering_agent.tester import TestEngineer as EngineeringTestRunner
from engineering_agent.tools import Permission, ToolBox


class TaskVerificationTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp())
        self.config = AgentConfig(project_root=self.root)

    def tearDown(self):
        shutil.rmtree(self.root, ignore_errors=True)

    def test_verification_deduplicates_commands_and_records_evidence(self):
        tester = EngineeringTestRunner(
            ToolBox(self.config, Permission.READ_ONLY)
        )
        verifier = TaskVerificationEngine(tester)

        task = TaskDecomposer.from_plan(
            EngineeringPlan(
                plan_id="plan",
                changes=[
                    PlannedChange(
                        file="app.py",
                        description="verify",
                        verification=[
                            'python -c "print(1)"',
                            'python -c "print(1)"',
                        ],
                    )
                ],
            )
        ).get("plan:task-001")

        outcome = verifier.verify(task)

        self.assertTrue(outcome.passed)
        self.assertEqual(len(outcome.results), 1)
        self.assertEqual(outcome.evidence["passed_count"], 1)
        self.assertEqual(outcome.evidence["failed_count"], 0)

    def test_missing_task_verification_is_not_treated_as_success(self):
        tester = EngineeringTestRunner(
            ToolBox(self.config, Permission.READ_ONLY)
        )
        verifier = TaskVerificationEngine(tester)

        task = TaskDecomposer.from_plan(
            EngineeringPlan(
                plan_id="plan",
                changes=[
                    PlannedChange(
                        file="app.py",
                        description="no verification",
                    )
                ],
            )
        ).get("plan:task-001")

        outcome = verifier.verify(task)

        self.assertFalse(outcome.passed)
        self.assertEqual(outcome.results, [])
        self.assertFalse(outcome.evidence["verification_defined"])


class TaskResumeTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp())
        (self.root / "app.py").write_text(
            "VALUE = 1\n",
            encoding="utf-8",
        )
        self.config = AgentConfig(project_root=self.root)

    def tearDown(self):
        shutil.rmtree(self.root, ignore_errors=True)

    def test_blocked_by_identifies_failed_dependency(self):
        backend = CallableBackend(
            "fake",
            lambda *args, **kwargs:
                '{"changes":[{"operation":"REPLACE_TEXT","expected_content":"missing","content":"VALUE = 2"}]}',
            ["code_generation"],
        )

        implementer = StructuredCodingImplementer(
            backend,
            ToolBox(self.config, Permission.IMPLEMENT),
        )
        tester = EngineeringTestRunner(
            ToolBox(self.config, Permission.READ_ONLY)
        )

        plan = EngineeringPlan(
            plan_id="plan",
            files=["app.py"],
            changes=[
                PlannedChange(
                    file="app.py",
                    description="first",
                ),
                PlannedChange(
                    file="app.py",
                    description="second",
                ),
            ],
        )

        graph = TaskDecomposer.from_plan(plan)
        machine = TaskExecutionStateMachine(graph)

        outcome = TaskExecutionEngine(
            implementer,
            tester,
        ).execute(
            "run",
            plan,
            graph,
            machine,
        )

        failed = outcome.failed_tasks[0]
        blocked = outcome.blocked_tasks[0]

        self.assertEqual(
            outcome.blocked_by[blocked],
            [failed],
        )

    def test_resume_executes_newly_unblocked_task(self):
        responses = [
            '{"changes":[{"operation":"REPLACE_TEXT","expected_content":"missing","content":"VALUE = 2"}]}',
            '{"changes":[{"operation":"REPLACE_TEXT","expected_content":"VALUE = 2","content":"VALUE = 3"}]}',
        ]

        def fake_complete(prompt, system=None, json_mode=False, timeout=120.0):
            return responses.pop(0)

        backend = CallableBackend(
            "fake",
            fake_complete,
            ["code_generation"],
        )

        implementer = StructuredCodingImplementer(
            backend,
            ToolBox(self.config, Permission.IMPLEMENT),
        )
        tester = EngineeringTestRunner(
            ToolBox(self.config, Permission.READ_ONLY)
        )

        verification = [
            'python -c "from pathlib import Path; assert Path(\'app.py\').read_text().strip() == \'VALUE = 3\'"'
        ]

        plan = EngineeringPlan(
            plan_id="plan",
            files=["app.py"],
            changes=[
                PlannedChange(
                    file="app.py",
                    description="first task",
                    verification=verification,
                ),
                PlannedChange(
                    file="app.py",
                    description="dependent task",
                    verification=verification,
                ),
            ],
        )

        graph = TaskDecomposer.from_plan(plan)
        machine = TaskExecutionStateMachine(graph)

        first = TaskExecutionEngine(
            implementer,
            tester,
        ).execute(
            "run",
            plan,
            graph,
            machine,
        )

        self.assertEqual(len(first.failed_tasks), 1)
        self.assertEqual(len(first.blocked_tasks), 1)

        failed_id = first.failed_tasks[0]

        # Simulate the existing recovery engineer successfully repairing
        # the failed task.
        (self.root / "app.py").write_text(
            "VALUE = 2\n",
            encoding="utf-8",
        )

        machine.transition(
            failed_id,
            TaskStatus.RECOVERING,
            reason="simulated recovery",
        )
        machine.transition(
            failed_id,
            TaskStatus.VERIFYING,
            reason="simulated recovery patch applied",
        )
        machine.transition(
            failed_id,
            TaskStatus.PASSED,
            reason="simulated recovery verification passed",
        )

        resumed = TaskExecutionEngine(
            implementer,
            tester,
        ).resume(
            "run",
            plan,
            graph,
            machine,
        )

        self.assertIn("plan:task-002", resumed.completed_tasks)
        self.assertEqual(
            graph.get("plan:task-002").status,
            TaskStatus.PASSED,
        )
        self.assertEqual(
            (self.root / "app.py").read_text(encoding="utf-8").strip(),
            "VALUE = 3",
        )


if __name__ == "__main__":
    unittest.main()

