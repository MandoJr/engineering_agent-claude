import shutil
import tempfile
import unittest
from pathlib import Path

from engineering_agent.config import AgentConfig
from engineering_agent.execution_state import TaskExecutionStateMachine
from engineering_agent.models import EngineeringPlan, PlannedChange
from engineering_agent.task_decomposer import TaskDecomposer
from engineering_agent.task_graph import TaskStatus
from engineering_agent.task_resume import TaskResumeEngine
from engineering_agent.task_verification import TaskVerificationEngine
from engineering_agent.tester import TestEngineer as EngineeringTestRunner
from engineering_agent.tools import Permission, ToolBox


class ResumeTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp())
        self.config = AgentConfig(project_root=self.root)

    def tearDown(self):
        shutil.rmtree(self.root, ignore_errors=True)

    def make_task(self, content, verification):
        plan = EngineeringPlan(
            plan_id="plan",
            files=["app.py"],
            changes=[
                PlannedChange(
                    file="app.py",
                    description="task",
                    verification=verification,
                )
            ],
        )
        (self.root / "app.py").write_text(
            content,
            encoding="utf-8",
        )
        graph = TaskDecomposer.from_plan(plan)
        task = graph.get("plan:task-001")
        machine = TaskExecutionStateMachine(graph)
        return graph, task, machine

    def verifier(self):
        return TaskVerificationEngine(
            EngineeringTestRunner(
                ToolBox(
                    self.config,
                    Permission.READ_ONLY,
                )
            )
        )

    def test_interrupted_task_that_already_completed_is_reconciled(self):
        graph, task, machine = self.make_task(
            "VALUE = 2\n",
            [
                'python -c "assert open(\'app.py\').read().strip() == \'VALUE = 2\'"'
            ],
        )

        task.status = TaskStatus.RUNNING
        machine.transition(
            task.task_id,
            TaskStatus.INTERRUPTED,
            reason="simulated interruption",
        )

        reconciliation = TaskResumeEngine(
            self.verifier()
        ).reconcile(graph, machine)

        self.assertEqual(
            reconciliation.reconciled_passed,
            ["plan:task-001"],
        )
        self.assertEqual(
            task.status,
            TaskStatus.PASSED,
        )

    def test_interrupted_task_that_did_not_complete_becomes_failed(self):
        graph, task, machine = self.make_task(
            "VALUE = 1\n",
            [
                'python -c "assert open(\'app.py\').read().strip() == \'VALUE = 2\'"'
            ],
        )

        task.status = TaskStatus.RUNNING
        machine.transition(
            task.task_id,
            TaskStatus.INTERRUPTED,
            reason="simulated interruption",
        )

        reconciliation = TaskResumeEngine(
            self.verifier()
        ).reconcile(graph, machine)

        self.assertEqual(
            reconciliation.reconciled_failed,
            ["plan:task-001"],
        )
        self.assertEqual(
            task.status,
            TaskStatus.FAILED,
        )

    def test_active_tasks_can_be_marked_interrupted(self):
        graph, task, machine = self.make_task(
            "VALUE = 1\n",
            [
                'python -c "print(1)"'
            ],
        )

        task.status = TaskStatus.READY
        machine.transition(
            task.task_id,
            TaskStatus.RUNNING,
            reason="started",
        )

        interrupted = TaskResumeEngine.mark_active_tasks_interrupted(
            graph,
            machine,
        )

        self.assertEqual(interrupted, ["plan:task-001"])
        self.assertEqual(
            task.status,
            TaskStatus.INTERRUPTED,
        )

    def test_passed_tasks_are_never_marked_interrupted(self):
        graph, task, machine = self.make_task(
            "VALUE = 2\n",
            [
                'python -c "assert open(\'app.py\').read().strip() == \'VALUE = 2\'"'
            ],
        )

        task.status = TaskStatus.READY
        machine.transition(
            task.task_id,
            TaskStatus.RUNNING,
        )
        machine.transition(
            task.task_id,
            TaskStatus.VERIFYING,
        )
        machine.transition(
            task.task_id,
            TaskStatus.PASSED,
        )

        interrupted = TaskResumeEngine.mark_active_tasks_interrupted(
            graph,
            machine,
        )

        self.assertEqual(interrupted, [])
        self.assertEqual(task.status, TaskStatus.PASSED)


if __name__ == "__main__":
    unittest.main()
