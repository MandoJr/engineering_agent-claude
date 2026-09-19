from engineering_agent.task_graph import TaskGraph
from engineering_agent.execution_state import TaskExecutionStateMachine
import json
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

from engineering_agent.backend import CallableBackend, SimpleRouter
from engineering_agent.models import RunStatus
from engineering_agent.orchestrator import ApprovalError, EngineeringOrchestrator, ResumeError
from engineering_agent.task_graph import TaskStatus


PLAN_JSON = json.dumps({
    "affected_systems": ["demo"],
    "files": ["consumer.py", "base.py"],
    "changes": [
        {
            "file": "consumer.py",
            "description": "Create consumer using base",
            "change_type": "CREATE",
            "objective": "Create consumer module",
            "prerequisites": ["base.py"],
            "verification": [
                "python -c \"from pathlib import Path; assert Path('consumer.py').read_text().strip().splitlines() == ['from base import VALUE', 'RESULT = VALUE + 1']\""
            ],
            "completion_criteria": ["consumer exists"],
            "risk": "LOW",
        },
        {
            "file": "base.py",
            "description": "Create base module",
            "change_type": "CREATE",
            "objective": "Create base module",
            "prerequisites": [],
            "verification": [
                "python -c \"from pathlib import Path; assert Path('base.py').read_text().strip() == 'VALUE = 41'\""
            ],
            "completion_criteria": ["base exists"],
            "risk": "LOW",
        },
    ],
    "risks": [],
    "tests": [
        "python -c \"from base import VALUE; from consumer import RESULT; assert RESULT == 42\""
    ],
    "benchmarks": [],
    "expected_result": "consumer uses base successfully",
    "risk_level": "LOW",
})


class PublicEngineeringLifecycleTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp())
        subprocess.run(
            ["git", "init", "-q"],
            cwd=self.root,
            check=True,
        )
        subprocess.run(
            ["git", "config", "user.email", "test@test.com"],
            cwd=self.root,
            check=True,
        )
        subprocess.run(
            ["git", "config", "user.name", "test"],
            cwd=self.root,
            check=True,
        )

        (self.root / "README.txt").write_text(
            "baseline\n",
            encoding="utf-8",
        )

        subprocess.run(
            ["git", "add", "-A"],
            cwd=self.root,
            check=True,
        )
        subprocess.run(
            ["git", "commit", "-q", "-m", "baseline"],
            cwd=self.root,
            check=True,
        )

        self.implementation_calls = []

        def fake_complete(
            prompt,
            system=None,
            json_mode=False,
            timeout=120.0,
        ):
            if system and "Coding Planner" in system:
                return PLAN_JSON

            if system and "Coding Implementer" in system:
                self.implementation_calls.append(prompt)

                if "APPROVED TASK FILE: base.py" in prompt:
                    return json.dumps({
                        "changes": [{
                            "operation": "CREATE_FILE",
                            "content": "VALUE = 41\n",
                            "reason": "Create base",
                        }]
                    })

                if "APPROVED TASK FILE: consumer.py" in prompt:
                    return json.dumps({
                        "changes": [{
                            "operation": "CREATE_FILE",
                            "content": (
                                "from base import VALUE\n"
                                "RESULT = VALUE + 1\n"
                            ),
                            "reason": "Create consumer",
                        }]
                    })

            return "ok"

        backend = CallableBackend(
            name="fake",
            fn=fake_complete,
            capabilities=[
                "code_generation",
                "debugging",
                "code_review",
                "repository_analysis",
                "deep_reasoning",
            ],
        )

        self.agent = EngineeringOrchestrator(
            project_root=str(self.root),
            router=SimpleRouter([backend]),
        )

    def tearDown(self):
        shutil.rmtree(self.root, ignore_errors=True)

    def test_public_lifecycle_executes_dependency_graph(self):
        proposal = self.agent.propose(
            "Create base and consumer modules",
            task_profile="IMPLEMENT",
        )

        self.assertEqual(
            proposal.status,
            "WAITING_FOR_APPROVAL",
        )

        with self.assertRaises(ApprovalError):
            self.agent.implement(proposal.proposal_id)

        self.agent.approve(proposal.proposal_id)

        run = self.agent.implement(proposal.proposal_id)

        self.assertEqual(
            run.status,
            RunStatus.COMPLETE.value,
        )
        self.assertIsNotNone(run.task_graph)
        self.assertTrue(run.task_review["passed"])

        statuses = {
            task_id: task["status"]
            for task_id, task in run.task_graph["tasks"].items()
        }

        self.assertTrue(
            all(
                status == TaskStatus.PASSED.value
                for status in statuses.values()
            )
        )

        self.assertEqual(
            (self.root / "base.py").read_text(
                encoding="utf-8"
            ).strip(),
            "VALUE = 41",
        )
        self.assertEqual(
            (self.root / "consumer.py").read_text(
                encoding="utf-8"
            ).strip(),
            "from base import VALUE\nRESULT = VALUE + 1",
        )

        history = self.agent.history()
        self.assertEqual(len(history), 1)
        self.assertEqual(history[0].run_id, run.run_id)

    def test_public_resume_reconciles_interrupted_task(self):
        proposal = self.agent.propose(
            "Create base and consumer modules",
            task_profile="IMPLEMENT",
        )
        self.agent.approve(proposal.proposal_id)

        run = self.agent._find_run_for_proposal(
            proposal.proposal_id
        )
        self.assertIsNotNone(run)

        persisted = self.agent._get_run(run.run_id)

        base_task_id = next(
            task_id
            for task_id, task in persisted.task_graph["tasks"].items()
            if task["metadata"]["source_file"] == "base.py"
        )

        graph = TaskGraph.from_dict(
            persisted.task_graph
        )
        state_machine = TaskExecutionStateMachine.from_dict(
            graph,
            {"transitions": persisted.state_transitions},
        )

        state_machine.transition(
            base_task_id,
            TaskStatus.RUNNING,
            reason="Simulated task execution before process interruption.",
        )
        state_machine.transition(
            base_task_id,
            TaskStatus.INTERRUPTED,
            reason="Simulated process interruption during active task.",
        )

        persisted.task_graph = graph.to_dict()
        persisted.state_transitions = [
            transition.to_dict()
            for transition in state_machine.transitions
        ]
        persisted.current_task_id = base_task_id

        # The implementation actually reached disk before the simulated
        # interruption.
        (self.root / "base.py").write_text(
            "VALUE = 41\n",
            encoding="utf-8",
        )

        self.agent.storage.save_run(persisted)

        resumed = self.agent.resume(run.run_id)

        self.assertEqual(
            resumed.status,
            RunStatus.COMPLETE.value,
        )

        self.assertEqual(
            resumed.task_graph["tasks"][base_task_id]["status"],
            TaskStatus.PASSED.value,
        )

        self.assertTrue(
            (self.root / "consumer.py").exists()
        )

        # Base was reconciled and therefore never sent back through the
        # implementer.
        self.assertEqual(
            len(self.implementation_calls),
            1,
        )
        self.assertIn(
            "consumer.py",
            self.implementation_calls[0],
        )

    def test_public_resume_rejects_corrupt_persisted_run(self):
        proposal = self.agent.propose(
            "Create base and consumer modules",
            task_profile="IMPLEMENT",
        )
        self.agent.approve(proposal.proposal_id)

        run = self.agent._find_run_for_proposal(
            proposal.proposal_id
        )
        self.assertIsNotNone(run)

        run_path = self.agent.storage.runs._path(run.run_id)
        run_path.write_text(
            '{"corrupted": ',
            encoding="utf-8",
        )

        with self.assertRaises(ResumeError):
            self.agent.resume(run.run_id)

    def test_public_resume_rejects_inconsistent_transition_history(self):
        proposal = self.agent.propose(
            "Create base and consumer modules",
            task_profile="IMPLEMENT",
        )
        self.agent.approve(proposal.proposal_id)

        run = self.agent._find_run_for_proposal(
            proposal.proposal_id
        )
        self.assertIsNotNone(run)

        persisted = self.agent._get_run(run.run_id)

        transitions = [
            {
                "task_id": "not-a-real-task",
                "previous_status": "READY",
                "new_status": "RUNNING",
                "reason": "tampered",
                "evidence": {},
            }
        ]

        persisted.state_transitions = transitions
        self.agent.storage.save_run(persisted)

        with self.assertRaises(ResumeError):
            self.agent.resume(run.run_id)

    def test_public_resume_still_requires_approval(self):
        proposal = self.agent.propose(
            "Create base and consumer modules",
            task_profile="IMPLEMENT",
        )

        run = self.agent._find_run_for_proposal(
            proposal.proposal_id
        )
        self.assertIsNotNone(run)

        with self.assertRaises(ApprovalError):
            self.agent.resume(run.run_id)


if __name__ == "__main__":
    unittest.main()
