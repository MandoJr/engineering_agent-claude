"""
End-to-end smoke test for the full engineering loop, using a
CallableBackend fake so no network/Ollama server is required.

Run with: python -m unittest engineering_agent.tests.test_smoke -v
(or via pytest if available)
"""

import json
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

from engineering_agent.backend import CallableBackend, SimpleRouter
from engineering_agent.orchestrator import ApprovalError, EngineeringOrchestrator


PLAN_JSON = json.dumps({
    "affected_systems": ["routing"],
    "files": ["core/routing/router.py"],
    "changes": [{
        "file": "core/routing/router.py",
        "description": "Add backend_b for task y",
        "change_type": "MODIFY",
    }],
    "risks": [],
    "tests": [
        "python3 -c \"import sys; sys.path.insert(0, '.'); "
        "from core.routing.router import route; "
        "assert route('y') == 'backend_b'\""
    ],
    "benchmarks": [],
    "expected_result": "router handles task y",
    "risk_level": "LOW",
})

NEW_FILE_CONTENT = (
    'def route(task):\n'
    '    if task == "y":\n'
    '        return "backend_b"\n'
    '    return "backend_a"\n'
)

PATCH_JSON = json.dumps({
    "changes": [{
        "operation": "REPLACE_SYMBOL",
        "target_symbol": "route",
        "content": NEW_FILE_CONTENT,
        "reason": "Add route for task y",
    }]
})


def fake_complete(prompt, system=None, json_mode=False, timeout=120.0):
    if system and "Coding Planner" in system:
        return PLAN_JSON
    if system and "Coding Implementer" in system:
        return PATCH_JSON
    return "ok"


class EngineeringLoopSmokeTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        (self.tmp / "core" / "routing").mkdir(parents=True)
        (self.tmp / "core" / "routing" / "router.py").write_text(
            'def route(task):\n    return "backend_a"\n'
        )
        subprocess.run(["git", "init", "-q"], cwd=self.tmp, check=True)
        subprocess.run(["git", "config", "user.email", "t@t.com"], cwd=self.tmp, check=True)
        subprocess.run(["git", "config", "user.name", "t"], cwd=self.tmp, check=True)
        subprocess.run(["git", "add", "-A"], cwd=self.tmp, check=True)
        subprocess.run(["git", "commit", "-q", "-m", "init"], cwd=self.tmp, check=True)

        backend = CallableBackend(
            name="ollama:fake", fn=fake_complete,
            capabilities=["code_generation", "debugging", "code_review",
                           "repository_analysis", "deep_reasoning"],
        )
        self.agent = EngineeringOrchestrator(
            project_root=str(self.tmp), router=SimpleRouter([backend])
        )

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_full_loop_requires_approval_and_succeeds(self):
        proposal = self.agent.propose("Improve backend routing to support task y",
                                       task_profile="IMPLEMENT")
        self.assertEqual(proposal.status, "WAITING_FOR_APPROVAL")

        with self.assertRaises(ApprovalError):
            self.agent.implement(proposal.proposal_id)

        self.agent.approve(proposal.proposal_id)
        run = self.agent.implement(proposal.proposal_id)

        self.assertEqual(run.status, "COMPLETE")
        self.assertTrue(all(t.passed for t in run.tests))
        self.assertTrue(run.evaluation.code_works)
        # No benchmarks were defined in this plan, so the evaluator honestly
        # reports "insufficient evidence" for improved_jarvis rather than
        # assuming success -- and, by design, no lesson is created from an
        # ambiguous outcome (see learning.py).
        self.assertIsNone(run.evaluation.improved_jarvis)
        self.assertEqual(len(run.lessons_created), 0)

        history = self.agent.history()
        self.assertEqual(len(history), 1)

    def test_rejecting_a_proposal_blocks_implementation(self):
        proposal = self.agent.propose("Improve backend routing to support task y",
                                       task_profile="IMPLEMENT")
        self.agent.reject(proposal.proposal_id, notes="not now")
        with self.assertRaises(ApprovalError):
            self.agent.implement(proposal.proposal_id)

    def test_tools_refuse_paths_outside_project_root(self):
        from engineering_agent.tools import ToolBox, Permission, PathNotAllowed
        tools = ToolBox(self.agent.config, permission=Permission.READ_ONLY)
        result = tools.read_file("../../etc/passwd")
        self.assertFalse(result.ok)


if __name__ == "__main__":
    unittest.main()



