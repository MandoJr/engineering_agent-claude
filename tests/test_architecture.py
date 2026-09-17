import shutil
import tempfile
import unittest
import json
from pathlib import Path

from ..config import AgentConfig
from ..models import EngineeringPlan, FileEdit, PlannedChange, TestResult
from ..models import ChangeOperation, ChangeSet, StructuredChange, PatchResult
from ..patching import PatchApplier, PatchValidationError, parse_change_set
from ..repository_graph import RepositoryGraph
from ..reviewer import CodeReviewer
from ..tools import Permission, ToolBox
from ..backend import CallableBackend, SimpleRouter
from ..orchestrator import EngineeringOrchestrator
from ..structured_recovery import StructuredFailureRecoveryEngineer
from ..tester import TestEngineer

class ArchitectureTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp())
        (self.root / "pkg").mkdir()
        (self.root / "pkg" / "producer.py").write_text("def useful():\n    return 1\n")
        (self.root / "pkg" / "consumer.py").write_text("from pkg.producer import useful\ndef use():\n    return useful()\n")
        self.config = AgentConfig(project_root=self.root)

    def tearDown(self):
        shutil.rmtree(self.root, ignore_errors=True)

    def test_graph_tracks_importers_callers_and_impact(self):
        graph = RepositoryGraph(self.root).build()
        self.assertEqual(graph.importers_of("pkg.producer"), ["pkg/consumer.py"])
        self.assertEqual(graph.callers_of("useful"), ["pkg/consumer.py"])
        self.assertIn("pkg/consumer.py", graph.impacted_by(["pkg/producer.py"]))

    def test_verification_shell_does_not_accept_composition(self):
        tools = ToolBox(self.config, Permission.READ_ONLY)
        self.assertFalse(tools.run_test("python -c \"print(1)\"; echo unsafe").ok)
        self.assertFalse(tools.run_test("powershell -Command Get-ChildItem").ok)
        self.assertTrue(tools.run_test("python -c \"print(1)\"").ok)

    def test_deletion_needs_configuration_gate(self):
        victim = self.root / "pkg" / "victim.py"; victim.write_text("x = 1\n")
        tools = ToolBox(self.config, Permission.IMPLEMENT)
        self.assertFalse(tools.delete_file("pkg/victim.py").ok)
        self.assertTrue(victim.exists())
        self.config.allow_file_deletion = True
        self.assertTrue(tools.delete_file("pkg/victim.py").ok)
        self.assertFalse(victim.exists())

    def test_independent_review_rejects_out_of_scope_and_failed_tests(self):
        plan = EngineeringPlan(files=["pkg/producer.py"], changes=[PlannedChange(file="pkg/producer.py")])
        review = CodeReviewer(self.config).review_implementation(
            plan, [FileEdit(file="other.py", diff="diff")],
            [TestResult(command="python -m pytest", passed=False)],
        )
        self.assertFalse(review.passed)
        self.assertTrue(any("NOT in the approved" in finding for finding in review.findings))

    def test_plan_revision_discards_prior_approval(self):
        payload = json.dumps({"affected_systems": ["pkg"], "files": ["pkg/producer.py"],
            "changes": [{"file": "pkg/producer.py", "description": "change", "change_type": "MODIFY"}],
            "risks": [], "tests": ["python -m pytest -q"], "benchmarks": [],
            "expected_result": "changed", "risk_level": "LOW"})
        backend = CallableBackend("fake", lambda *args, **kwargs: payload,
            ["code_generation", "debugging", "code_review", "repository_analysis", "deep_reasoning"])
        agent = EngineeringOrchestrator(str(self.root), router=SimpleRouter([backend]), config=self.config)
        proposal = agent.propose("Change producer", task_profile="IMPLEMENT")
        agent.approve(proposal.proposal_id)
        revised = agent.revise_plan(proposal.proposal_id, ["test showed a boundary case"])
        self.assertEqual(revised.status, "WAITING_FOR_APPROVAL")
        self.assertEqual(revised.plan.revision, 1)

    def test_structured_patch_preview_apply_and_diff(self):
        path = self.root / "pkg" / "producer.py"
        tools = ToolBox(self.config, Permission.IMPLEMENT)
        patch = ChangeSet(changes=[StructuredChange(file="pkg/producer.py", operation="REPLACE_TEXT",
            expected_content="return 1", content="return 2", reason="update value")])
        preview = PatchApplier(tools).preview(patch, ["pkg/producer.py"])
        self.assertIn("-    return 1", preview.changes[0].diff)
        results, success = PatchApplier(tools).apply(patch, ["pkg/producer.py"])
        self.assertTrue(success); self.assertEqual(results[0].lines_added, 1)
        self.assertIn("return 2", path.read_text())

    def test_patch_rejects_stale_ambiguous_scope_and_invalid_python(self):
        tools = ToolBox(self.config, Permission.IMPLEMENT)
        applier = PatchApplier(tools)
        stale = ChangeSet(changes=[StructuredChange(file="pkg/producer.py", operation="REPLACE_TEXT", expected_content="missing", content="x")])
        result, ok = applier.apply(stale, ["pkg/producer.py"])
        self.assertFalse(ok); self.assertEqual(result[0].classification, "STALE_CONTEXT")
        outside = ChangeSet(changes=[StructuredChange(file="elsewhere.py", operation="CREATE_FILE", content="x=1")])
        result, ok = applier.apply(outside, ["pkg/producer.py"])
        self.assertFalse(ok); self.assertEqual(result[0].classification, "SCOPE_VIOLATION")
        invalid = ChangeSet(changes=[StructuredChange(file="pkg/producer.py", operation="REPLACE_SYMBOL", target_symbol="useful", content="def useful(:\n")])
        result, ok = applier.apply(invalid, ["pkg/producer.py"])
        self.assertFalse(ok); self.assertEqual(result[0].classification, "SYNTAX_INVALID")

    def test_multi_file_patch_and_duplicate_prevention(self):
        tools = ToolBox(self.config, Permission.IMPLEMENT)
        changes = ChangeSet(changes=[
            StructuredChange(change_id="one", file="pkg/producer.py", operation="REPLACE_TEXT", expected_content="return 1", content="return 3"),
            StructuredChange(change_id="two", file="pkg/new.py", operation="CREATE_FILE", content="VALUE = 3\n"),
        ])
        results, ok = PatchApplier(tools).apply(changes, ["pkg/producer.py", "pkg/new.py"])
        self.assertTrue(ok); self.assertEqual(len([r for r in results if r.applied]), 2)
        duplicate = ChangeSet(changes=[StructuredChange(change_id="same", file="pkg/producer.py", operation="REPLACE_TEXT", expected_content="return 3", content="return 4"), StructuredChange(change_id="same", file="pkg/producer.py", operation="REPLACE_TEXT", expected_content="return 4", content="return 5")])
        result, ok = PatchApplier(tools).apply(duplicate, ["pkg/producer.py"])
        self.assertFalse(ok); self.assertEqual(result[0].classification, "DUPLICATE_PATCH")

    def test_parse_rejects_malformed_patch(self):
        with self.assertRaises(PatchValidationError):
            parse_change_set("not json")

    def test_apply_failure_rolls_back_prior_file(self):
        (self.root / "pkg" / "second.py").write_text("VALUE = 1\n")
        class FailingTools(ToolBox):
            def edit_file(inner, relative_path, new_content):
                if relative_path == "pkg/second.py":
                    from ..tools import ToolResult
                    return ToolResult(False, error="simulated write failure")
                return super(FailingTools, inner).edit_file(relative_path, new_content)
        tools = FailingTools(self.config, Permission.IMPLEMENT)
        patch = ChangeSet(changes=[
            StructuredChange(file="pkg/producer.py", operation="REPLACE_TEXT", expected_content="return 1", content="return 9"),
            StructuredChange(file="pkg/second.py", operation="REPLACE_TEXT", expected_content="VALUE = 1", content="VALUE = 9"),
        ])
        results, ok = PatchApplier(tools).apply(patch, ["pkg/producer.py", "pkg/second.py"])
        self.assertFalse(ok); self.assertTrue(results[-1].rollback_performed)
        self.assertIn("return 1", (self.root / "pkg" / "producer.py").read_text())

    def test_recovery_uses_patches_and_prevents_duplicate_retry(self):
        response = json.dumps({"root_cause": "simulated", "changes": [{"operation": "REPLACE_TEXT", "expected_content": "return 1", "content": "return 2"}]})
        backend = CallableBackend("fake", lambda *args, **kwargs: response, ["code_generation"])
        tools = ToolBox(self.config, Permission.IMPLEMENT)
        failed = TestResult(command="python -c \"import sys; sys.exit(1)\"", passed=False, classification="ASSERTION_FAILURE")
        plan = EngineeringPlan(plan_id="plan", files=["pkg/producer.py"])
        attempts = StructuredFailureRecoveryEngineer(backend, tools, TestEngineer(ToolBox(self.config)), max_attempts=2).recover(plan, "pkg/producer.py", failed)
        self.assertEqual(len(attempts), 2)
        self.assertIn("DUPLICATE_PATCH", attempts[-1].evidence)
        self.assertIn("return 2", (self.root / "pkg" / "producer.py").read_text())

    def test_reviewer_reads_patch_and_git_diff_evidence(self):
        plan = EngineeringPlan(files=["pkg/producer.py"], changes=[PlannedChange(file="pkg/producer.py")])
        review = CodeReviewer(self.config).review_implementation(
            plan, [FileEdit(file="pkg/producer.py", diff="@@\n-old\n+new\n")], [TestResult(passed=True)],
            [PatchResult(change_id="p", file="pkg/producer.py", operation="REPLACE_TEXT", applied=True, lines_removed=101)],
            "+++ b/unrelated.py\n",
        )
        self.assertFalse(review.passed)
        self.assertTrue(any("full-file rewrite" in item for item in review.findings))
        self.assertTrue(any("outside plan scope" in item for item in review.findings))

if __name__ == "__main__":
    unittest.main()
