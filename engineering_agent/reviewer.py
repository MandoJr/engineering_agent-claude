"""
Code Reviewer.

Two jobs:
1. Pre-implementation: validate a plan's scope (file count, forbidden
   paths, risk level) and turn it into an EngineeringProposal.
2. Post-implementation: check the actual edits stayed within the
   approved plan's file list -- catching the "implementation agent
   quietly touched an unrelated file" failure mode the spec calls out.
"""

from __future__ import annotations

from typing import List, Tuple

from .config import AgentConfig
from .models import EngineeringGoal, EngineeringPlan, EngineeringProposal, FileEdit, PatchResult, RiskLevel, ReviewResult, TestResult


class ScopeViolation(RuntimeError):
    pass


class CodeReviewer:
    def __init__(self, config: AgentConfig):
        self.config = config

    def review_plan(self, goal: EngineeringGoal, plan: EngineeringPlan) -> Tuple[List[str], str]:
        """Returns (warnings, effective_risk_level). Does not mutate the plan."""
        warnings: List[str] = []
        risk = plan.risk_level

        if len(plan.files) > self.config.max_files_per_change:
            warnings.append(
                f"Plan touches {len(plan.files)} files, over the "
                f"{self.config.max_files_per_change}-file guardrail -- consider narrowing scope."
            )
            risk = RiskLevel.HIGH.value

        for forbidden in self.config.forbidden_paths:
            hits = [f for f in plan.files if f == forbidden or f.startswith(forbidden + "/")]
            if hits:
                warnings.append(f"Plan touches forbidden path '{forbidden}': {hits}")
                risk = RiskLevel.HIGH.value

        if not plan.tests:
            warnings.append("Plan defines no tests to validate the change.")
            if risk == RiskLevel.LOW.value:
                risk = RiskLevel.MEDIUM.value

        if not plan.changes:
            warnings.append("Plan has no concrete changes -- likely a degraded/fallback plan.")
            risk = RiskLevel.HIGH.value

        return warnings, risk

    def build_proposal(self, goal: EngineeringGoal, plan: EngineeringPlan,
                        is_self_improvement: bool = False) -> EngineeringProposal:
        warnings, risk = self.review_plan(goal, plan)
        expected = plan.expected_result
        if warnings:
            expected += "\n\nReviewer warnings:\n" + "\n".join(f"- {w}" for w in warnings)

        return EngineeringProposal(
            goal=goal,
            plan=plan,
            affected_files=list(plan.files),
            risk=risk,
            expected_improvement=expected,
            required_approval="USER",
            is_self_improvement=is_self_improvement,
        )

    def check_implementation_scope(self, plan: EngineeringPlan, edits: List[FileEdit]) -> List[str]:
        """Post-implementation scope check: flag any edited file that wasn't
        in the approved plan. This does NOT undo anything -- it surfaces
        the violation so the orchestrator/evaluator can record it and,
        per the spec, feed it back as a self-improvement signal."""
        approved = set(plan.files)
        violations = []
        for edit in edits:
            if edit.file not in approved:
                violations.append(
                    f"Edited '{edit.file}' which was NOT in the approved plan's file list"
                )
        return violations

    def review_implementation(self, plan: EngineeringPlan, edits: List[FileEdit],
                              tests: List[TestResult], patch_results: List[PatchResult] = None,
                              git_diff: str = "") -> ReviewResult:
        """Independent, deterministic post-change review over actual evidence.

        This does not trust the implementer's success flag: it checks the
        concrete diff, approved scope, empty edits, and verification outcome.
        A backend reviewer may be layered on top of this evidence later.
        """
        findings = self.check_implementation_scope(plan, edits)
        declared = {change.file for change in plan.changes}
        for edit in edits:
            if edit.file not in declared:
                findings.append(f"Edited '{edit.file}' without a concrete planned change")
            if edit.change_type != "DELETE" and not edit.diff:
                findings.append(f"'{edit.file}' produced no recorded diff")
        for patch in patch_results or []:
            if not patch.applied:
                findings.append(f"Patch {patch.change_id or '<unknown>'} was not applied: {patch.classification}")
            if patch.applied and patch.operation == "REPLACE_TEXT" and patch.lines_removed > 100:
                findings.append(f"Patch {patch.change_id} replaced {patch.lines_removed} lines; possible full-file rewrite")
            if patch.applied and patch.file not in declared:
                findings.append(f"Structured patch changed '{patch.file}' outside declared tasks")
        if not tests:
            findings.append("No verification results were recorded")
        elif not all(result.passed for result in tests):
            findings.append("One or more verification commands failed")
        evidence = [f"reviewed {len(edits)} recorded edit(s)",
                    f"{sum(t.passed for t in tests)}/{len(tests)} verification command(s) passed"]
        if git_diff:
            diff_files = {line[6:] for line in git_diff.splitlines() if line.startswith("+++ b/")}
            unexpected = sorted(diff_files - set(plan.files))
            evidence.append(f"reviewed Git diff for {len(diff_files)} file(s)")
            if unexpected:
                findings.append("Git diff includes files outside plan scope: " + ", ".join(unexpected))
        return ReviewResult(passed=not findings, findings=findings, evidence=evidence)
