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
from .models import EngineeringGoal, EngineeringPlan, EngineeringProposal, FileEdit, RiskLevel


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
