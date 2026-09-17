"""
Engineering Analyst / Coding Planner.

Turns (goal, repository context, relevant lessons) into a structured
EngineeringPlan by prompting the routed backend for JSON and validating
the result defensively -- a malformed or incomplete backend response
degrades to a minimal, clearly-marked plan rather than raising, so the
loop can still surface something reviewable to the user.
"""

from __future__ import annotations

import json
import re
from typing import List, Optional

from .backend import ModelBackend
from .models import EngineeringGoal, EngineeringLesson, EngineeringPlan, PlannedChange, RepositoryContext, RiskLevel

PLANNER_SYSTEM_PROMPT = """You are the Coding Planner inside an engineering agent for a system \
called JARVIS. You turn an engineering goal, plus structured repository context, into a \
precise, reviewable engineering plan. You do NOT write implementation code here -- only a plan.

Respond with ONLY a JSON object (no prose, no markdown fences) with exactly these keys:
{
  "affected_systems": [string],
  "files": [string],              // relative paths, prefer files from candidate_files when possible
  "changes": [{"file": string, "description": string, "change_type": "MODIFY"|"CREATE"|"DELETE", "objective": string, "prerequisites": [string], "verification": [string], "completion_criteria": [string], "risk": "LOW"|"MEDIUM"|"HIGH"}],
  "risks": [string],
  "tests": [string],              // test file paths or commands that should validate this change
  "benchmarks": [string],         // named signals worth comparing before/after (may be empty)
  "expected_result": string,
  "risk_level": "LOW"|"MEDIUM"|"HIGH"
}

Prefer minimal, targeted changes over rewriting whole files. If the goal is too vague to plan \
safely, set risk_level to "HIGH", keep changes minimal/empty, and explain why in expected_result."""


def _extract_json(text: str) -> dict:
    text = text.strip()
    # Strip markdown code fences if the backend added them despite instructions.
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.MULTILINE)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Best-effort: find the first {...} block.
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if match:
            return json.loads(match.group(0))
        raise


class CodingPlanner:
    def __init__(self, backend: ModelBackend):
        self.backend = backend

    def plan(self, goal: EngineeringGoal, repo_context: RepositoryContext,
              relevant_lessons: Optional[List[EngineeringLesson]] = None) -> EngineeringPlan:
        lessons_block = self._format_lessons(relevant_lessons or [])
        prompt = f"""ENGINEERING GOAL
{goal.description}

TASK PROFILE: {goal.task_profile}

REPOSITORY CONTEXT
Affected systems (heuristic guess): {repo_context.affected_systems}
Candidate files: {repo_context.candidate_files}
Related tests found: {repo_context.related_tests}
Related config found: {repo_context.related_config}
Known top-level symbols per file: {json.dumps(repo_context.symbols)[:4000]}
Structural dependency map: {json.dumps(repo_context.module_graph)[:6000]}
Repository-analysis risks: {repo_context.risks}
Entry points: {repo_context.entry_points}
Git status: {repo_context.git_status_summary[:1000]}

RELEVANT PAST LESSONS
{lessons_block}

Produce the engineering plan JSON now."""

        raw = self.backend.complete(prompt, system=PLANNER_SYSTEM_PROMPT, json_mode=True)

        try:
            parsed = _extract_json(raw)
        except (json.JSONDecodeError, AttributeError):
            return self._fallback_plan(goal, repo_context, raw,
                                        reason="backend response was not valid JSON")

        try:
            changes = [
                PlannedChange(
                    file=c.get("file", ""),
                    description=c.get("description", ""),
                    change_type=c.get("change_type", "MODIFY"),
                    objective=c.get("objective", c.get("description", "")),
                    prerequisites=list(c.get("prerequisites", [])),
                    verification=list(c.get("verification", [])),
                    completion_criteria=list(c.get("completion_criteria", [])),
                    risk=c.get("risk", RiskLevel.LOW.value),
                )
                for c in parsed.get("changes", [])
            ]
            risk_level = parsed.get("risk_level", RiskLevel.MEDIUM.value)
            if risk_level not in (RiskLevel.LOW.value, RiskLevel.MEDIUM.value, RiskLevel.HIGH.value):
                risk_level = RiskLevel.MEDIUM.value

            return EngineeringPlan(
                goal_id=goal.goal_id,
                affected_systems=parsed.get("affected_systems", repo_context.affected_systems),
                files=parsed.get("files", []),
                changes=changes,
                risks=list(repo_context.risks) + parsed.get("risks", []),
                tests=parsed.get("tests", repo_context.related_tests),
                benchmarks=parsed.get("benchmarks", []),
                expected_result=parsed.get("expected_result", ""),
                risk_level=risk_level,
                raw_backend_output=raw,
                assumptions=parsed.get("assumptions", []),
                verification_requirements=parsed.get("verification_requirements", []),
            )
        except (TypeError, AttributeError) as exc:
            return self._fallback_plan(goal, repo_context, raw,
                                        reason=f"backend response had unexpected shape: {exc}")

    def revise(self, goal: EngineeringGoal, repo_context: RepositoryContext,
               prior_plan: EngineeringPlan, evidence: List[str],
               relevant_lessons: Optional[List[EngineeringLesson]] = None) -> EngineeringPlan:
        """Re-plan with observed evidence instead of retrying a stale plan."""
        revision_goal = EngineeringGoal(
            goal_id=goal.goal_id, task_profile=goal.task_profile,
            description=(f"{goal.description}\n\nPLAN REVISION REQUIRED. "
                         f"Previous plan: {prior_plan.to_dict()}\n"
                         f"Observed evidence: {evidence}"),
        )
        revised = self.plan(revision_goal, repo_context, relevant_lessons)
        revised.revision = prior_plan.revision + 1
        return revised

    def _fallback_plan(self, goal: EngineeringGoal, repo_context: RepositoryContext,
                        raw: str, reason: str) -> EngineeringPlan:
        return EngineeringPlan(
            goal_id=goal.goal_id,
            affected_systems=repo_context.affected_systems,
            files=[],
            changes=[],
            risks=[f"Plan generation degraded: {reason}. Needs manual planning before proceeding."],
            tests=[],
            benchmarks=[],
            expected_result="UNKNOWN -- planning failed, do not implement without a human plan.",
            risk_level=RiskLevel.HIGH.value,
            raw_backend_output=raw,
        )

    def _format_lessons(self, lessons: List[EngineeringLesson]) -> str:
        if not lessons:
            return "(none retrieved)"
        lines = []
        for lesson in lessons:
            lines.append(
                f"- [{lesson.confidence:.2f} confidence] {lesson.task}: "
                f"{lesson.solution or lesson.result} "
                f"(cause: {lesson.cause or 'n/a'})"
            )
        return "\n".join(lines)
