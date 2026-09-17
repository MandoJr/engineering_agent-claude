"""
Engineering Learning System.

Creates EngineeringLesson records from completed runs, and retrieves
relevant lessons for future planning. Confidence starts conservative and
is only raised by repeated corroborating evidence across runs (mirrors
JARVIS's LESSONS_MIN_CONFIDENCE gate -- a lesson below that threshold is
retrieved but weighted low rather than discarded, so the planner sees
the trend without over-trusting a single data point).

If JARVIS already has a lessons system, this module is the natural place
to swap in an adapter that writes to / reads from that system instead of
local JSON storage (see storage.py's docstring for the same pattern).
"""

from __future__ import annotations

from typing import List, Optional

from .config import LESSONS_MIN_CONFIDENCE
from .models import EngineeringLesson, EngineeringRun, EvaluationResult
from .storage import EngineeringStorage

INITIAL_CONFIDENCE_SUCCESS = 0.55
INITIAL_CONFIDENCE_FAILURE = 0.6
CONFIDENCE_STEP = 0.08
MAX_CONFIDENCE = 0.97


class EngineeringLearningSystem:
    def __init__(self, storage: EngineeringStorage):
        self.storage = storage

    def create_lessons_from_run(self, run: EngineeringRun) -> List[EngineeringLesson]:
        lessons: List[EngineeringLesson] = []

        if run.evaluation is None:
            return lessons

        evaluation: EvaluationResult = run.evaluation
        task = run.goal.description if run.goal else "unknown goal"
        context = f"task_profile={run.task_profile}; affected_systems=" + \
                  (", ".join(run.plan.affected_systems) if run.plan else "unknown")

        if evaluation.improved_jarvis is False or run.failures:
            failure_summary = "; ".join(run.failures) if run.failures else evaluation.summary
            cause = self._infer_cause(run)
            solution = self._infer_solution(run)
            lesson = EngineeringLesson(
                task=task, context=context,
                attempt=self._attempt_summary(run),
                result="FAILURE",
                failure=failure_summary,
                cause=cause,
                solution=solution,
                confidence=self._merge_confidence(task, cause, INITIAL_CONFIDENCE_FAILURE),
                evidence=evaluation.evidence,
                source_run_id=run.run_id,
                tags=list(run.plan.affected_systems) if run.plan else [],
            )
            lessons.append(lesson)

        elif evaluation.improved_jarvis is True:
            lesson = EngineeringLesson(
                task=task, context=context,
                attempt=self._attempt_summary(run),
                result="SUCCESS",
                solution=f"Approach validated: {evaluation.summary}",
                confidence=self._merge_confidence(task, "success", INITIAL_CONFIDENCE_SUCCESS),
                evidence=evaluation.evidence,
                source_run_id=run.run_id,
                tags=list(run.plan.affected_systems) if run.plan else [],
            )
            lessons.append(lesson)

        # improved_jarvis is None (insufficient evidence): deliberately no
        # lesson is created -- recording an ambiguous outcome as a "lesson"
        # would pollute future planning with unearned confidence.

        for lesson in lessons:
            self.storage.save_lesson(lesson)

        return lessons

    def relevant_lessons(self, affected_systems: List[str], task_profile: str,
                          min_confidence: float = 0.0, limit: int = 10) -> List[EngineeringLesson]:
        """Retrieve lessons touching any of the given systems, most confident
        and most recent first. `min_confidence` defaults to 0 so callers see
        the full trend; pass LESSONS_MIN_CONFIDENCE to only see lessons
        JARVIS's existing threshold would consider trustworthy enough to
        act on directly."""
        all_lessons = self.storage.all_lessons()
        matches = []
        for data in all_lessons:
            tags = data.get("tags", [])
            if affected_systems and not (set(tags) & set(affected_systems)):
                continue
            if data.get("confidence", 0.0) < min_confidence:
                continue
            matches.append(EngineeringLesson(**data))

        matches.sort(key=lambda l: (l.confidence, l.created_at), reverse=True)
        return matches[:limit]

    def actionable_lessons(self, affected_systems: List[str]) -> List[EngineeringLesson]:
        return self.relevant_lessons(affected_systems, task_profile="",
                                      min_confidence=LESSONS_MIN_CONFIDENCE)

    # -- internals --------------------------------------------------------

    def _merge_confidence(self, task: str, cause: Optional[str], initial: float) -> float:
        """If a similar (same task text + cause) lesson already exists,
        nudge confidence up rather than starting over -- crude but honest
        corroboration tracking without a vector store."""
        existing = [
            EngineeringLesson(**d) for d in self.storage.all_lessons()
            if d.get("task") == task and d.get("cause") == cause
        ]
        if not existing:
            return initial
        best = max(e.confidence for e in existing)
        return min(best + CONFIDENCE_STEP, MAX_CONFIDENCE)

    def _attempt_summary(self, run: EngineeringRun) -> str:
        if run.implementation and run.implementation.edits:
            files = ", ".join(e.file for e in run.implementation.edits)
            return f"Edited: {files}"
        return "No implementation was applied."

    def _infer_cause(self, run: EngineeringRun) -> str:
        if run.recovery_attempts:
            last = run.recovery_attempts[-1]
            return last.root_cause or "Unknown root cause after recovery attempts."
        if run.tests and not all(t.passed for t in run.tests):
            failing = [t for t in run.tests if not t.passed]
            return f"Test failure: {failing[0].command} -> {failing[0].stderr_tail[-200:]}"
        return "Unknown -- no test or recovery evidence captured."

    def _infer_solution(self, run: EngineeringRun) -> Optional[str]:
        for attempt in reversed(run.recovery_attempts):
            if attempt.succeeded:
                return f"Recovery succeeded via: {attempt.recovery_plan}"
        return None
