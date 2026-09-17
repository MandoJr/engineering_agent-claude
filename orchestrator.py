"""
Engineering Orchestrator.

The single entry point JARVIS is meant to call:

    agent = EngineeringOrchestrator(project_root="/path/to/jarvis")
    proposal = agent.propose("Improve backend routing")
    agent.approve(proposal.proposal_id)                 # explicit human step
    result = agent.implement(proposal.proposal_id)
    evaluation = agent.evaluate(result.run_id)
    agent.self_improve()

The orchestrator decides which specialists a task actually needs (a pure
ANALYZE/PLAN task never touches the implementer, tester, or recovery
engineer at all) rather than always running the full pipeline.

APPROVAL BOUNDARY: `implement()` refuses to run unless the stored
proposal's status is APPROVED. There is no parameter that overrides
this. Git commit/push go through git_utils, which enforces the same
rule independently at that layer too.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, List, Optional

from .backend import BackendError, ModelBackend, OllamaBackend, Router, SimpleRouter
from .config import AgentConfig, TASK_PROFILES
from .evaluator import BenchmarkFn, EvaluationEngineer, HealthChecker
from .git_utils import GitWorkflow
from .implementer import CodingImplementer
from .learning import EngineeringLearningSystem
from .models import (
    ApprovalStatus, EngineeringGoal, EngineeringProposal, EngineeringRun,
    RiskLevel, RunStatus, TaskProfileName,
)
from .planner import CodingPlanner
from .recovery import FailureRecoveryEngineer
from .repository import RepositoryAnalyst
from .reviewer import CodeReviewer
from .storage import EngineeringStorage
from .tester import TestEngineer
from .tools import Permission, ToolBox


class ApprovalError(RuntimeError):
    pass


class EngineeringOrchestrator:
    def __init__(self, project_root: str,
                 router: Optional[Router] = None,
                 health_checker: Optional[HealthChecker] = None,
                 benchmark_fn: Optional[BenchmarkFn] = None,
                 config: Optional[AgentConfig] = None):
        self.config = config or AgentConfig(project_root=Path(project_root))
        self.storage = EngineeringStorage(self.config.storage_dir)
        self.learning = EngineeringLearningSystem(self.storage)
        self.evaluator = EvaluationEngineer(health_checker=health_checker, benchmark_fn=benchmark_fn)
        self.reviewer = CodeReviewer(self.config)

        if router is not None:
            self.router = router
        else:
            self.router = SimpleRouter([OllamaBackend(
                endpoint=self.config.ollama_endpoint, model=self.config.ollama_model,
            )])

        try:
            self.git = GitWorkflow(self.config.project_root)
        except Exception:  # noqa: BLE001 -- git is optional for propose()/analysis-only use
            self.git = None

    # -- classification ---------------------------------------------------

    def classify(self, goal_description: str, task_profile: Optional[str] = None) -> EngineeringGoal:
        profile = task_profile or self._guess_task_profile(goal_description)
        if profile not in TASK_PROFILES:
            profile = TaskProfileName.ANALYZE.value
        return EngineeringGoal(description=goal_description, task_profile=profile)

    def _guess_task_profile(self, description: str) -> str:
        lowered = description.lower()
        keyword_map = [
            (("debug", "fix the error", "why does it fail"), TaskProfileName.DEBUG.value),
            (("refactor",), TaskProfileName.REFACTOR.value),
            (("optimi", "speed up", "performance"), TaskProfileName.OPTIMIZE.value),
            (("architecture", "redesign"), TaskProfileName.ARCHITECTURE.value),
            (("self-improve", "improve yourself", "improve your own"), TaskProfileName.SELF_IMPROVE.value),
            (("test",), TaskProfileName.TEST.value),
            (("review",), TaskProfileName.REVIEW.value),
            (("analy",), TaskProfileName.ANALYZE.value),
            (("plan",), TaskProfileName.PLAN.value),
        ]
        for keywords, profile in keyword_map:
            if any(k in lowered for k in keywords):
                return profile
        return TaskProfileName.IMPLEMENT.value

    # -- backend selection --------------------------------------------------

    def _select_backend(self, task_profile: str) -> ModelBackend:
        profile = TASK_PROFILES[task_profile]
        decision = self.router.select(task_profile, profile.required_backend_capabilities)
        return decision.backend

    # -- PROPOSE: goal -> analysis -> plan -> proposal ----------------------

    def propose(self, goal_description: str, task_profile: Optional[str] = None) -> EngineeringProposal:
        goal = self.classify(goal_description, task_profile)
        profile = TASK_PROFILES[goal.task_profile]

        run = EngineeringRun(goal=goal, task_profile=goal.task_profile, status=RunStatus.ANALYZING.value)

        backend = self._select_backend(goal.task_profile)
        run.backend = backend.name

        read_tools = ToolBox(self.config, permission=Permission.READ_ONLY)
        analyst = RepositoryAnalyst(read_tools, backend)
        repo_context = analyst.analyze(goal_description)
        run.repository_context = repo_context

        lessons = self.learning.relevant_lessons(repo_context.affected_systems, goal.task_profile)

        planner = CodingPlanner(backend)
        try:
            plan = planner.plan(goal, repo_context, lessons)
        except BackendError as exc:
            run.status = RunStatus.FAILED.value
            run.final_result = f"Planning failed: {exc}"
            run.failures.append(str(exc))
            self.storage.save_run(run)
            self.router.report_outcome(backend.name, goal.task_profile, success=False)
            raise

        run.plan = plan
        run.status = RunStatus.PLANNED.value

        proposal = self.reviewer.build_proposal(
            goal, plan, is_self_improvement=(goal.task_profile == TaskProfileName.SELF_IMPROVE.value)
        )
        if not profile.requires_user_approval:
            proposal.status = ApprovalStatus.NOT_REQUIRED.value

        run.proposal_id = proposal.proposal_id
        run.approval = proposal.status
        run.status = RunStatus.WAITING_FOR_APPROVAL.value if profile.requires_user_approval else RunStatus.PLANNED.value

        self.storage.save_proposal(proposal)
        self.storage.save_run(run)
        return proposal

    # -- REVIEW / APPROVE / REJECT ------------------------------------------

    def review(self, proposal_id: str) -> EngineeringProposal:
        data = self.storage.proposals.load(proposal_id)
        if data is None:
            raise KeyError(f"No such proposal: {proposal_id}")
        return _proposal_from_dict(data)

    def approve(self, proposal_id: str, notes: Optional[str] = None) -> EngineeringProposal:
        proposal = self.review(proposal_id)
        proposal.status = ApprovalStatus.APPROVED.value
        proposal.decided_at = __import__("time").time()
        proposal.decision_notes = notes
        self.storage.save_proposal(proposal)
        self._sync_run_approval(proposal)
        return proposal

    def reject(self, proposal_id: str, notes: Optional[str] = None) -> EngineeringProposal:
        proposal = self.review(proposal_id)
        proposal.status = ApprovalStatus.REJECTED.value
        proposal.decided_at = __import__("time").time()
        proposal.decision_notes = notes
        self.storage.save_proposal(proposal)
        self._sync_run_approval(proposal)
        return proposal

    def request_changes(self, proposal_id: str, notes: str) -> EngineeringProposal:
        proposal = self.review(proposal_id)
        proposal.status = ApprovalStatus.CHANGES_REQUESTED.value
        proposal.decision_notes = notes
        self.storage.save_proposal(proposal)
        self._sync_run_approval(proposal)
        return proposal

    def _sync_run_approval(self, proposal: EngineeringProposal) -> None:
        for data in self.storage.all_runs():
            if data.get("proposal_id") == proposal.proposal_id:
                run = _run_from_dict(data)
                run.approval = proposal.status
                if proposal.status == ApprovalStatus.REJECTED.value:
                    run.status = RunStatus.REJECTED.value
                run.touch()
                self.storage.save_run(run)
                break

    # -- IMPLEMENT: approved proposal -> edits on disk ----------------------

    def implement(self, proposal_id: str) -> EngineeringRun:
        proposal = self.review(proposal_id)
        if proposal.status not in (ApprovalStatus.APPROVED.value, ApprovalStatus.NOT_REQUIRED.value):
            raise ApprovalError(
                f"Refusing to implement proposal {proposal_id}: status is "
                f"'{proposal.status}', not APPROVED."
            )

        run = self._find_run_for_proposal(proposal_id)
        if run is None:
            raise KeyError(f"No engineering run found for proposal {proposal_id}")

        profile = TASK_PROFILES[run.task_profile]
        plan = proposal.plan

        # Baseline benchmarks, captured before any change (BASELINE step).
        baseline = self.evaluator.capture_baseline(plan.benchmarks)
        health_before = self.evaluator.snapshot_health(plan.affected_systems)

        write_tools = ToolBox(self.config, permission=Permission.IMPLEMENT)
        backend = self._select_backend(run.task_profile)
        implementer = CodingImplementer(backend, write_tools)

        run.status = RunStatus.IMPLEMENTING.value
        run.touch()
        self.storage.save_run(run)

        impl_result = implementer.implement(run.run_id, plan)
        run.implementation = impl_result

        if not impl_result.success:
            run.status = RunStatus.FAILED.value
            run.final_result = impl_result.error
            run.failures.append(impl_result.error or "implementation failed")
            run.touch()
            self.storage.save_run(run)
            self.router.report_outcome(backend.name, run.task_profile, success=False)
            return run

        scope_violations = self.reviewer.check_implementation_scope(plan, impl_result.edits)
        run.failures.extend(scope_violations)

        # TEST
        run.status = RunStatus.TESTING.value
        read_tools = ToolBox(self.config, permission=Permission.READ_ONLY)
        tester = TestEngineer(read_tools)
        test_results = tester.run_tests(plan.tests)
        run.tests = test_results

        # FAILURE RECOVERY IF NEEDED (bounded)
        if profile.requires_tests and not all(t.passed for t in test_results):
            run.status = RunStatus.RECOVERING.value
            self.storage.save_run(run)
            recovery_engineer = FailureRecoveryEngineer(
                backend, write_tools, tester, max_attempts=self.config.max_recovery_attempts
            )
            failing_test = next(t for t in test_results if not t.passed)
            failing_file = impl_result.edits[-1].file if impl_result.edits else (plan.files[0] if plan.files else "")
            if failing_file:
                attempts = recovery_engineer.recover(plan, failing_file, failing_test)
                run.recovery_attempts = attempts
                if attempts and attempts[-1].succeeded:
                    run.tests = tester.run_tests(plan.tests)
                else:
                    run.failures.append(
                        f"Recovery exhausted after {len(attempts)} attempt(s); "
                        f"last test result: {failing_test.command} failed."
                    )

        # EVALUATE
        run.status = RunStatus.EVALUATING.value
        health_after = self.evaluator.snapshot_health(plan.affected_systems)
        benchmarks = self.evaluator.run_benchmarks(plan.benchmarks, baseline)
        evaluation = self.evaluator.evaluate(
            run.run_id, plan, run.tests, health_before, health_after, benchmarks
        )
        run.evaluation = evaluation

        overall_success = evaluation.code_works and not scope_violations
        run.status = RunStatus.COMPLETE.value if overall_success else RunStatus.FAILED.value
        run.final_result = evaluation.summary
        run.touch()

        # LEARN
        lessons = self.learning.create_lessons_from_run(run)
        run.lessons_created = [l.lesson_id for l in lessons]

        self.storage.save_run(run)
        self.router.report_outcome(backend.name, run.task_profile, success=overall_success,
                                    evidence={"evaluation": evaluation.to_dict()})
        return run

    def _find_run_for_proposal(self, proposal_id: str) -> Optional[EngineeringRun]:
        for data in self.storage.all_runs():
            if data.get("proposal_id") == proposal_id:
                return _run_from_dict(data)
        return None

    # -- direct test/evaluate re-entry (CLI: `test <run_id>`, `evaluate <run_id>`) --

    def test(self, run_id: str) -> EngineeringRun:
        run = self._get_run(run_id)
        read_tools = ToolBox(self.config, permission=Permission.READ_ONLY)
        tester = TestEngineer(read_tools)
        run.tests = tester.run_tests(run.plan.tests if run.plan else [])
        run.touch()
        self.storage.save_run(run)
        return run

    def evaluate(self, run_id: str) -> EngineeringRun:
        run = self._get_run(run_id)
        if run.plan is None:
            raise ValueError(f"Run {run_id} has no plan to evaluate against")
        health_before = self.evaluator.snapshot_health(run.plan.affected_systems)
        benchmarks = self.evaluator.run_benchmarks(run.plan.benchmarks, {})
        evaluation = self.evaluator.evaluate(
            run.run_id, run.plan, run.tests, health_before, health_before, benchmarks
        )
        run.evaluation = evaluation
        run.touch()
        self.storage.save_run(run)
        return run

    def _get_run(self, run_id: str) -> EngineeringRun:
        data = self.storage.runs.load(run_id)
        if data is None:
            raise KeyError(f"No such run: {run_id}")
        return _run_from_dict(data)

    # -- HISTORY / LESSONS ---------------------------------------------------

    def history(self, limit: int = 20) -> List[EngineeringRun]:
        runs = [_run_from_dict(d) for d in self.storage.all_runs()]
        runs.sort(key=lambda r: r.created_at, reverse=True)
        return runs[:limit]

    def lessons(self, limit: int = 50) -> List:
        from .models import EngineeringLesson
        lessons = [EngineeringLesson(**d) for d in self.storage.all_lessons()]
        lessons.sort(key=lambda l: l.created_at, reverse=True)
        return lessons[:limit]

    # -- SELF-IMPROVEMENT -----------------------------------------------------

    def self_improve(self, min_runs_for_pattern: int = 3) -> Optional[EngineeringProposal]:
        """
        Analyzes engineering history for recurring weaknesses (repeated
        planning/implementation failures, scope violations, weak recovery,
        etc.), and if a real pattern is found, produces a self-improvement
        EngineeringProposal for the *engineering agent's own process* --
        gated by the exact same propose/approve/implement pipeline as any
        other engineering task. This method NEVER modifies engineering_agent
        source itself; it only ever returns a proposal for a human to
        review and separately approve+implement via the normal pipeline.
        """
        runs = [_run_from_dict(d) for d in self.storage.all_runs()]
        if len(runs) < min_runs_for_pattern:
            return None

        weakness = self._detect_weakness(runs, min_runs_for_pattern)
        if weakness is None:
            return None

        goal_description = (
            f"Self-improvement: {weakness['description']} "
            f"(observed in {weakness['count']} of the last {len(runs)} engineering runs). "
            f"Evidence: {weakness['evidence']}"
        )
        return self.propose(goal_description, task_profile=TaskProfileName.SELF_IMPROVE.value)

    def _detect_weakness(self, runs: List[EngineeringRun], min_count: int) -> Optional[Dict]:
        scope_violation_runs = [r for r in runs if any("NOT in the approved plan" in f for f in r.failures)]
        if len(scope_violation_runs) >= min_count:
            return {
                "description": "The implementer repeatedly modifies files outside the approved plan scope",
                "count": len(scope_violation_runs),
                "evidence": f"run_ids={[r.run_id for r in scope_violation_runs][-5:]}",
            }

        recovery_failed_runs = [
            r for r in runs if r.recovery_attempts and not r.recovery_attempts[-1].succeeded
        ]
        if len(recovery_failed_runs) >= min_count:
            return {
                "description": "Failure recovery repeatedly exhausts its attempt budget without fixing the test",
                "count": len(recovery_failed_runs),
                "evidence": f"run_ids={[r.run_id for r in recovery_failed_runs][-5:]}",
            }

        planning_failed_runs = [r for r in runs if r.status == RunStatus.FAILED.value and r.plan is None]
        if len(planning_failed_runs) >= min_count:
            return {
                "description": "The planner repeatedly fails to produce a valid structured plan",
                "count": len(planning_failed_runs),
                "evidence": f"run_ids={[r.run_id for r in planning_failed_runs][-5:]}",
            }

        no_test_plans = [r for r in runs if r.plan and not r.plan.tests]
        if len(no_test_plans) >= min_count:
            return {
                "description": "Plans are repeatedly produced with no tests defined to validate the change",
                "count": len(no_test_plans),
                "evidence": f"run_ids={[r.run_id for r in no_test_plans][-5:]}",
            }

        return None


# ---------------------------------------------------------------------------
# dict -> dataclass reconstruction helpers (storage is plain JSON)
# ---------------------------------------------------------------------------

def _proposal_from_dict(data: Dict) -> EngineeringProposal:
    from .models import EngineeringGoal, EngineeringPlan, PlannedChange
    goal_data = data.get("goal") or {}
    plan_data = data.get("plan") or {}
    changes = [PlannedChange(**c) for c in plan_data.get("changes", [])]
    plan = EngineeringPlan(**{**plan_data, "changes": changes}) if plan_data else None
    goal = EngineeringGoal(**goal_data) if goal_data else None
    proposal = EngineeringProposal(**{**data, "goal": goal, "plan": plan})
    return proposal


def _run_from_dict(data: Dict) -> EngineeringRun:
    from .models import (
        EngineeringGoal, EngineeringPlan, EvaluationResult, ImplementationResult,
        PlannedChange, RecoveryAttempt, RepositoryContext, TestResult, FileEdit,
        HealthSnapshot, BenchmarkResult,
    )
    d = dict(data)

    goal_data = d.pop("goal", None) or {}
    goal = EngineeringGoal(**goal_data) if goal_data else None

    plan_data = d.pop("plan", None)
    plan = None
    if plan_data:
        changes = [PlannedChange(**c) for c in plan_data.get("changes", [])]
        plan = EngineeringPlan(**{**plan_data, "changes": changes})

    repo_data = d.pop("repository_context", None)
    repo_context = RepositoryContext(**repo_data) if repo_data else None

    impl_data = d.pop("implementation", None)
    implementation = None
    if impl_data:
        edits = [FileEdit(**e) for e in impl_data.get("edits", [])]
        implementation = ImplementationResult(**{**impl_data, "edits": edits})

    tests_data = d.pop("tests", [])
    tests = [TestResult(**t) for t in tests_data]

    recovery_data = d.pop("recovery_attempts", [])
    recovery_attempts = []
    for r in recovery_data:
        r = dict(r)
        edits = [FileEdit(**e) for e in r.get("edits", [])]
        tr = r.get("test_result")
        test_result = TestResult(**tr) if tr else None
        recovery_attempts.append(RecoveryAttempt(**{**r, "edits": edits, "test_result": test_result}))

    eval_data = d.pop("evaluation", None)
    evaluation = None
    if eval_data:
        benchmarks = [BenchmarkResult(**b) for b in eval_data.get("benchmarks", [])]
        health_before = [HealthSnapshot(**h) for h in eval_data.get("health_before", [])]
        health_after = [HealthSnapshot(**h) for h in eval_data.get("health_after", [])]
        evaluation = EvaluationResult(**{
            **eval_data, "benchmarks": benchmarks,
            "health_before": health_before, "health_after": health_after,
        })

    return EngineeringRun(
        **d, goal=goal, plan=plan, repository_context=repo_context,
        implementation=implementation, tests=tests,
        recovery_attempts=recovery_attempts, evaluation=evaluation,
    )
