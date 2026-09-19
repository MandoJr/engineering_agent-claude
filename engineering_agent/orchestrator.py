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
from .structured_implementer import StructuredCodingImplementer
from .learning import EngineeringLearningSystem
from .models import (
    ApprovalStatus, EngineeringGoal, EngineeringProposal, EngineeringRun,
    RiskLevel, RunStatus, TaskProfileName,
)
from .planner import CodingPlanner
from .structured_recovery import StructuredFailureRecoveryEngineer
from .repository import RepositoryAnalyst
from .reviewer import CodeReviewer
from .tester import TestEngineer
from .task_decomposer import TaskDecomposer
from .task_execution import TaskExecutionEngine
from .task_recovery import TaskRecoveryCoordinator
from .task_review import IndependentTaskGraphReviewer
from .task_verification import TaskVerificationEngine
from .task_graph import (
    TaskGraph,
    TaskStatus,
    UnknownTaskError,
)
from .execution_state import (
    ExecutionStateError,
    StateIntegrityError,
    TaskExecutionStateMachine,
)
from .task_resume import ResumeError, TaskResumeEngine
from .storage import EngineeringStorage, StorageError
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
        self._trace(run, "goal_classified", profile=goal.task_profile)

        backend = self._select_backend(goal.task_profile)
        run.backend = backend.name

        read_tools = ToolBox(self.config, permission=Permission.READ_ONLY)
        analyst = RepositoryAnalyst(read_tools, backend)
        repo_context = analyst.analyze(goal_description)
        run.repository_context = repo_context
        self._trace(run, "repository_analyzed", candidates=len(repo_context.candidate_files), risks=len(repo_context.risks))

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

        task_graph = TaskDecomposer.from_plan(plan)
        run.task_graph = task_graph.to_dict()
        run.state_transitions = []
        run.current_task_id = None

        self._trace(
            run,
            "task_graph_created",
            tasks=len(task_graph.tasks),
            ready_tasks=len(task_graph.ready_tasks()),
        )
        self._trace(run, "plan_created", changes=len(plan.changes), risk=plan.risk_level)
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

    def revise_plan(self, proposal_id: str, evidence: List[str]) -> EngineeringProposal:
        """Replace a proposal's plan with a bounded, evidence-driven revision.

        Revision never carries approval forward: new scope or risk requires a
        fresh human decision, even if the original proposal was approved.
        """
        proposal = self.review(proposal_id)
        if proposal.plan is None or proposal.goal is None:
            raise ValueError("Proposal has no goal/plan to revise")
        if proposal.plan.revision >= self.config.max_plan_revisions:
            raise ValueError("Plan revision budget exhausted; human intervention is required")
        run = self._find_run_for_proposal(proposal_id)
        if run is None or run.repository_context is None:
            raise ValueError("Proposal has no repository context to revise against")
        backend = self._select_backend(run.task_profile)
        lessons = self.learning.relevant_lessons(run.repository_context.affected_systems, run.task_profile)
        revised = CodingPlanner(backend).revise(proposal.goal, run.repository_context, proposal.plan, evidence, lessons)
        proposal.plan = revised
        proposal.affected_files = list(revised.files)
        proposal.risk = self.reviewer.review_plan(proposal.goal, revised)[1]
        proposal.status = ApprovalStatus.WAITING_FOR_APPROVAL.value
        proposal.decided_at = None
        proposal.decision_notes = "Plan revised from execution evidence; fresh approval required."
        run.plan = revised

        task_graph = TaskDecomposer.from_plan(revised)
        run.task_graph = task_graph.to_dict()
        run.state_transitions = []
        run.current_task_id = None

        run.approval = proposal.status
        self._trace(run, "plan_revised", revision=revised.revision, evidence_count=len(evidence))
        run.touch()
        self.storage.save_proposal(proposal)
        self.storage.save_run(run)
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
        if proposal.status not in (
            ApprovalStatus.APPROVED.value,
            ApprovalStatus.NOT_REQUIRED.value,
        ):
            raise ApprovalError(
                f"Refusing to implement proposal {proposal_id}: status is "
                f"'{proposal.status}', not APPROVED."
            )

        run = self._find_run_for_proposal(proposal_id)
        if run is None:
            raise KeyError(
                f"No engineering run found for proposal {proposal_id}"
            )

        profile = TASK_PROFILES[run.task_profile]
        plan = proposal.plan

        if plan is None:
            raise ValueError(
                f"Proposal {proposal_id} has no plan to implement"
            )

        # Capture baseline measurements exactly once. A resumed run must
        # not redefine its baseline after some tasks have already changed
        # the repository.
        if run.baseline_benchmarks:
            baseline = dict(run.baseline_benchmarks)
        else:
            baseline = self.evaluator.capture_baseline(plan.benchmarks)
            run.baseline_benchmarks = dict(baseline)

        if run.health_before:
            from .models import HealthSnapshot
            health_before = [
                HealthSnapshot(**item)
                for item in run.health_before
            ]
        else:
            health_before = self.evaluator.snapshot_health(
                plan.affected_systems
            )
            run.health_before = [
                snapshot.to_dict()
                for snapshot in health_before
            ]

        write_tools = ToolBox(
            self.config,
            permission=Permission.IMPLEMENT,
        )
        backend = self._select_backend(run.task_profile)

        implementer = StructuredCodingImplementer(
            backend,
            write_tools,
        )
        read_tools = ToolBox(
            self.config,
            permission=Permission.READ_ONLY,
        )
        tester = TestEngineer(read_tools)

        if run.task_graph:
            graph = TaskGraph.from_dict(run.task_graph)
        else:
            graph = TaskDecomposer.from_plan(plan)

        if run.state_transitions:
            state_machine = TaskExecutionStateMachine.from_dict(
                graph,
                {"transitions": run.state_transitions},
            )
        else:
            state_machine = TaskExecutionStateMachine(graph)

        graph.validate()
        state_machine.validate()

        run.task_graph = graph.to_dict()
        run.state_transitions = [
            transition.to_dict()
            for transition in state_machine.transitions
        ]
        run.current_task_id = None
        run.status = RunStatus.IMPLEMENTING.value
        self._trace(
            run,
            "task_execution_started",
            tasks=len(graph.tasks),
            backend=backend.name,
        )
        run.touch()
        self.storage.save_run(run)

        def checkpoint(task_id: str, task_status: TaskStatus) -> None:
            run.task_graph = graph.to_dict()
            run.state_transitions = [
                transition.to_dict()
                for transition in state_machine.transitions
            ]
            run.current_task_id = task_id

            if task_status == TaskStatus.RUNNING:
                run.status = RunStatus.IMPLEMENTING.value
            elif task_status == TaskStatus.VERIFYING:
                run.status = RunStatus.TESTING.value

            self._trace(
                run,
                "task_state_changed",
                task_id=task_id,
                task_status=task_status.value,
            )
            run.touch()
            self.storage.save_run(run)

        execution = TaskExecutionEngine(
            implementer,
            tester,
            checkpoint=checkpoint,
        )

        outcome = execution.execute(
            run.run_id,
            plan,
            graph,
            state_machine,
        )

        run.implementation = outcome.implementation
        run.tests = outcome.tests
        run.failures.extend(outcome.failures)

        run.task_graph = graph.to_dict()
        run.state_transitions = [
            transition.to_dict()
            for transition in state_machine.transitions
        ]
        run.current_task_id = None

        self._trace(
            run,
            "task_execution_finished",
            completed_tasks=outcome.completed_tasks,
            failed_tasks=outcome.failed_tasks,
            blocked_tasks=outcome.blocked_tasks,
        )

        # Per-task recovery coordinates bounded recovery across the whole DAG.
        if profile.requires_tests and outcome.failed_tasks:
            recovery_coordinator = TaskRecoveryCoordinator(
                backend=backend,
                tools=write_tools,
                tester=tester,
                execution_engine=execution,
                max_attempts=self.config.max_recovery_attempts,
                checkpoint=checkpoint,
            )

            recovery = recovery_coordinator.recover(
                run_id=run.run_id,
                plan=plan,
                graph=graph,
                state_machine=state_machine,
                initial_outcome=outcome,
            )

            run.recovery_attempts.extend(recovery.attempts)

            # Recovery patches are real implementation evidence and therefore
            # become part of the final independent-review input.
            recovery_edits = [
                edit
                for attempt in recovery.attempts
                for edit in attempt.edits
            ]
            recovery_patch_results = [
                patch
                for attempt in recovery.attempts
                for patch in attempt.patch_results
            ]

            if run.implementation is None:
                run.implementation = outcome.implementation

            run.implementation.edits.extend(recovery_edits)
            run.implementation.patch_results.extend(
                recovery_patch_results
            )

            # Preserve every verification result. Repeated executions of
            # the same command are distinct evidence, especially across
            # recovery and resume.
            run.tests.extend(recovery.tests)

            run.failures.extend(recovery.failures)

            self._trace(
                run,
                "task_recovery_coordinator_finished",
                recovered_tasks=recovery.recovered_tasks,
                failed_tasks=recovery.failed_tasks,
                blocked_tasks=recovery.blocked_tasks,
                attempts=len(recovery.attempts),
            )

        # Remaining independent tasks may still have completed, while tasks
        # that depend on a failed task remain BLOCKED.
        actual_diff = read_tools.git_diff()
        run.review = self.reviewer.review_implementation(
            plan,
            run.implementation.edits if run.implementation else [],
            run.tests,
            run.implementation.patch_results
            if run.implementation
            else [],
            actual_diff.data if actual_diff.ok else "",
        )

        graph_review = IndependentTaskGraphReviewer().review(
            graph=graph,
            plan=plan,
            edits=run.implementation.edits
            if run.implementation
            else [],
            tests=run.tests,
            patch_results=run.implementation.patch_results
            if run.implementation
            else [],
            git_diff=actual_diff.data if actual_diff.ok else "",
        )
        run.task_review = graph_review.to_dict()

        if not graph_review.passed:
            run.failures.extend(graph_review.findings)

        if run.review.findings:
            run.failures.extend(run.review.findings)

        self._trace(
            run,
            "independent_review_finished",
            passed=run.review.passed,
            findings=len(run.review.findings),
        )

        run.status = RunStatus.EVALUATING.value

        health_after = self.evaluator.snapshot_health(
            plan.affected_systems
        )
        benchmarks = self.evaluator.run_benchmarks(
            plan.benchmarks,
            baseline,
        )

        evaluation = self.evaluator.evaluate(
            run.run_id,
            plan,
            run.tests,
            health_before,
            health_after,
            benchmarks,
        )
        run.evaluation = evaluation

        graph_complete = all(
            task.status == TaskStatus.PASSED
            for task in graph.tasks.values()
        ) if graph.tasks else False

        overall_success = (
            graph_complete
            and evaluation.code_works
            and run.review.passed
        )

        run.status = (
            RunStatus.COMPLETE.value
            if overall_success
            else RunStatus.FAILED.value
        )
        run.final_result = evaluation.summary

        run.task_graph = graph.to_dict()
        run.state_transitions = [
            transition.to_dict()
            for transition in state_machine.transitions
        ]
        run.current_task_id = None

        lessons = self.learning.create_lessons_from_run(run)
        run.lessons_created = [
            lesson.lesson_id
            for lesson in lessons
        ]

        run.touch()
        self.storage.save_run(run)

        self.router.report_outcome(
            backend.name,
            run.task_profile,
            success=overall_success,
            evidence={
                "evaluation": evaluation.to_dict(),
                "completed_tasks": [
                    task_id
                    for task_id in graph.tasks
                    if graph.get(task_id).status == TaskStatus.PASSED
                ],
                "failed_tasks": [
                    task_id
                    for task_id in graph.tasks
                    if graph.get(task_id).status == TaskStatus.FAILED
                ],
                "blocked_tasks": [
                    task_id
                    for task_id in graph.tasks
                    if graph.get(task_id).status == TaskStatus.BLOCKED
                ],
            },
        )
        return run

    @staticmethod
    def _trace(run: EngineeringRun, event: str, **data) -> None:
        """Append an auditable, secret-free execution event."""
        import time
        run.trace.append({"event": event, "at": time.time(), "data": data})

    def _find_run_for_proposal(self, proposal_id: str) -> Optional[EngineeringRun]:
        for data in self.storage.all_runs():
            if data.get("proposal_id") == proposal_id:
                return _run_from_dict(data)
        return None

    def resume(self, run_id: str) -> EngineeringRun:
        """
        Safely resume an interrupted engineering run.

        Resumption never bypasses approval. Completed tasks remain terminal
        and are not re-executed. In-flight tasks are reconciled against the
        repository before any new implementation is generated.
        """
        try:
            run = self._get_run(run_id)
        except (
            StorageError,
            ValueError,
            TypeError,
            KeyError,
        ) as exc:
            raise ResumeError(
                f"Run {run_id} could not be loaded safely from "
                f"persistent storage: {exc}"
            ) from exc

        if run.proposal_id is None:
            raise ApprovalError(
                f"Run {run_id} has no approval-bearing proposal and cannot be resumed."
            )

        try:
            proposal = self.review(run.proposal_id)
        except (
            StorageError,
            ValueError,
            TypeError,
            KeyError,
        ) as exc:
            raise ResumeError(
                f"Run {run_id} references a proposal that could not "
                f"be loaded safely: {exc}"
            ) from exc

        if proposal.status not in (
            ApprovalStatus.APPROVED.value,
            ApprovalStatus.NOT_REQUIRED.value,
        ):
            raise ApprovalError(
                f"Refusing to resume run {run_id}: proposal status is "
                f"'{proposal.status}', not APPROVED."
            )

        if run.plan is None:
            raise ResumeError(
                f"Run {run_id} has no engineering plan."
            )

        if not run.task_graph:
            raise ResumeError(
                f"Run {run_id} has no persisted task graph."
            )

        try:
            graph = TaskGraph.from_dict(run.task_graph)

            if run.state_transitions:
                state_machine = TaskExecutionStateMachine.from_dict(
                    graph,
                    {"transitions": run.state_transitions},
                )
            else:
                state_machine = TaskExecutionStateMachine(graph)

            graph.validate()
            state_machine.validate()

        except (
            StateIntegrityError,
            ExecutionStateError,
            UnknownTaskError,
            ValueError,
            TypeError,
            KeyError,
        ) as exc:
            raise ResumeError(
                f"Run {run_id} contains inconsistent persisted "
                f"execution state and cannot be resumed safely: {exc}"
            ) from exc

        read_tools = ToolBox(
            self.config,
            permission=Permission.READ_ONLY,
        )
        tester = TestEngineer(read_tools)
        verifier = TaskVerificationEngine(tester)

        reconciliation = TaskResumeEngine(verifier).reconcile(
            graph,
            state_machine,
        )

        run.resume_count += 1
        run.interruption_reason = None
        run.task_graph = graph.to_dict()
        run.state_transitions = [
            transition.to_dict()
            for transition in state_machine.transitions
        ]
        self._trace(
            run,
            "run_resume_reconciled",
            reconciled_passed=reconciliation.reconciled_passed,
            reconciled_failed=reconciliation.reconciled_failed,
        )
        run.touch()
        self.storage.save_run(run)

        # Reuse the normal implementation pipeline. It executes only READY
        # tasks, so PASSED tasks are naturally skipped.
        resumed = self.implement(run.proposal_id)

        resumed.resume_count = run.resume_count
        resumed.touch()
        self.storage.save_run(resumed)
        return resumed

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
        PlannedChange, RecoveryAttempt, RepositoryContext, TestResult, FileEdit, PatchResult,
        HealthSnapshot, BenchmarkResult,
        ReviewResult,
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
        patch_results = [PatchResult(**p) for p in impl_data.get("patch_results", [])]
        implementation = ImplementationResult(**{**impl_data, "edits": edits, "patch_results": patch_results})

    tests_data = d.pop("tests", [])
    tests = [TestResult(**t) for t in tests_data]

    recovery_data = d.pop("recovery_attempts", [])
    recovery_attempts = []
    for r in recovery_data:
        r = dict(r)
        edits = [FileEdit(**e) for e in r.get("edits", [])]
        patch_results = [
            PatchResult(**p)
            for p in r.get("patch_results", [])
        ]
        tr = r.get("test_result")
        test_result = TestResult(**tr) if tr else None
        recovery_attempts.append(
            RecoveryAttempt(
                **{
                    **r,
                    "edits": edits,
                    "patch_results": patch_results,
                    "test_result": test_result,
                }
            )
        )

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

    review_data = d.pop("review", None)
    review = ReviewResult(**review_data) if review_data else None

    return EngineeringRun(
        **d, goal=goal, plan=plan, repository_context=repo_context,
        implementation=implementation, tests=tests,
        recovery_attempts=recovery_attempts, evaluation=evaluation, review=review,
    )

