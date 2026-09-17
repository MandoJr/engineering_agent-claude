# engineering_agent

A self-improving software-engineering agent designed to drop into an
existing JARVIS codebase as the engineering/organization "brain":
receives an engineering goal, understands the repo, plans a change,
gets explicit human approval, implements it, tests it, recovers from
failures within a bounded budget, evaluates whether it actually helped
(not just whether it ran), and writes a lesson either way.

```
GOAL -> ANALYZE -> PLAN -> PROPOSE -> USER APPROVAL -> IMPLEMENT
     -> TEST -> [RECOVER if needed] -> EVALUATE -> LEARN
```

## Install

No third-party dependencies -- pure standard library. Drop the
`engineering_agent/` package anywhere on JARVIS's Python path.

## Quickstart (standalone, uses local Ollama)

```python
from engineering_agent import EngineeringOrchestrator

agent = EngineeringOrchestrator(project_root="/path/to/jarvis")

proposal = agent.propose("Improve backend routing")
print(proposal.plan.to_dict())          # review before approving

agent.approve(proposal.proposal_id)     # <-- the only thing that unlocks implement()
result = agent.implement(proposal.proposal_id)

evaluation = agent.evaluate(result.run_id)
print(evaluation.evaluation.summary)

# Later, once there's enough history:
self_improvement_proposal = agent.self_improve()
```

By default this talks to a local Ollama server running
`qwen3-coder:30b` at `http://localhost:11434/api/chat`, matching
JARVIS's current setup, and uses a small in-package `SimpleRouter` that
mirrors (at a much smaller scale) the exploration/exploitation shape of
JARVIS's real router.

## Wiring into JARVIS for real

See `integration.py` for the seam. In short:

- **Routing** -- wrap JARVIS's real router with `ExistingRouterAdapter`
  (in `backend.py`) instead of using `SimpleRouter`, and register
  Gemini/Hermes as `CallableBackend`s pointing at your existing clients.
- **Health** -- pass a `health_checker(system_name) -> "HEALTHY"|"DEGRADED"|"FAILING"|"UNKNOWN"`
  callable into `EngineeringOrchestrator(health_checker=...)`.
- **Benchmarks** -- pass a `benchmark_fn(name) -> float | None` callable
  into `EngineeringOrchestrator(benchmark_fn=...)`.
- **Lessons/history storage** -- `storage.py` uses plain JSON files under
  `<project_root>/.engineering_agent/` by default. If JARVIS already has
  a lessons store, swap `EngineeringStorage` for an adapter with the
  same method surface (`save_run`, `save_proposal`, `save_lesson`,
  `all_runs`, `all_lessons`, `all_proposals`).

None of this requires importing JARVIS-internal modules from inside
`engineering_agent/` itself -- the package stays droppable and testable
standalone; you inject JARVIS's systems in from the outside.

## Hard safety boundaries (enforced in code, not just by prompt)

- `implement()` raises `ApprovalError` unless the proposal's status is
  `APPROVED` (or the task profile is configured to not require
  approval at all, e.g. pure `ANALYZE`/`REVIEW`). No parameter
  overrides this.
- `git_utils.GitWorkflow.commit()` / `.push()` independently require
  `approved=True` to be passed explicitly by the caller -- a second,
  separate gate from the proposal-approval check above.
- File tools (`tools.py`) refuse to touch anything outside
  `project_root`, anything under a configured forbidden path
  (`.git`, `.env`, `secrets/`, `node_modules/`, ...), or anything that
  looks like a credential file by name.
- `DELETE` planned changes are never auto-applied -- they're surfaced
  for manual handling.
- Failure recovery (`recovery.py`) is bounded by
  `config.MAX_RECOVERY_ATTEMPTS` (default 3) -- there is no retry loop
  without a ceiling.
- The evaluator (`evaluator.py`) reports `improved_jarvis = None`
  ("insufficient evidence") rather than assuming success whenever there
  is no benchmark to compare against -- it never claims an improvement
  it can't support.
- `self_improve()` only ever returns a *proposal*. It never edits
  `engineering_agent`'s own source. Getting that proposal implemented
  still goes through the exact same propose -> approve -> implement
  pipeline as any other engineering task.

## Package layout

| file | role |
|---|---|
| `models.py` | All data models (Goal, Plan, Proposal, Run, Lesson, ...) |
| `config.py` | Task profiles + tuning constants mirroring JARVIS's router concepts |
| `backend.py` | `ModelBackend` abstraction, `OllamaBackend`, router adapters |
| `tools.py` | Permission-gated file/git/shell tools (`READ_FILE`, `EDIT_FILE`, ...) |
| `git_utils.py` | Safe git workflow; approval-gated commit/push |
| `repository.py` | `RepositoryAnalyst` -- codebase understanding |
| `planner.py` | `CodingPlanner` -- goal + context -> structured plan |
| `reviewer.py` | `CodeReviewer` -- risk analysis, scope validation, proposal building |
| `implementer.py` | `CodingImplementer` -- applies an approved plan's changes |
| `tester.py` | `TestEngineer` -- runs a plan's tests |
| `recovery.py` | `FailureRecoveryEngineer` -- bounded retry loop |
| `evaluator.py` | `EvaluationEngineer` -- "code works" vs "improved JARVIS", health + benchmarks |
| `learning.py` | `EngineeringLearningSystem` -- lesson creation/retrieval |
| `storage.py` | JSON persistence for runs/proposals/lessons |
| `orchestrator.py` | `EngineeringOrchestrator` -- the public API + self-improvement |
| `cli.py` / `__main__.py` | `python -m engineering_agent ...` |
| `integration.py` | Documented wiring guide for JARVIS's real router/health/benchmarks |

## What's deliberately NOT here yet

Per the "strong foundation first" approach: deeper architecture
analysis, a full multi-specialist LLM ensemble (right now specialists
are focused modules/prompts sharing one routed backend call each,
exactly as the spec allows), symbol-level dependency graphs (repository
understanding is currently keyword + regex heuristics, good enough to
seed the planner but not a real language server), and a richer
benchmark suite are all meant to be layered on once the core loop is
proven out in your actual JARVIS environment.
