"""
Evaluation Engineer / Benchmark Engineer.

Deliberately separates two questions:
  - CODE WORKS: did tests pass / did it execute without error?
  - IMPROVED JARVIS: is there actual evidence (benchmark deltas, health
    state, task success rate) that this made the affected system better?

`improved_jarvis` is left as None ("insufficient evidence") rather than
defaulting to True whenever there isn't a benchmark to compare against --
per the spec, the agent must never claim an improvement without evidence
when evidence is available, and must not manufacture evidence when it
isn't.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional

from .models import (
    BenchmarkResult, EngineeringPlan, EvaluationResult, HealthSnapshot,
    HealthState, TestResult,
)

# Signature JARVIS's real health system should satisfy if wired in:
#   health_checker(system_name: str) -> HealthState
HealthChecker = Callable[[str], str]

# Signature for a benchmark function: given a metric name, return its
# current numeric value (e.g. task success rate, routing accuracy).
BenchmarkFn = Callable[[str], Optional[float]]


class EvaluationEngineer:
    def __init__(self, health_checker: Optional[HealthChecker] = None,
                 benchmark_fn: Optional[BenchmarkFn] = None):
        # Both are optional injection points for JARVIS's real health and
        # benchmark systems. Without them, evaluation still works but is
        # explicitly honest about reduced evidence.
        self._health_checker = health_checker
        self._benchmark_fn = benchmark_fn

    def snapshot_health(self, systems: List[str]) -> List[HealthSnapshot]:
        snapshots = []
        for system in systems:
            if self._health_checker is None:
                snapshots.append(HealthSnapshot(
                    system=system, state=HealthState.UNKNOWN.value,
                    detail="No health checker wired in from JARVIS's health system.",
                ))
                continue
            try:
                state = self._health_checker(system)
                state_value = state.value if hasattr(state, "value") else str(state)
            except Exception as exc:  # noqa: BLE001 -- health checks must not crash evaluation
                state_value = HealthState.UNKNOWN.value
                snapshots.append(HealthSnapshot(system=system, state=state_value,
                                                 detail=f"Health check raised: {exc}"))
                continue
            snapshots.append(HealthSnapshot(system=system, state=state_value, detail=""))
        return snapshots

    def run_benchmarks(self, benchmark_names: List[str],
                        baseline: Optional[Dict[str, float]] = None) -> List[BenchmarkResult]:
        results = []
        for name in benchmark_names:
            baseline_value = (baseline or {}).get(name)
            post_value = self._benchmark_fn(name) if self._benchmark_fn else None
            improved = None
            if baseline_value is not None and post_value is not None:
                improved = post_value > baseline_value
            results.append(BenchmarkResult(
                name=name, baseline_value=baseline_value, post_change_value=post_value,
                improved=improved,
                notes="" if self._benchmark_fn else "No benchmark function wired in from JARVIS.",
            ))
        return results

    def capture_baseline(self, benchmark_names: List[str]) -> Dict[str, float]:
        """Call BEFORE implementation, per BASELINE -> CHANGE -> TEST -> BENCHMARK -> COMPARE."""
        if not self._benchmark_fn:
            return {}
        baseline = {}
        for name in benchmark_names:
            value = self._benchmark_fn(name)
            if value is not None:
                baseline[name] = value
        return baseline

    def evaluate(self, run_id: str, plan: EngineeringPlan, tests: List[TestResult],
                 health_before: List[HealthSnapshot], health_after: List[HealthSnapshot],
                 benchmarks: List[BenchmarkResult]) -> EvaluationResult:
        code_works = bool(tests) and all(t.passed for t in tests)

        evidence: List[str] = []
        if tests:
            evidence.append(f"{sum(t.passed for t in tests)}/{len(tests)} test(s) passed")

        regressed_health = [
            s for s in health_after
            if s.state in (HealthState.DEGRADED.value, HealthState.FAILING.value)
        ]
        if regressed_health:
            evidence.append(
                "Health regression detected in: " +
                ", ".join(s.system for s in regressed_health)
            )

        benchmark_improvements = [b for b in benchmarks if b.improved is True]
        benchmark_regressions = [b for b in benchmarks if b.improved is False]
        if benchmark_improvements:
            evidence.append(
                "Benchmarks improved: " + ", ".join(b.name for b in benchmark_improvements)
            )
        if benchmark_regressions:
            evidence.append(
                "Benchmarks regressed: " + ", ".join(b.name for b in benchmark_regressions)
            )

        improved_jarvis: Optional[bool]
        if not code_works:
            improved_jarvis = False
            evidence.append("Code does not pass its own tests -- cannot have improved JARVIS.")
        elif regressed_health or benchmark_regressions:
            improved_jarvis = False
        elif benchmark_improvements:
            improved_jarvis = True
        else:
            # Tests pass and nothing regressed, but there's no benchmark
            # evidence either way -- stay honest about that gap.
            improved_jarvis = None
            evidence.append(
                "No benchmark evidence available; 'code works' but improvement is unverified."
            )

        summary = self._summarize(code_works, improved_jarvis, evidence)

        return EvaluationResult(
            run_id=run_id,
            code_works=code_works,
            improved_jarvis=improved_jarvis,
            evidence=evidence,
            benchmarks=benchmarks,
            health_before=health_before,
            health_after=health_after,
            summary=summary,
        )

    def _summarize(self, code_works: bool, improved: Optional[bool], evidence: List[str]) -> str:
        works_str = "passes its tests" if code_works else "does NOT pass its tests"
        if improved is True:
            improved_str = "evidence supports an improvement"
        elif improved is False:
            improved_str = "evidence does NOT support an improvement (or shows regression)"
        else:
            improved_str = "insufficient evidence to say whether it improved JARVIS"
        return f"Code {works_str}; {improved_str}."
