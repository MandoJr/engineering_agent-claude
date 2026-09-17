"""
Backend abstraction.

The engineering agent never picks a model directly. It expresses a
*requirement* (deep_reasoning, code_generation, debugging, code_review,
repository_analysis) and asks a router for a backend that satisfies it.
This is designed to be satisfied by JARVIS's existing
core/agency/agency_model_worker.py routing system -- see
`ExistingRouterAdapter` below, which is the seam you wire your real
router into. A dependency-free `SimpleRouter` fallback is provided so
this package works standalone (e.g. in tests, or before you've wired it
into JARVIS).

Ollama/Qwen3-Coder is implemented as a first-class backend, matching
JARVIS's current setup (http://localhost:11434/api/chat,
qwen3-coder:30b), and the interface stays generic so Gemini/Hermes
backends can be plugged in the same way.
"""

from __future__ import annotations

import abc
import json
import urllib.request
import urllib.error
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional


class BackendError(RuntimeError):
    pass


class ModelBackend(abc.ABC):
    """Common interface every backend (Ollama, Gemini, Hermes, ...) implements."""

    name: str = "unknown"
    capabilities: List[str] = []

    @abc.abstractmethod
    def complete(self, prompt: str, system: Optional[str] = None,
                 json_mode: bool = False, timeout: float = 120.0) -> str:
        """Return the raw text completion for `prompt`."""
        raise NotImplementedError


class OllamaBackend(ModelBackend):
    """Talks to a local Ollama server, matching JARVIS's current setup."""

    def __init__(self, endpoint: str = "http://localhost:11434/api/chat",
                 model: str = "qwen3-coder:30b"):
        self.endpoint = endpoint
        self.model = model
        self.name = f"ollama:{model}"
        # qwen3-coder is a coding-focused model -- advertise coding-relevant
        # capabilities so the router can match task profiles to it.
        self.capabilities = [
            "code_generation", "debugging", "code_review",
            "repository_analysis", "deep_reasoning",
        ]

    def complete(self, prompt: str, system: Optional[str] = None,
                 json_mode: bool = False, timeout: float = 120.0) -> str:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": False,
        }
        if json_mode:
            payload["format"] = "json"

        req = urllib.request.Request(
            self.endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                body = json.loads(resp.read().decode("utf-8"))
        except urllib.error.URLError as exc:
            raise BackendError(f"Ollama request failed: {exc}") from exc
        except json.JSONDecodeError as exc:
            raise BackendError(f"Ollama returned invalid JSON: {exc}") from exc

        message = body.get("message", {})
        content = message.get("content")
        if content is None:
            raise BackendError(f"Ollama response missing content: {body}")
        return content


class CallableBackend(ModelBackend):
    """Wraps an arbitrary callable(prompt, system, json_mode) -> str.

    Use this to adapt JARVIS's existing Gemini/Hermes clients without this
    package needing to import them directly -- just pass a bound method.
    """

    def __init__(self, name: str, fn: Callable[..., str], capabilities: List[str]):
        self.name = name
        self._fn = fn
        self.capabilities = capabilities

    def complete(self, prompt: str, system: Optional[str] = None,
                 json_mode: bool = False, timeout: float = 120.0) -> str:
        try:
            return self._fn(prompt, system=system, json_mode=json_mode, timeout=timeout)
        except BackendError:
            raise
        except Exception as exc:  # adapters must expose one stable failure type
            raise BackendError(f"Backend '{self.name}' failed: {exc}") from exc


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------

@dataclass
class RoutingDecision:
    backend: ModelBackend
    reason: str


class Router(abc.ABC):
    @abc.abstractmethod
    def select(self, task_profile_name: str,
               required_capabilities: List[str]) -> RoutingDecision:
        raise NotImplementedError

    @abc.abstractmethod
    def report_outcome(self, backend_name: str, task_profile_name: str,
                        success: bool, evidence: Optional[Dict[str, Any]] = None) -> None:
        raise NotImplementedError


class ExistingRouterAdapter(Router):
    """
    Adapter around JARVIS's real router (core/agency/agency_model_worker.py).

    Wire your existing router in like:

        from core.agency.agency_model_worker import router as jarvis_router
        adapter = ExistingRouterAdapter(
            select_fn=lambda profile, caps: jarvis_router.select_backend(
                task_type=profile, required_capabilities=caps),
            report_fn=jarvis_router.report_outcome,
            backend_lookup=my_backend_lookup,  # name -> ModelBackend
        )

    `select_fn` should return something with at least a `.name` (or a
    plain string backend name); this adapter normalizes either shape and
    resolves it to a concrete ModelBackend via `backend_lookup`.
    """

    def __init__(self, select_fn: Callable[[str, List[str]], Any],
                 report_fn: Callable[..., None],
                 backend_lookup: Dict[str, ModelBackend]):
        self._select_fn = select_fn
        self._report_fn = report_fn
        self._backends = backend_lookup

    def select(self, task_profile_name: str,
               required_capabilities: List[str]) -> RoutingDecision:
        raw = self._select_fn(task_profile_name, required_capabilities)
        backend_name = getattr(raw, "name", raw)
        backend = self._backends.get(backend_name)
        if backend is None:
            raise BackendError(
                f"JARVIS router selected unknown backend '{backend_name}'; "
                f"known backends: {list(self._backends)}"
            )
        return RoutingDecision(backend=backend, reason="jarvis_router")

    def report_outcome(self, backend_name: str, task_profile_name: str,
                        success: bool, evidence: Optional[Dict[str, Any]] = None) -> None:
        self._report_fn(
            backend_name=backend_name,
            task_type=task_profile_name,
            success=success,
            evidence=evidence or {},
        )


class SimpleRouter(Router):
    """
    Dependency-free fallback router used when no JARVIS router is wired in.

    Mirrors the *shape* of the existing JARVIS routing concepts
    (exploration interval, learning minimums, bounded learned bonus) at a
    small scale, purely so the package is runnable standalone. This is
    NOT meant to replace the real router -- prefer ExistingRouterAdapter
    whenever JARVIS's router is available.
    """

    def __init__(self, backends: List[ModelBackend],
                 exploration_interval: int = 10,
                 learning_min_executions: int = 5,
                 learned_bonus_cap: float = 0.20):
        if not backends:
            raise ValueError("SimpleRouter needs at least one backend")
        self._backends = {b.name: b for b in backends}
        self._exploration_interval = exploration_interval
        self._learning_min_executions = learning_min_executions
        self._learned_bonus_cap = learned_bonus_cap
        self._executions: Dict[str, int] = {name: 0 for name in self._backends}
        self._successes: Dict[str, int] = {name: 0 for name in self._backends}
        self._total_calls = 0

    def _capable(self, required_capabilities: List[str]) -> List[ModelBackend]:
        capable = [
            b for b in self._backends.values()
            if all(cap in b.capabilities for cap in required_capabilities)
        ]
        if not capable:
            raise BackendError(
                f"No backend satisfies {required_capabilities}; available="
                f"{ {name: backend.capabilities for name, backend in self._backends.items()} }"
            )
        return capable

    def select(self, task_profile_name: str,
               required_capabilities: List[str]) -> RoutingDecision:
        self._total_calls += 1
        candidates = self._capable(required_capabilities)

        # Periodic exploration: try the least-used capable backend.
        if self._total_calls % self._exploration_interval == 0:
            chosen = min(candidates, key=lambda b: self._executions[b.name])
            return RoutingDecision(chosen, reason="exploration")

        # Otherwise exploit the best observed success rate among backends
        # that have crossed the learning-minimum execution threshold;
        # fall back to the first capable backend (baseline) otherwise.
        learned = [
            b for b in candidates
            if self._executions[b.name] >= self._learning_min_executions
        ]
        if learned:
            def score(b: ModelBackend) -> float:
                execs = self._executions[b.name]
                rate = self._successes[b.name] / max(execs, 1)
                bonus = min(rate, self._learned_bonus_cap)
                return rate + bonus
            chosen = max(learned, key=score)
            return RoutingDecision(chosen, reason="learned")

        return RoutingDecision(candidates[0], reason="baseline")

    def report_outcome(self, backend_name: str, task_profile_name: str,
                        success: bool, evidence: Optional[Dict[str, Any]] = None) -> None:
        if backend_name not in self._executions:
            return
        self._executions[backend_name] += 1
        if success:
            self._successes[backend_name] += 1
