"""
Integration seams for wiring this package into an existing JARVIS.

This file is intentionally *not* imported by the rest of the package --
it's a documented reference for how to construct an EngineeringOrchestrator
that uses JARVIS's real routing/health/lessons systems instead of the
dependency-free fallbacks. Copy/adapt the function below into your
JARVIS startup code once the real import paths are known.

Nothing in engineering_agent imports core.agency.agency_model_worker (or
any other JARVIS-internal module) directly -- this keeps the package
droppable into the codebase without immediate hard coupling, and lets it
run standalone (e.g. under tests) before the wiring is done.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .backend import CallableBackend, ExistingRouterAdapter, OllamaBackend
from .config import AgentConfig
from .orchestrator import EngineeringOrchestrator


def build_orchestrator_wired_into_jarvis(project_root: str) -> EngineeringOrchestrator:
    """
    Example wiring. Uncomment and adapt the imports once you're ready to
    connect this to the real JARVIS modules -- left as plain comments
    (not conditional imports) so this file has zero import-time
    dependency on your JARVIS internals until you actively edit it in.
    """
    # from core.agency.agency_model_worker import router as jarvis_router
    # from core.agency.agency_model_worker import gemini_client, hermes_client
    # from core.health import health_system
    # from core.benchmarks import benchmark_system

    config = AgentConfig(project_root=Path(project_root))

    ollama = OllamaBackend(endpoint=config.ollama_endpoint, model=config.ollama_model)

    # gemini = CallableBackend(
    #     name="gemini", capabilities=["deep_reasoning", "code_review", "repository_analysis"],
    #     fn=lambda prompt, system=None, json_mode=False, timeout=120.0:
    #         gemini_client.complete(prompt, system_prompt=system, response_format="json" if json_mode else "text"),
    # )
    # hermes = CallableBackend(
    #     name="hermes", capabilities=["code_generation", "debugging"],
    #     fn=lambda prompt, system=None, json_mode=False, timeout=120.0:
    #         hermes_client.complete(prompt, system=system),
    # )

    backend_lookup = {ollama.name: ollama}
    # backend_lookup.update({gemini.name: gemini, hermes.name: hermes})

    # router = ExistingRouterAdapter(
    #     select_fn=lambda profile, caps: jarvis_router.select_backend(
    #         task_type=profile, required_capabilities=caps),
    #     report_fn=jarvis_router.report_outcome,
    #     backend_lookup=backend_lookup,
    # )
    router = None  # falls back to SimpleRouter(ollama) until the above is wired in

    # def health_checker(system_name: str) -> str:
    #     return health_system.check(system_name).value  # HEALTHY/DEGRADED/FAILING/UNKNOWN
    health_checker = None

    # def benchmark_fn(name: str):
    #     return benchmark_system.current_value(name)
    benchmark_fn = None

    return EngineeringOrchestrator(
        project_root=project_root,
        router=router,
        health_checker=health_checker,
        benchmark_fn=benchmark_fn,
        config=config,
    )
