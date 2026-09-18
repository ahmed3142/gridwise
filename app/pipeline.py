"""End-to-end request pipeline:  LLM interpreter -> guardrails -> LP optimizer -> replay validator.

Infeasibility handling (a hard directive set that no schedule can satisfy): organizer scenarios are
guaranteed feasible, so infeasibility almost always means a misread note. We find the note(s) whose
removal restores feasibility, re-ask the LLM about exactly those notes with the reason as feedback,
and only if that fails fall back to a least-violation plan (flagged in plan_summary).
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field

from starlette.concurrency import run_in_threadpool

from .config import Settings
from .directives import Directive, interpretation_set_problems
from .interpreter.service import InterpretationService, NoteResult
from .optimizer.lp import build_problem, infeasibility_culprits, solve, solve_relaxed, statically_infeasible
from .optimizer.plan import build_hourly_plan, plan_summary, plan_totals
from .schemas import OptimizeRequest
from .validator import response_violations

log = logging.getLogger("gridwise.pipeline")


class ScenarioInfeasible(Exception):
    """Even the physical rules alone cannot be satisfied (should be rejected as HTTP 422)."""


@dataclass
class PipelineResult:
    body: dict
    degraded: bool = False
    timings_ms: dict[str, float] = field(default_factory=dict)
    notes: list[NoteResult] = field(default_factory=list)


async def run_pipeline(req: OptimizeRequest, service: InterpretationService, settings: Settings) -> PipelineResult:
    loop = asyncio.get_running_loop()
    deadline = loop.time() + settings.request_deadline_s
    timings: dict[str, float] = {}
    notes = list(req.operator_notes)

    t0 = time.perf_counter()
    results = await service.interpret(notes, req.battery, deadline)
    timings["interpret"] = (time.perf_counter() - t0) * 1000
    directives: list[Directive] = [r.directive for r in results]
    problems = interpretation_set_problems(directives, len(notes))
    if problems:  # pragma: no cover - service guarantees one directive per note in order
        raise RuntimeError(f"interpretation set invalid: {problems}")

    t1 = time.perf_counter()
    problem = build_problem(req.hours, req.battery, directives)
    solution = await run_in_threadpool(solve, problem)

    if solution is None and service.configured:
        culprits = await run_in_threadpool(infeasibility_culprits, req.hours, req.battery, directives)
        reasons = statically_infeasible(problem) or ["no schedule can satisfy the physical battery/grid limits together with it"]
        log.warning("directives infeasible; re-asking notes %s (%s)", culprits, reasons[:2])
        feedback = [
            "Applying your previous reading of this note makes the day's schedule impossible to satisfy: "
            + "; ".join(reasons[:2])
            + ". Check the hours, the number and its unit (kWh vs percent of capacity, remaining vs reduction)."
        ]
        remaining = deadline - loop.time()
        if remaining > 3.0:
            retried = await asyncio.gather(
                *(service.reinterpret(i, notes, req.battery, deadline, feedback) for i in culprits)
            )
            for r in retried:
                results[r.directive.note_index] = r
                directives[r.directive.note_index] = r.directive
            problem = build_problem(req.hours, req.battery, directives)
            solution = await run_in_threadpool(solve, problem)

    if solution is None:
        solution = await run_in_threadpool(solve_relaxed, problem)
        if solution is None:
            raise ScenarioInfeasible("the scenario is infeasible even without operator directives")
        log.error("serving least-violation plan: %s", solution.violations[:4])
    timings["optimize"] = (time.perf_counter() - t1) * 1000

    plan = build_hourly_plan(problem, solution)
    totals = plan_totals(plan, problem.tariff)
    body = {
        "scenario_id": req.scenario_id,
        "directive_interpretation": [d.to_interpretation() for d in directives],
        "hourly_plan": plan,
        **totals,
        "plan_summary": plan_summary(problem, plan, totals, directives, solution),
    }

    t2 = time.perf_counter()
    violations = response_violations(req, body, tol=1e-3)
    timings["validate"] = (time.perf_counter() - t2) * 1000
    if violations and not solution.relaxed:
        log.error("self-check found %d violation(s): %s", len(violations), violations[:5])
    return PipelineResult(
        body=body,
        degraded=any(r.degraded for r in results) or solution.relaxed,
        timings_ms=timings,
        notes=results,
    )
