from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any


PLAN_TYPE = "WAN22_SAMPLING_PLAN"
PLAN_VERSION = 3

TASK_PROFILES = {
    "T2V": {
        "boundary": 0.875,
        "native_shift": 12.0,
    },
    "I2V": {
        "boundary": 0.900,
        "native_shift": 5.0,
    },
}
TASK_OPTIONS = list(TASK_PROFILES)

ACCELERATION_NONE = "None"
ACCELERATION_HIGH = "High only"
ACCELERATION_LOW = "Low only"
ACCELERATION_BOTH = "High + Low"
ACCELERATION_OPTIONS = [
    ACCELERATION_BOTH,
    ACCELERATION_HIGH,
    ACCELERATION_LOW,
    ACCELERATION_NONE,
]

PRIORITY_BALANCED = "Balanced"
PRIORITY_EVEN = "50/50 Split"
PRIORITY_MOTION = "Motion / Structure"
PRIORITY_DETAIL = "Detail / Refinement"
PRIORITY_OPTIONS = [
    PRIORITY_BALANCED,
    PRIORITY_EVEN,
    PRIORITY_MOTION,
    PRIORITY_DETAIL,
]

CURVE_PROFILE_NATIVE = "wan_native"
CURVE_PROFILE_DISTILLED = "lightx2v_distilled"
CURVE_PROFILE_COMFYUI = "comfyui_yaw"

SIGMA_BUDGET_PROJECTED = "projected"
SIGMA_BUDGET_ACCELERATED_50_50 = "accelerated_50_50"

DENSE_REFERENCE_STEPS = 256
MAX_REFERENCE_STEPS = 4096
MIN_SHIFT = 0.01
MAX_SHIFT = 100.0
SIGMA_TOLERANCE = 1e-8


SigmaProvider = Callable[[float, int, str], Sequence[Any]]


@dataclass(frozen=True)
class ShiftSearchResult:
    """Compatibility result for the retired exact-boundary shift search."""

    shift: float
    sigmas: Sequence[Any]
    boundary_error: float
    shift_sensitive: bool


def _as_float(value: Any) -> float:
    try:
        return float(value.item())
    except (AttributeError, TypeError):
        return float(value)


def _sigma_values(sigmas: Sequence[Any]) -> list[float]:
    try:
        return [_as_float(value) for value in sigmas]
    except TypeError as error:
        raise ValueError("The scheduler did not return a sigma sequence.") from error


def _sequence_like(template: Sequence[Any], values: Sequence[float]) -> Sequence[Any]:
    new_tensor = getattr(template, "new_tensor", None)
    if callable(new_tensor):
        return new_tensor(values)

    if isinstance(template, tuple):
        return tuple(values)
    if isinstance(template, list):
        try:
            return type(template)(values)
        except TypeError:
            return list(values)

    try:
        return type(template)(values)
    except (TypeError, ValueError):
        return list(values)


def _validated_sigmas(
    sigmas: Sequence[Any],
    *,
    context: str,
    require_terminal_zero: bool = True,
) -> tuple[Sequence[Any], list[float]]:
    values = _sigma_values(sigmas)
    if len(values) < 2:
        raise ValueError(f"{context} returned fewer than two sigma values.")

    for index, value in enumerate(values):
        if not math.isfinite(value):
            raise ValueError(f"{context} returned a non-finite sigma at index {index}.")
        if value < 0.0:
            raise ValueError(f"{context} returned a negative sigma at index {index}.")

    for index, (previous, current) in enumerate(zip(values, values[1:]), start=1):
        if current >= previous:
            description = "duplicate" if current == previous else "increasing"
            raise ValueError(
                f"{context} returned a {description} sigma at index {index}; "
                "Wan 2.2 plans require a strictly descending schedule."
            )

    if require_terminal_zero and not math.isclose(
        values[-1],
        0.0,
        abs_tol=SIGMA_TOLERANCE,
    ):
        raise ValueError(f"{context} must end at sigma zero.")

    return sigmas, values


def _provided_sigmas(
    *,
    sigma_provider: SigmaProvider,
    shift: float,
    steps: int,
    scheduler: str,
    context: str,
) -> tuple[Sequence[Any], list[float]]:
    try:
        sigmas = sigma_provider(float(shift), int(steps), scheduler)
    except Exception as error:
        raise ValueError(f"{context} failed: {error}") from error
    return _validated_sigmas(sigmas, context=context)


def _crossing_position(values: Sequence[float], boundary: float) -> float:
    """Return the fractional transition index where the curve crosses boundary."""
    for index in range(1, len(values)):
        previous = values[index - 1]
        current = values[index]
        if previous >= boundary and current < boundary:
            span = previous - current
            if span <= 0.0:
                break
            return (index - 1) + (previous - boundary) / span
    raise ValueError(
        f"The scheduler curve does not cross the Wan expert boundary σ={boundary:.3f}."
    )


def boundary_crossing_step(
    sigmas: Sequence[Any],
    boundary: float,
    steps: int | None = None,
) -> int:
    """Return the first sigma index below the expert boundary."""
    values = _sigma_values(sigmas)
    limit = len(values) - 1 if steps is None else min(int(steps), len(values) - 1)
    for index in range(1, limit + 1):
        if values[index - 1] >= boundary and values[index] < boundary:
            return index
    raise ValueError(
        f"The scheduler curve does not cross the Wan expert boundary σ={boundary:.3f}."
    )


def _profile_for(
    *,
    task: str,
    acceleration: str,
    accelerated_steps: int,
) -> tuple[str, float]:
    if acceleration == ACCELERATION_NONE:
        return CURVE_PROFILE_NATIVE, float(TASK_PROFILES[task]["native_shift"])
    if accelerated_steps <= 4:
        return CURVE_PROFILE_DISTILLED, 5.0
    return CURVE_PROFILE_COMFYUI, 8.0


def priority_range_share(
    *,
    native_share: float,
    priority: str,
) -> float:
    """Choose the high-noise share of the full denoising range."""
    if priority == PRIORITY_EVEN:
        return 0.5
    if priority == PRIORITY_BALANCED:
        return native_share
    if priority == PRIORITY_MOTION:
        return min(0.95, native_share + 0.10)
    if priority == PRIORITY_DETAIL:
        return max(0.05, native_share - 0.10)
    raise ValueError(f"Unknown Wan 2.2 priority: {priority}")


def _rounded_stage_share(budget: int, share: float) -> int:
    if math.isclose(share, 0.5, abs_tol=1e-12):
        result = budget // 2
    else:
        result = math.floor(budget * share + 0.5)
    return max(1, min(result, budget - 1))


def project_stage_steps(
    *,
    accelerated_steps: int,
    full_steps: int,
    acceleration: str,
    high_range_share: float,
) -> tuple[int, int, int, int]:
    """Project each active stage's full-range-equivalent budget independently."""
    accelerate_high = acceleration in (ACCELERATION_HIGH, ACCELERATION_BOTH)
    accelerate_low = acceleration in (ACCELERATION_LOW, ACCELERATION_BOTH)
    high_budget = accelerated_steps if accelerate_high else full_steps
    low_budget = accelerated_steps if accelerate_low else full_steps

    steps_high = _rounded_stage_share(high_budget, high_range_share)
    if high_budget == low_budget:
        steps_low = high_budget - steps_high
    else:
        steps_low = _rounded_stage_share(low_budget, 1.0 - high_range_share)

    return steps_high, steps_low, high_budget, low_budget


def _interpolate(values: Sequence[float], position: float) -> float:
    last = len(values) - 1
    if position <= 0.0:
        return values[0]
    if position >= last:
        return values[-1]
    lower = math.floor(position)
    fraction = position - lower
    return values[lower] + (values[lower + 1] - values[lower]) * fraction


def _piecewise_curve(
    *,
    template: Sequence[Any],
    reference_values: Sequence[float],
    boundary: float,
    steps_high: int,
    steps_low: int,
) -> Sequence[Any]:
    crossing = _crossing_position(reference_values, boundary)
    transitions = len(reference_values) - 1
    epsilon = max(transitions * 1e-7, 1e-7)
    handoff = min(crossing + epsilon, transitions - epsilon)
    if handoff <= 0.0 or handoff >= transitions:
        raise ValueError("The scheduler curve leaves no usable range around the boundary.")

    high_values = [
        _interpolate(reference_values, handoff * index / steps_high)
        for index in range(steps_high + 1)
    ]
    low_values = [
        _interpolate(
            reference_values,
            handoff + (transitions - handoff) * index / steps_low,
        )
        for index in range(steps_low + 1)
    ]
    values = high_values + low_values[1:]
    return _sequence_like(template, values)


def _reference_steps(requested_steps: int) -> int:
    return min(
        MAX_REFERENCE_STEPS,
        max(DENSE_REFERENCE_STEPS, requested_steps * 8),
    )


def _shift_candidates(profile: str, anchor: float) -> list[float]:
    bounds = {
        CURVE_PROFILE_NATIVE: (
            max(MIN_SHIFT, anchor * 0.67),
            min(MAX_SHIFT, anchor * 1.50),
        ),
        CURVE_PROFILE_DISTILLED: (3.0, 8.0),
        CURVE_PROFILE_COMFYUI: (5.0, 12.0),
    }
    lower, upper = bounds[profile]
    candidates = [anchor]
    for index in range(1, 17):
        fraction = index / 16.0
        candidates.extend(
            [
                anchor - (anchor - lower) * fraction,
                anchor + (upper - anchor) * fraction,
            ]
        )
    return list(dict.fromkeys(float(value) for value in candidates))


def _reference_for_shift(
    *,
    sigma_provider: SigmaProvider,
    scheduler: str,
    requested_steps: int,
    boundary: float,
    profile: str,
    shift: float,
    allow_adjustment: bool,
) -> tuple[float, Sequence[Any], list[float], str]:
    reference_steps = _reference_steps(requested_steps)
    sigmas, values = _provided_sigmas(
        sigma_provider=sigma_provider,
        shift=shift,
        steps=reference_steps,
        scheduler=scheduler,
        context=f"Scheduler '{scheduler}' reference curve at shift {shift:g}",
    )
    try:
        _crossing_position(values, boundary)
    except ValueError as error:
        if not allow_adjustment:
            raise
        anchor_error_text = str(error)
    else:
        return shift, sigmas, values, "profile"

    errors = [anchor_error_text]
    for candidate in _shift_candidates(profile, shift)[1:]:
        try:
            sigmas, values = _provided_sigmas(
                sigma_provider=sigma_provider,
                shift=candidate,
                steps=reference_steps,
                scheduler=scheduler,
                context=f"Scheduler '{scheduler}' reference curve at shift {candidate:g}",
            )
            _crossing_position(values, boundary)
        except ValueError as error:
            errors.append(str(error))
            continue
        return candidate, sigmas, values, "profile_adjusted"

    detail = errors[0] if errors else "No valid scheduler curve was returned."
    raise ValueError(
        "Unable to build a boundary-safe Wan 2.2 sigma curve. " + detail
    )


def optimize_shift(
    *,
    sigma_provider: SigmaProvider,
    scheduler: str,
    steps: int,
    steps_high: int,
    boundary: float,
    initial_shift: float,
) -> ShiftSearchResult:
    """Compatibility helper that preserves the supplied profile shift.

    Planner v3 deliberately does not solve for a sigma exactly equal to the
    expert boundary. New code should call build_plan and inspect shift_source.
    """
    sigmas, values = _provided_sigmas(
        sigma_provider=sigma_provider,
        shift=initial_shift,
        steps=steps,
        scheduler=scheduler,
        context=f"Scheduler '{scheduler}' curve at shift {initial_shift:g}",
    )
    split = max(1, min(int(steps_high), len(values) - 1))
    before = values[split - 1]
    after = values[split]
    return ShiftSearchResult(
        shift=float(initial_shift),
        sigmas=sigmas,
        boundary_error=min(abs(before - boundary), abs(after - boundary)),
        shift_sensitive=False,
    )


def _warnings_for_plan(
    *,
    acceleration: str,
    sigma_budget_mode: str,
    steps: int,
    steps_high: int,
    steps_low: int,
    priority: str,
    accelerated_steps: int,
    full_steps: int,
    shift_source: str,
    legacy_override_ignored: bool,
    scheduler_requested_steps: int,
    scheduler_actual_steps: int,
) -> list[str]:
    warnings: list[str] = []
    accelerate_high = acceleration in (ACCELERATION_HIGH, ACCELERATION_BOTH)
    accelerate_low = acceleration in (ACCELERATION_LOW, ACCELERATION_BOTH)

    if (accelerate_high or accelerate_low) and accelerated_steps > 4:
        warnings.append(
            "Extended acceleration profile: use a matched high/low acceleration "
            "model pair and begin at the vendor's documented strength. LightX2V "
            "4-step LoRAs are documented at strength 1.0; higher strengths or "
            "mismatched full-model/LoRA files can become unstable."
        )
    if accelerate_high and steps_high > 6:
        warnings.append(
            "Higher accelerated high-noise step counts may require lower LoRA strength."
        )
    if accelerate_low and steps_low > 6:
        warnings.append(
            "Higher accelerated low-noise step counts may require lower LoRA strength."
        )
    if not accelerate_high and not accelerate_low and steps < 8:
        warnings.append("Very low base-model step counts may reduce quality.")
    elif not accelerate_high and steps_high < 3:
        warnings.append("The unaccelerated high-noise stage has very few steps.")
    elif not accelerate_low and steps_low < 3:
        warnings.append("The unaccelerated low-noise stage has very few steps.")

    if sigma_budget_mode == SIGMA_BUDGET_ACCELERATED_50_50:
        warnings.append(
            "Progressive 50/50 Sigma Control is active: model and CFG routing "
            "remain Low only, but the sigma curve uses the accelerated budget "
            "split 50/50."
        )

    selected_budget = (
        accelerated_steps
        if sigma_budget_mode == SIGMA_BUDGET_ACCELERATED_50_50
        else
        accelerated_steps
        if acceleration == ACCELERATION_BOTH
        else full_steps
        if acceleration == ACCELERATION_NONE
        else None
    )
    if priority == PRIORITY_EVEN and selected_budget is not None and selected_budget % 2:
        warnings.append(
            "An odd full-range budget cannot split exactly 50/50; the extra step is "
            "assigned to the low-noise stage."
        )
    if shift_source == "profile_adjusted":
        warnings.append(
            "The profile shift was adjusted within its safe range because the "
            "scheduler did not cross the Wan expert boundary at the anchor shift."
        )
    if legacy_override_ignored:
        warnings.append(
            "The legacy range percentage was ignored because an exact high-step "
            "override is active."
        )
    if scheduler_actual_steps != scheduler_requested_steps:
        warnings.append(
            f"The scheduler returned {scheduler_actual_steps} transitions for a "
            f"{scheduler_requested_steps}-step request; the authoritative sigma "
            "curve was resampled to the planned count."
        )
    return warnings


def _summary(plan: dict[str, Any]) -> str:
    warning_text = ""
    if plan["warnings"]:
        warning_text = " | " + " ".join(plan["warnings"])
    budget_mode_text = ""
    if plan.get("sigma_budget_mode") == SIGMA_BUDGET_ACCELERATED_50_50:
        budget_mode_text = " · accelerated 50/50 sigmas"
    return (
        f"{plan['task']} · {plan['acceleration']} · {plan['priority']} | "
        f"{plan['steps']} steps = {plan['steps_high']} high + "
        f"{plan['steps_low']} low | budgets A{plan['accelerated_steps']}/"
        f"F{plan['full_steps']}{budget_mode_text} · "
        f"range {plan['high_range_percent']:.1f}% high | "
        f"{plan['curve_profile']} / {plan['curve_mode']} · "
        f"shift {plan['shift']:.4f} ({plan['shift_source']}) | "
        f"boundary σ={plan['boundary']:.3f}, handoff σ={plan['split_sigma']:.4f}"
        f"{warning_text}"
    )


def build_plan(
    *,
    model: Any,
    task: str,
    acceleration: str,
    accelerated_steps: int,
    full_steps: int,
    scheduler: str,
    priority: str,
    sigma_provider: SigmaProvider,
    forced_steps_high: int | None = None,
    forced_shift: float | None = None,
    forced_high_range_percent: float | None = None,
    forced_sigma_budget_mode: str | None = None,
) -> dict[str, Any]:
    if task not in TASK_PROFILES:
        raise ValueError(f"Unknown Wan 2.2 task: {task}")
    if acceleration not in ACCELERATION_OPTIONS:
        raise ValueError(f"Unknown Wan 2.2 acceleration mode: {acceleration}")
    if priority not in PRIORITY_OPTIONS:
        raise ValueError(f"Unknown Wan 2.2 priority: {priority}")
    if forced_sigma_budget_mode is not None and (
        forced_sigma_budget_mode != SIGMA_BUDGET_ACCELERATED_50_50
    ):
        raise ValueError(
            f"Unknown Wan 2.2 sigma budget override: {forced_sigma_budget_mode}"
        )
    if (
        forced_sigma_budget_mode == SIGMA_BUDGET_ACCELERATED_50_50
        and acceleration != ACCELERATION_LOW
    ):
        raise ValueError(
            "Progressive 50/50 Sigma Control only applies when acceleration "
            "is Low only. It preserves base high / accelerated low model routing."
        )

    accelerated_steps = int(accelerated_steps)
    full_steps = int(full_steps)
    if accelerated_steps < 2:
        raise ValueError("accelerated_steps must be at least 2.")
    if full_steps < 2:
        raise ValueError("full_steps must be at least 2.")
    if forced_steps_high is not None:
        forced_steps_high = int(forced_steps_high)
        if forced_steps_high == 0:
            forced_steps_high = None
        elif forced_steps_high < 0:
            raise ValueError("forced_steps_high must be zero or a positive integer.")

    profile_data = TASK_PROFILES[task]
    boundary = float(profile_data["boundary"])
    native_shift = float(profile_data["native_shift"])
    curve_profile, shift_anchor = _profile_for(
        task=task,
        acceleration=acceleration,
        accelerated_steps=accelerated_steps,
    )

    if forced_shift is None:
        requested_shift = shift_anchor
        shift, reference, reference_values, shift_source = _reference_for_shift(
            sigma_provider=sigma_provider,
            scheduler=scheduler,
            requested_steps=max(accelerated_steps, full_steps),
            boundary=boundary,
            profile=curve_profile,
            shift=requested_shift,
            allow_adjustment=True,
        )
    else:
        requested_shift = float(forced_shift)
        if (
            not math.isfinite(requested_shift)
            or requested_shift < MIN_SHIFT
            or requested_shift > MAX_SHIFT
        ):
            raise ValueError(
                f"forced_shift must be between {MIN_SHIFT:g} and {MAX_SHIFT:g}."
            )
        shift, reference, reference_values, _ = _reference_for_shift(
            sigma_provider=sigma_provider,
            scheduler=scheduler,
            requested_steps=max(accelerated_steps, full_steps),
            boundary=boundary,
            profile=curve_profile,
            shift=requested_shift,
            allow_adjustment=False,
        )
        shift_source = "override"

    if curve_profile == CURVE_PROFILE_DISTILLED:
        profile_share = 0.5
    else:
        profile_share = _crossing_position(reference_values, boundary) / (
            len(reference_values) - 1
        )

    legacy_override_ignored = (
        forced_steps_high is not None and forced_high_range_percent is not None
    ) or (
        forced_sigma_budget_mode == SIGMA_BUDGET_ACCELERATED_50_50
        and forced_high_range_percent is not None
        and forced_steps_high is None
    )
    if forced_high_range_percent is not None and forced_steps_high is None:
        forced_high_range_percent = float(forced_high_range_percent)
        if not 1.0 <= forced_high_range_percent <= 99.0:
            raise ValueError(
                "forced_high_range_percent must be between 1 and 99; "
                f"got {forced_high_range_percent}."
            )
        high_range_share = forced_high_range_percent / 100.0
    else:
        high_range_share = priority_range_share(
            native_share=profile_share,
            priority=priority,
        )

    steps_high, steps_low, high_budget, low_budget = project_stage_steps(
        accelerated_steps=accelerated_steps,
        full_steps=full_steps,
        acceleration=acceleration,
        high_range_share=high_range_share,
    )
    projected_steps_high = steps_high
    projected_steps_low = steps_low
    sigma_budget_mode = SIGMA_BUDGET_PROJECTED

    if forced_sigma_budget_mode == SIGMA_BUDGET_ACCELERATED_50_50:
        sigma_budget_mode = SIGMA_BUDGET_ACCELERATED_50_50
        high_range_share = 0.5
        high_budget = accelerated_steps
        low_budget = accelerated_steps
        steps_high = _rounded_stage_share(accelerated_steps, high_range_share)
        steps_low = accelerated_steps - steps_high

    if forced_steps_high is not None:
        projected_total = steps_high + steps_low
        if not 1 <= forced_steps_high < projected_total:
            raise ValueError(
                "forced_steps_high must leave at least one low-noise transition; "
                f"expected 1..{projected_total - 1}, got {forced_steps_high}."
            )
        steps_high = forced_steps_high
        steps_low = projected_total - steps_high

    steps = steps_high + steps_low
    exact_sigmas, exact_values = _provided_sigmas(
        sigma_provider=sigma_provider,
        shift=shift,
        steps=steps,
        scheduler=scheduler,
        context=f"Scheduler '{scheduler}' exact curve at shift {shift:g}",
    )
    scheduler_actual_steps = len(exact_values) - 1
    exact_crossing = boundary_crossing_step(exact_values, boundary)
    if len(exact_values) == steps + 1 and exact_crossing == steps_high:
        sigmas = exact_sigmas
        curve_mode = "exact"
    else:
        sigmas = _piecewise_curve(
            template=reference,
            reference_values=reference_values,
            boundary=boundary,
            steps_high=steps_high,
            steps_low=steps_low,
        )
        curve_mode = "piecewise"

    sigmas, sigma_values = _validated_sigmas(
        sigmas,
        context="Generated Wan 2.2 sigma curve",
    )
    sigmas_high = sigmas[: steps_high + 1]
    sigmas_low = sigmas[steps_high:]
    boundary_sigma_before = sigma_values[steps_high - 1]
    boundary_sigma_after = sigma_values[steps_high]
    boundary_distance = min(
        abs(boundary_sigma_before - boundary),
        abs(boundary_sigma_after - boundary),
    )

    overrides: dict[str, int | float] = {}
    if forced_steps_high is not None:
        overrides["steps_high"] = forced_steps_high
    if forced_shift is not None:
        overrides["shift"] = float(forced_shift)
    if forced_sigma_budget_mode == SIGMA_BUDGET_ACCELERATED_50_50:
        overrides["sigma_budget_mode"] = SIGMA_BUDGET_ACCELERATED_50_50
    if forced_high_range_percent is not None and forced_steps_high is None:
        overrides["high_range_percent"] = float(forced_high_range_percent)

    warnings = _warnings_for_plan(
        acceleration=acceleration,
        sigma_budget_mode=sigma_budget_mode,
        steps=steps,
        steps_high=steps_high,
        steps_low=steps_low,
        priority=priority,
        accelerated_steps=accelerated_steps,
        full_steps=full_steps,
        shift_source=shift_source,
        legacy_override_ignored=legacy_override_ignored,
        scheduler_requested_steps=steps,
        scheduler_actual_steps=scheduler_actual_steps,
    )
    source_controls = {
        "model": model,
        "task": task,
        "acceleration": acceleration,
        "accelerated_steps": accelerated_steps,
        "full_steps": full_steps,
        "scheduler": scheduler,
        "priority": priority,
        "sigma_provider": sigma_provider,
    }
    plan: dict[str, Any] = {
        "type": PLAN_TYPE,
        "version": PLAN_VERSION,
        "model": model,
        "source_controls": source_controls,
        "overrides": overrides,
        "task": task,
        "acceleration": acceleration,
        "accelerate_high": acceleration
        in (ACCELERATION_HIGH, ACCELERATION_BOTH),
        "accelerate_low": acceleration
        in (ACCELERATION_LOW, ACCELERATION_BOTH),
        "accelerated_steps": accelerated_steps,
        "full_steps": full_steps,
        "high_budget": high_budget,
        "low_budget": low_budget,
        "sigma_budget_mode": sigma_budget_mode,
        "high_range_share": high_range_share,
        "high_range_percent": high_range_share * 100.0,
        "profile_high_range_share": profile_share,
        "projected_steps_high": projected_steps_high,
        "projected_steps_low": projected_steps_low,
        "steps": steps,
        "requested_steps": steps,
        "actual_steps": len(sigma_values) - 1,
        "scheduler_requested_steps": steps,
        "scheduler_actual_steps": scheduler_actual_steps,
        "scheduler": scheduler,
        "priority": priority,
        "boundary": boundary,
        "native_shift": native_shift,
        "curve_profile": curve_profile,
        "curve_mode": curve_mode,
        "shift_anchor": shift_anchor,
        "shift": float(shift),
        "shift_source": shift_source,
        "requested_shift": requested_shift,
        "steps_high": steps_high,
        "steps_low": steps_low,
        "requested_steps_high": steps_high,
        "requested_steps_low": steps_low,
        "actual_steps_high": len(sigmas_high) - 1,
        "actual_steps_low": len(sigmas_low) - 1,
        "sigmas": sigmas,
        "sigmas_high": sigmas_high,
        "sigmas_low": sigmas_low,
        "split_sigma": boundary_sigma_after,
        "boundary_sigma_before": boundary_sigma_before,
        "boundary_sigma_after": boundary_sigma_after,
        "boundary_distance": boundary_distance,
        "boundary_error": boundary_distance,
        "boundary_straddled": (
            boundary_sigma_before >= boundary > boundary_sigma_after
        ),
        "manual_split": (
            forced_steps_high is not None
            or forced_high_range_percent is not None
            or forced_sigma_budget_mode is not None
        ),
        "warnings": warnings,
    }
    plan["summary"] = _summary(plan)
    return validate_plan(plan)


def _sequences_close(left: Sequence[Any], right: Sequence[Any]) -> bool:
    left_values = _sigma_values(left)
    right_values = _sigma_values(right)
    if len(left_values) != len(right_values):
        return False
    return all(
        math.isclose(a, b, rel_tol=1e-9, abs_tol=SIGMA_TOLERANCE)
        for a, b in zip(left_values, right_values)
    )


def validate_plan(plan: Any) -> dict[str, Any]:
    if not isinstance(plan, dict) or plan.get("type") != PLAN_TYPE:
        raise ValueError("Expected a Wan 2.2 Sampling Plan.")
    if plan.get("version") != PLAN_VERSION:
        raise ValueError(
            f"Unsupported Wan 2.2 Sampling Plan version: {plan.get('version')}."
        )

    required = {
        "source_controls",
        "overrides",
        "curve_profile",
        "curve_mode",
        "shift_anchor",
        "shift",
        "shift_source",
        "steps",
        "steps_high",
        "steps_low",
        "sigmas",
        "sigmas_high",
        "sigmas_low",
        "boundary",
    }
    missing = sorted(required.difference(plan))
    if missing:
        raise ValueError("Sampling Plan is missing: " + ", ".join(missing))
    if not isinstance(plan["source_controls"], dict):
        raise ValueError("Sampling Plan source_controls must be a dictionary.")
    if not isinstance(plan["overrides"], dict):
        raise ValueError("Sampling Plan overrides must be a dictionary.")
    if plan["curve_mode"] not in {"exact", "piecewise"}:
        raise ValueError(f"Unknown Sampling Plan curve mode: {plan['curve_mode']}.")

    steps = int(plan["steps"])
    steps_high = int(plan["steps_high"])
    steps_low = int(plan["steps_low"])
    if steps_high < 1 or steps_low < 1 or steps != steps_high + steps_low:
        raise ValueError("Sampling Plan must contain nonempty high and low stages.")

    _, values = _validated_sigmas(
        plan["sigmas"],
        context="Sampling Plan sigma curve",
    )
    _, high_values = _validated_sigmas(
        plan["sigmas_high"],
        context="Sampling Plan high sigma curve",
        require_terminal_zero=False,
    )
    _, low_values = _validated_sigmas(
        plan["sigmas_low"],
        context="Sampling Plan low sigma curve",
    )
    if len(values) != steps + 1:
        raise ValueError("Sampling Plan step count disagrees with its sigma curve.")
    if len(high_values) != steps_high + 1:
        raise ValueError("Sampling Plan high-step count disagrees with its sigma curve.")
    if len(low_values) != steps_low + 1:
        raise ValueError("Sampling Plan low-step count disagrees with its sigma curve.")
    if not _sequences_close(plan["sigmas_high"], plan["sigmas"][: steps_high + 1]):
        raise ValueError("Sampling Plan high sigmas are not a prefix of the full curve.")
    if not _sequences_close(plan["sigmas_low"], plan["sigmas"][steps_high:]):
        raise ValueError("Sampling Plan low sigmas are not a suffix of the full curve.")
    if not math.isclose(
        high_values[-1],
        low_values[0],
        rel_tol=1e-9,
        abs_tol=SIGMA_TOLERANCE,
    ):
        raise ValueError("Sampling Plan stages must share exactly one handoff sigma.")

    boundary = float(plan["boundary"])
    before = values[steps_high - 1]
    after = values[steps_high]
    if not before >= boundary > after:
        raise ValueError(
            "Sampling Plan does not straddle the Wan expert boundary at its handoff."
        )

    shift = float(plan["shift"])
    if not math.isfinite(shift) or not MIN_SHIFT <= shift <= MAX_SHIFT:
        raise ValueError("Sampling Plan shift is outside the supported range.")
    return plan
