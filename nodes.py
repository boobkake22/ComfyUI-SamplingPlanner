from __future__ import annotations

from typing import Any

import comfy.samplers
from comfy_extras.nodes_model_advanced import ModelSamplingSD3

from .wan22_planner import (
    ACCELERATION_BOTH,
    ACCELERATION_OPTIONS,
    SIGMA_BUDGET_ACCELERATED_50_50,
    PLAN_TYPE,
    PRIORITY_BALANCED,
    PRIORITY_OPTIONS,
    TASK_OPTIONS,
    build_plan,
    validate_plan,
)


CATEGORY = "sampling/Sampling Planner/Wan 2.2"
HELPER_CATEGORY = "sampling/Sampling Planner/Helpers"
ACCELERATION_STATE_TYPE = "SAMPLING_ACCELERATION_STATE"
STEP_BUDGET_TYPE = "SAMPLING_STEP_BUDGET"
SIGMA_BUDGET_OVERRIDE_TYPE = "SAMPLING_SIGMA_BUDGET_OVERRIDE"


def installed_scheduler_options():
    """Return the live scheduler combo used by installed core KSampler nodes.

    Returning the original list object, rather than a copy, preserves ComfyUI's
    legacy combo-output compatibility with KSampler scheduler inputs.
    """
    ksampler = getattr(comfy.samplers, "KSampler", None)
    schedulers = getattr(ksampler, "SCHEDULERS", None)
    if schedulers:
        return schedulers

    schedulers = getattr(comfy.samplers, "SCHEDULER_NAMES", None)
    if schedulers:
        return schedulers

    handlers = getattr(comfy.samplers, "SCHEDULER_HANDLERS", {})
    options = list(handlers)
    return options or ["simple"]


def default_scheduler() -> str:
    options = installed_scheduler_options()
    return "simple" if "simple" in options else options[0]


def installed_sampler_options():
    """Return the live sampler combo used by installed core KSampler nodes."""
    ksampler = getattr(comfy.samplers, "KSampler", None)
    samplers = getattr(ksampler, "SAMPLERS", None)
    if samplers:
        return samplers

    samplers = getattr(comfy.samplers, "SAMPLER_NAMES", None)
    if samplers:
        return samplers

    return ["euler"]


def default_sampler() -> str:
    options = installed_sampler_options()
    return "euler" if "euler" in options else options[0]


class _SchedulerSelectorReturnTypes:
    def __get__(self, instance, owner):
        del instance, owner
        return (installed_scheduler_options(),)


class _SamplerSelectorReturnTypes:
    def __get__(self, instance, owner):
        del instance, owner
        return (installed_sampler_options(),)


class _TaskSelectorReturnTypes:
    def __get__(self, instance, owner):
        del instance, owner
        return (TASK_OPTIONS,)


class _PrioritySelectorReturnTypes:
    def __get__(self, instance, owner):
        del instance, owner
        return (PRIORITY_OPTIONS,)


class _AccelerationSelectorReturnTypes:
    def __get__(self, instance, owner):
        del instance, owner
        return (ACCELERATION_OPTIONS,)


def _patch_model(model: Any, shift: float) -> Any:
    return ModelSamplingSD3().patch(model, float(shift))[0]


def _sigma_provider(model: Any):
    def calculate(shift: float, steps: int, scheduler: str):
        patched = _patch_model(model, shift)
        return comfy.samplers.calculate_sigmas(
            patched.get_model_object("model_sampling"),
            scheduler,
            steps,
        ).cpu()

    return calculate


def _ui_result(plan: dict[str, Any], *result: Any) -> dict[str, Any]:
    return {
        "ui": {"text": [plan["summary"]]},
        "result": result,
    }


def _acceleration_state(plan: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": ACCELERATION_STATE_TYPE,
        "mode": plan["acceleration"],
        "accelerate_high": plan["accelerate_high"],
        "accelerate_low": plan["accelerate_low"],
    }


def _rebuild_plan(plan: Any, **override_updates: Any) -> dict[str, Any]:
    source = validate_plan(plan)
    overrides = dict(source.get("overrides", {}))
    overrides.update(override_updates)
    return build_plan(
        model=source["model"],
        task=source["task"],
        acceleration=source["acceleration"],
        accelerated_steps=source["accelerated_steps"],
        full_steps=source["full_steps"],
        scheduler=source["scheduler"],
        priority=source["priority"],
        sigma_provider=_sigma_provider(source["model"]),
        forced_steps_high=overrides.get("steps_high"),
        forced_shift=overrides.get("shift"),
        forced_high_range_percent=overrides.get("high_range_percent"),
        forced_sigma_budget_mode=overrides.get("sigma_budget_mode"),
    )


def _validate_step_budget(step_budget: Any) -> dict[str, Any]:
    if (
        not isinstance(step_budget, dict)
        or step_budget.get("type") != STEP_BUDGET_TYPE
    ):
        raise ValueError("Expected a Step Budget.")
    return step_budget


def _validate_sigma_budget_override(sigma_override: Any) -> str | None:
    if sigma_override is None:
        return None

    if isinstance(sigma_override, dict):
        if sigma_override.get("type") != SIGMA_BUDGET_OVERRIDE_TYPE:
            raise ValueError("Expected a Sigma Budget Override.")
        mode = sigma_override.get("mode")
    else:
        mode = sigma_override

    if mode in (None, ""):
        return None
    if mode != SIGMA_BUDGET_ACCELERATED_50_50:
        raise ValueError(f"Unknown Sigma Budget Override: {mode}")
    return mode


class SamplingPlanWan22:
    DESCRIPTION = (
        "Creates a task-aware Wan 2.2 sampling plan from accelerated and full "
        "range-equivalent step budgets."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": (
                    "MODEL",
                    {
                        "tooltip": (
                            "A Wan 2.2 model used only to calculate the sigma schedule. "
                            "Connect either the high- or low-noise model."
                        )
                    },
                ),
                "task": (
                    TASK_OPTIONS,
                    {
                        "default": "T2V",
                        "tooltip": "Selects the official Wan 2.2 expert boundary.",
                    },
                ),
                "acceleration": (
                    ACCELERATION_OPTIONS,
                    {
                        "default": ACCELERATION_BOTH,
                        "tooltip": (
                            "Which expert stages receive a compatible acceleration "
                            "LoRA or distilled model. Single-stage modes reserve more "
                            "steps for the unaccelerated stage."
                        ),
                    },
                ),
                "step_budget": (
                    STEP_BUDGET_TYPE,
                    {
                        "tooltip": (
                            "Accelerated/full range-equivalent budgets from "
                            "Step Budget."
                        )
                    },
                ),
                "scheduler": (
                    installed_scheduler_options(),
                    {
                        "default": default_scheduler(),
                        "tooltip": (
                            "Runtime scheduler list accepted by the installed core "
                            "KSampler. Use Scheduler Selector to share this "
                            "choice across sampler branches."
                        ),
                    },
                ),
                "priority": (
                    PRIORITY_OPTIONS,
                    {
                        "default": PRIORITY_BALANCED,
                        "tooltip": (
                            "Balanced uses the profile allocation. Motion / Structure "
                            "allocates more of the budget above the expert boundary. "
                            "Detail / Refinement allocates more below it. 50/50 Split "
                            "uses equal high/low steps regardless of the profile."
                        ),
                    },
                ),
            },
            "optional": {
                "sigma_override": (
                    SIGMA_BUDGET_OVERRIDE_TYPE,
                    {
                        "tooltip": (
                            "Optional side-channel override. This is intentionally "
                            "not part of the required plan chain, so a muted group "
                            "can disable it without severing the plan wire."
                        )
                    },
                ),
            },
        }

    RETURN_TYPES = (PLAN_TYPE, ACCELERATION_STATE_TYPE)
    RETURN_NAMES = ("plan", "acceleration_state")
    FUNCTION = "create_plan"
    CATEGORY = CATEGORY

    def create_plan(
        self,
        model,
        task,
        acceleration,
        step_budget,
        scheduler,
        priority,
        sigma_override=None,
    ):
        step_budget = _validate_step_budget(step_budget)
        sigma_budget_mode = _validate_sigma_budget_override(sigma_override)
        plan = build_plan(
            model=model,
            task=task,
            acceleration=acceleration,
            accelerated_steps=step_budget["accelerated_steps"],
            full_steps=step_budget["full_steps"],
            scheduler=scheduler,
            priority=priority,
            sigma_provider=_sigma_provider(model),
            forced_sigma_budget_mode=sigma_budget_mode,
        )
        print(f"Sampling Plan (Wan 2.2): {plan['summary']}")
        return _ui_result(plan, plan, _acceleration_state(plan))


class SchedulerSelector:
    DESCRIPTION = (
        "Selects from the scheduler names currently registered for the installed "
        "core KSampler and outputs a compatible combo value for fan-out wiring."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "scheduler": (
                    installed_scheduler_options(),
                    {
                        "default": default_scheduler(),
                        "tooltip": (
                            "Detected from the live KSampler scheduler registry. "
                            "Connect this output to the Wan planner and each KSampler "
                            "scheduler input."
                        ),
                    },
                )
            }
        }

    RETURN_TYPES = _SchedulerSelectorReturnTypes()
    RETURN_NAMES = ("scheduler",)
    FUNCTION = "select_scheduler"
    CATEGORY = HELPER_CATEGORY

    def select_scheduler(self, scheduler):
        if scheduler not in installed_scheduler_options():
            raise ValueError(
                f"Scheduler '{scheduler}' is not currently registered with KSampler."
            )
        return (scheduler,)


class SamplerSelector:
    DESCRIPTION = (
        "Selects from the sampler names currently registered for the installed "
        "core KSampler and outputs a compatible combo value for fan-out wiring."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "sampler": (
                    installed_sampler_options(),
                    {
                        "default": default_sampler(),
                        "tooltip": (
                            "Detected from the live KSampler sampler registry. "
                            "Connect this output to compatible KSampler sampler inputs."
                        ),
                    },
                )
            }
        }

    RETURN_TYPES = _SamplerSelectorReturnTypes()
    RETURN_NAMES = ("sampler",)
    FUNCTION = "select_sampler"
    CATEGORY = HELPER_CATEGORY

    def select_sampler(self, sampler):
        if sampler not in installed_sampler_options():
            raise ValueError(
                f"Sampler '{sampler}' is not currently registered with KSampler."
            )
        return (sampler,)


class TaskSelector:
    DESCRIPTION = (
        "Selects the generation task and outputs a compatible combo value for "
        "the Sampling Plan."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "task": (
                    TASK_OPTIONS,
                    {
                        "default": "T2V",
                        "tooltip": (
                            "T2V uses Wan's 0.875 expert boundary. I2V uses 0.900."
                        ),
                    },
                )
            }
        }

    RETURN_TYPES = _TaskSelectorReturnTypes()
    RETURN_NAMES = ("task",)
    FUNCTION = "select_task"
    CATEGORY = HELPER_CATEGORY

    def select_task(self, task):
        if task not in TASK_OPTIONS:
            raise ValueError(f"Unknown task: {task}")
        return (task,)


class PrioritySelector:
    DESCRIPTION = (
        "Selects the sampling allocation priority and outputs a compatible combo "
        "value for the Sampling Plan."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "priority": (
                    PRIORITY_OPTIONS,
                    {
                        "default": PRIORITY_BALANCED,
                        "tooltip": (
                            "Balanced follows the profile, 50/50 forces an even "
                            "split, Motion favors high noise, and Detail favors "
                            "low-noise refinement."
                        ),
                    },
                )
            }
        }

    RETURN_TYPES = _PrioritySelectorReturnTypes()
    RETURN_NAMES = ("priority",)
    FUNCTION = "select_priority"
    CATEGORY = HELPER_CATEGORY

    def select_priority(self, priority):
        if priority not in PRIORITY_OPTIONS:
            raise ValueError(f"Unknown priority: {priority}")
        return (priority,)


class StepBudget:
    DESCRIPTION = (
        "Defines full-range-equivalent step budgets for accelerated and "
        "unaccelerated sampling."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "accelerated_steps": (
                    "INT",
                    {
                        "default": 10,
                        "min": 2,
                        "max": 99,
                        "step": 1,
                        "tooltip": (
                            "Step count used if the entire denoising range were "
                            "accelerated."
                        ),
                    },
                ),
                "full_steps": (
                    "INT",
                    {
                        "default": 30,
                        "min": 2,
                        "max": 99,
                        "step": 1,
                        "tooltip": (
                            "Step count used if the entire denoising range were "
                            "unaccelerated."
                        ),
                    },
                ),
            }
        }

    RETURN_TYPES = (STEP_BUDGET_TYPE,)
    RETURN_NAMES = ("step_budget",)
    FUNCTION = "create_budget"
    CATEGORY = HELPER_CATEGORY

    def create_budget(self, accelerated_steps, full_steps):
        accelerated_steps = int(accelerated_steps)
        full_steps = int(full_steps)
        if accelerated_steps < 2 or full_steps < 2:
            raise ValueError("Step budgets must both be at least 2.")
        return (
            {
                "type": STEP_BUDGET_TYPE,
                "accelerated_steps": accelerated_steps,
                "full_steps": full_steps,
            },
        )


class AccelerationSelector:
    DESCRIPTION = (
        "Selects which Wan expert stages receive acceleration and outputs a "
        "compatible combo value for the Sampling Plan."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "acceleration": (
                    ACCELERATION_OPTIONS,
                    {
                        "default": ACCELERATION_BOTH,
                        "tooltip": (
                            "Connect this output to the Sampling Plan acceleration "
                            "input. The plan produces the derived routing state."
                        ),
                    },
                )
            }
        }

    RETURN_TYPES = _AccelerationSelectorReturnTypes()
    RETURN_NAMES = ("acceleration",)
    FUNCTION = "select_acceleration"
    CATEGORY = HELPER_CATEGORY

    def select_acceleration(self, acceleration):
        if acceleration not in ACCELERATION_OPTIONS:
            raise ValueError(f"Unknown acceleration mode: {acceleration}")
        return (acceleration,)


class AccelerationModelPair:
    DESCRIPTION = (
        "Lazily selects base or accelerated high/low models from an Acceleration "
        "Sampling Plan state and routes matching CFG values. Unselected "
        "acceleration branches are not evaluated."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "state": (
                    ACCELERATION_STATE_TYPE,
                    {
                        "tooltip": (
                            "Bundled state from Sampling Plan. None selects "
                            "both base models."
                        )
                    },
                ),
                "base_high": (
                    "MODEL",
                    {"lazy": True, "tooltip": "High-noise model without acceleration."},
                ),
                "accelerated_high": (
                    "MODEL",
                    {
                        "lazy": True,
                        "tooltip": "High-noise model with its acceleration LoRA.",
                    },
                ),
                "base_low": (
                    "MODEL",
                    {"lazy": True, "tooltip": "Low-noise model without acceleration."},
                ),
                "accelerated_low": (
                    "MODEL",
                    {
                        "lazy": True,
                        "tooltip": "Low-noise model with its acceleration LoRA.",
                    },
                ),
                "base_cfg": (
                    "FLOAT",
                    {
                        "default": 3.5,
                        "min": 0.0,
                        "max": 100.0,
                        "step": 0.01,
                        "forceInput": True,
                        "tooltip": "CFG used by stages without acceleration.",
                    },
                ),
                "accelerated_cfg": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.0,
                        "max": 100.0,
                        "step": 0.01,
                        "forceInput": True,
                        "tooltip": "CFG used by accelerated stages.",
                    },
                ),
            }
        }

    RETURN_TYPES = ("MODEL", "MODEL", "FLOAT", "FLOAT")
    RETURN_NAMES = ("model_high", "model_low", "cfg_high", "cfg_low")
    FUNCTION = "select_models"
    CATEGORY = HELPER_CATEGORY

    @staticmethod
    def _validate_state(state):
        if (
            not isinstance(state, dict)
            or state.get("type") != ACCELERATION_STATE_TYPE
        ):
            raise ValueError("Expected acceleration state from Sampling Plan.")
        return state

    def check_lazy_status(
        self,
        state,
        base_high=None,
        accelerated_high=None,
        base_low=None,
        accelerated_low=None,
        base_cfg=None,
        accelerated_cfg=None,
    ):
        del base_cfg, accelerated_cfg
        state = self._validate_state(state)
        needed = []

        high_name = (
            "accelerated_high" if state["accelerate_high"] else "base_high"
        )
        low_name = "accelerated_low" if state["accelerate_low"] else "base_low"

        if locals()[high_name] is None:
            needed.append(high_name)
        if locals()[low_name] is None:
            needed.append(low_name)
        return needed

    def select_models(
        self,
        state,
        base_high=None,
        accelerated_high=None,
        base_low=None,
        accelerated_low=None,
        base_cfg=3.5,
        accelerated_cfg=1.0,
    ):
        state = self._validate_state(state)
        model_high = accelerated_high if state["accelerate_high"] else base_high
        model_low = accelerated_low if state["accelerate_low"] else base_low
        cfg_high = accelerated_cfg if state["accelerate_high"] else base_cfg
        cfg_low = accelerated_cfg if state["accelerate_low"] else base_cfg
        return (model_high, model_low, cfg_high, cfg_low)


class StepSplitOverride:
    DESCRIPTION = (
        "Overrides the exact number of high-noise transitions. The complete "
        "plan and sigma curve are rebuilt and validated."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "plan": (
                    PLAN_TYPE,
                    {"tooltip": "The Wan 2.2 plan to refine."},
                ),
                "steps_high": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 9999,
                        "step": 1,
                        "tooltip": (
                            "Exact number of sampling transitions assigned to "
                            "the high-noise expert. Use 0 to pass the plan "
                            "through unchanged."
                        ),
                    },
                ),
            }
        }

    RETURN_TYPES = (PLAN_TYPE,)
    RETURN_NAMES = ("plan",)
    FUNCTION = "override_split"
    CATEGORY = CATEGORY

    def override_split(self, plan, steps_high):
        steps_high = int(steps_high)
        revised = _rebuild_plan(
            plan,
            steps_high=None if steps_high == 0 else steps_high,
        )
        print(f"Step Split Override: {revised['summary']}")
        return _ui_result(revised, revised)


class AcceleratedSigmaOverride:
    DESCRIPTION = (
        "Side-channel control for progressive upscale workflows: when connected "
        "to Sampling Plan (Wan 2.2), Low-only acceleration uses the accelerated "
        "step budget split 50/50 for the sigma curve while preserving Low-only "
        "model and CFG routing. Because it is not in the required plan chain, "
        "it can be muted with its group."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {}}

    RETURN_TYPES = (SIGMA_BUDGET_OVERRIDE_TYPE,)
    RETURN_NAMES = ("sigma_override",)
    FUNCTION = "create_override"
    CATEGORY = CATEGORY

    def create_override(self):
        override = {
            "type": SIGMA_BUDGET_OVERRIDE_TYPE,
            "mode": SIGMA_BUDGET_ACCELERATED_50_50,
        }
        return {"ui": {"text": ["accelerated 50/50 sigmas"]}, "result": (override,)}


class RangeSplitOverrideLegacy:
    DESCRIPTION = (
        "Legacy percentage-based split override. New workflows should use "
        "Step Split Override for an exact high-stage transition count."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "plan": (PLAN_TYPE,),
                "high_range_percent": (
                    "FLOAT",
                    {
                        "default": 50.0,
                        "min": 1.0,
                        "max": 99.0,
                        "step": 1,
                        "round": 0.1,
                    },
                ),
            }
        }

    RETURN_TYPES = (PLAN_TYPE,)
    RETURN_NAMES = ("plan",)
    FUNCTION = "override_split"
    CATEGORY = CATEGORY

    def override_split(self, plan, high_range_percent):
        revised = _rebuild_plan(
            plan,
            high_range_percent=float(high_range_percent),
        )
        print(f"Range Split Override (Legacy): {revised['summary']}")
        return _ui_result(revised, revised)


class ShiftOverride:
    DESCRIPTION = (
        "Overrides the plan shift and rebuilds the complete sigma curve. Use "
        "Model Pair Breakout so both expert models receive the same shift."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "plan": (PLAN_TYPE,),
                "shift": (
                    "FLOAT",
                    {
                        "default": 8.0,
                        "min": 0.01,
                        "max": 100.0,
                        "step": 0.01,
                        "round": 0.001,
                    },
                ),
            }
        }

    RETURN_TYPES = (PLAN_TYPE,)
    RETURN_NAMES = ("plan",)
    FUNCTION = "override_shift"
    CATEGORY = CATEGORY

    def override_shift(self, plan, shift):
        revised = _rebuild_plan(plan, shift=float(shift))
        print(f"Shift Override: {revised['summary']}")
        return _ui_result(revised, revised)


class ModelPairBreakout:
    DESCRIPTION = (
        "Applies the plan's exact shift to both Wan expert models. This keeps "
        "model sampling and the planned sigma curve synchronized."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "plan": (PLAN_TYPE,),
                "model_high": ("MODEL",),
                "model_low": ("MODEL",),
            }
        }

    RETURN_TYPES = ("MODEL", "MODEL", PLAN_TYPE)
    RETURN_NAMES = ("model_high", "model_low", "plan")
    FUNCTION = "breakout"
    CATEGORY = CATEGORY

    def breakout(self, plan, model_high, model_low):
        plan = validate_plan(plan)
        shift = float(plan["shift"])
        return _ui_result(
            plan,
            _patch_model(model_high, shift),
            _patch_model(model_low, shift),
            plan,
        )


class KSamplerBreakout:
    DESCRIPTION = (
        "Prepares a high/low Wan 2.2 model pair and exposes the coordinated "
        "KSampler Advanced controls."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "plan": (PLAN_TYPE,),
                "model_high": (
                    "MODEL",
                    {"tooltip": "Wan 2.2 high-noise expert model."},
                ),
                "model_low": (
                    "MODEL",
                    {"tooltip": "Wan 2.2 low-noise expert model."},
                ),
                "sampler": (
                    installed_sampler_options(),
                    {
                        "default": default_sampler(),
                        "tooltip": (
                            "Used to confirm that core KSampler can reproduce "
                            "the plan's exact sigma curve."
                        ),
                    },
                ),
            }
        }

    RETURN_TYPES = (
        "MODEL",
        "MODEL",
        "INT",
        "INT",
        "INT",
        "INT",
        "INT",
        "FLOAT",
        PLAN_TYPE,
    )
    RETURN_NAMES = (
        "model_high",
        "model_low",
        "steps",
        "high_end_step",
        "low_start_step",
        "steps_high",
        "steps_low",
        "shift",
        "plan",
    )
    FUNCTION = "breakout"
    CATEGORY = CATEGORY

    def breakout(self, plan, model_high, model_low, sampler):
        plan = validate_plan(plan)
        if plan.get("curve_mode") != "exact":
            raise ValueError(
                "This plan uses a piecewise sigma curve and cannot be represented "
                "exactly by KSampler Advanced. Use Sigma Breakout with "
                "SamplerCustomAdvanced."
            )
        discard = getattr(
            getattr(comfy.samplers, "KSampler", None),
            "DISCARD_PENULTIMATE_SIGMA_SAMPLERS",
            set(),
        )
        if sampler in discard:
            raise ValueError(
                f"Sampler '{sampler}' changes the core KSampler sigma count. "
                "Use Sigma Breakout so the planned curve remains authoritative."
            )
        shift = float(plan["shift"])
        patched_high = _patch_model(model_high, shift)
        patched_low = _patch_model(model_low, shift)

        return _ui_result(
            plan,
            patched_high,
            patched_low,
            plan["steps"],
            plan["steps_high"],
            plan["steps_high"],
            plan["steps_high"],
            plan["steps_low"],
            shift,
            plan,
        )


class SigmaBreakout:
    DESCRIPTION = (
        "Exposes the complete, high-noise, and low-noise sigma schedules from "
        "a Wan 2.2 sampling plan."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"plan": (PLAN_TYPE,)}}

    RETURN_TYPES = (
        "SIGMAS",
        "SIGMAS",
        "SIGMAS",
        "INT",
        "INT",
        "INT",
        "FLOAT",
        PLAN_TYPE,
    )
    RETURN_NAMES = (
        "sigmas_high",
        "sigmas_low",
        "sigmas_all",
        "steps",
        "steps_high",
        "steps_low",
        "shift",
        "plan",
    )
    FUNCTION = "breakout"
    CATEGORY = CATEGORY

    def breakout(self, plan):
        plan = validate_plan(plan)
        return _ui_result(
            plan,
            plan["sigmas_high"],
            plan["sigmas_low"],
            plan["sigmas"],
            plan["steps"],
            plan["steps_high"],
            plan["steps_low"],
            float(plan["shift"]),
            plan,
        )


NODE_CLASS_MAPPINGS = {
    "SchedulerSelector": SchedulerSelector,
    "SamplerSelector": SamplerSelector,
    "TaskSelector": TaskSelector,
    "PrioritySelector": PrioritySelector,
    "StepBudget": StepBudget,
    "AccelerationSelector": AccelerationSelector,
    "AccelerationModelPair": AccelerationModelPair,
    "SamplingPlanWan22": SamplingPlanWan22,
    "StepSplitOverride": StepSplitOverride,
    "AcceleratedSigmaOverride": AcceleratedSigmaOverride,
    "RangeSplitOverrideLegacy": RangeSplitOverrideLegacy,
    "ShiftOverride": ShiftOverride,
    "ModelPairBreakout": ModelPairBreakout,
    "KSamplerBreakout": KSamplerBreakout,
    "SigmaBreakout": SigmaBreakout,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SchedulerSelector": "Scheduler Selector",
    "SamplerSelector": "Sampler Selector",
    "TaskSelector": "Task Selector",
    "PrioritySelector": "Priority Selector",
    "StepBudget": "Step Budget",
    "AccelerationSelector": "Acceleration Selector",
    "AccelerationModelPair": "Acceleration Model Pair",
    "SamplingPlanWan22": "Sampling Plan (Wan 2.2)",
    "StepSplitOverride": "Step Split Override",
    "AcceleratedSigmaOverride": "Accelerated 50/50 Sigma Override",
    "RangeSplitOverrideLegacy": "Range Split Override (Legacy)",
    "ShiftOverride": "Shift Override",
    "ModelPairBreakout": "Model Pair Breakout",
    "KSamplerBreakout": "KSampler Breakout",
    "SigmaBreakout": "Sigma Breakout",
}
