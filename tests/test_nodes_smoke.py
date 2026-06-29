import importlib.util
import pathlib
import sys
import types
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
PACKAGE_NAME = "sampling_planner_smoke"


class FakeModel:
    def __init__(self, shift=1.0):
        self.shift = shift

    def clone(self):
        return FakeModel(self.shift)

    def get_model_object(self, name):
        if name != "model_sampling":
            raise KeyError(name)
        return self


class FakeModelSamplingSD3:
    def patch(self, model, shift, multiplier=1000):
        del multiplier
        patched = model.clone()
        patched.shift = float(shift)
        return (patched,)


def fake_calculate_sigmas(model_sampling, scheduler, steps):
    del scheduler
    shift = model_sampling.shift
    values = []
    for index in range(steps + 1):
        timestep = 1.0 - index / steps
        if timestep == 0.0:
            values.append(0.0)
        else:
            values.append(
                shift * timestep / (1.0 + (shift - 1.0) * timestep)
            )
    return FakeTensor(values)


class FakeTensor(list):
    def cpu(self):
        return self

    def __getitem__(self, item):
        value = super().__getitem__(item)
        if isinstance(item, slice):
            return FakeTensor(value)
        return value


def load_nodes_module():
    comfy = types.ModuleType("comfy")
    comfy.__path__ = []
    comfy_samplers = types.ModuleType("comfy.samplers")
    installed_schedulers = ["simple", "karras", "custom_installed"]
    installed_samplers = ["euler", "heun", "custom_sampler"]
    comfy_samplers.SCHEDULER_NAMES = installed_schedulers
    comfy_samplers.KSampler = types.SimpleNamespace(
        SCHEDULERS=installed_schedulers,
        SAMPLERS=installed_samplers,
        DISCARD_PENULTIMATE_SIGMA_SAMPLERS={"custom_sampler"},
    )
    comfy_samplers.SAMPLER_NAMES = installed_samplers
    comfy_samplers.calculate_sigmas = fake_calculate_sigmas
    comfy.samplers = comfy_samplers

    comfy_extras = types.ModuleType("comfy_extras")
    comfy_extras.__path__ = []
    model_advanced = types.ModuleType("comfy_extras.nodes_model_advanced")
    model_advanced.ModelSamplingSD3 = FakeModelSamplingSD3

    sys.modules["comfy"] = comfy
    sys.modules["comfy.samplers"] = comfy_samplers
    sys.modules["comfy_extras"] = comfy_extras
    sys.modules["comfy_extras.nodes_model_advanced"] = model_advanced

    spec = importlib.util.spec_from_file_location(
        PACKAGE_NAME,
        ROOT / "__init__.py",
        submodule_search_locations=[str(ROOT)],
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[PACKAGE_NAME] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return sys.modules[f"{PACKAGE_NAME}.nodes"]


class NodeSmokeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.nodes = load_nodes_module()

    def test_all_nodes_are_registered(self):
        self.assertEqual(
            set(self.nodes.NODE_CLASS_MAPPINGS),
            {
                "SchedulerSelector",
                "SamplerSelector",
                "TaskSelector",
                "PrioritySelector",
                "StepBudget",
                "AccelerationSelector",
                "AccelerationModelPair",
                "SamplingPlanWan22",
                "StepSplitOverride",
                "RangeSplitOverrideLegacy",
                "ShiftOverride",
                "ModelPairBreakout",
                "KSamplerBreakout",
                "SigmaBreakout",
            },
        )

    def test_scheduler_selector_uses_live_installed_combo(self):
        selector = self.nodes.SchedulerSelector
        options = selector.INPUT_TYPES()["required"]["scheduler"][0]
        self.assertIn("custom_installed", options)
        self.assertIs(
            options,
            sys.modules["comfy.samplers"].KSampler.SCHEDULERS,
        )
        self.assertEqual(
            selector.RETURN_TYPES,
            (sys.modules["comfy.samplers"].KSampler.SCHEDULERS,),
        )
        self.assertEqual(
            selector().select_scheduler("custom_installed"),
            ("custom_installed",),
        )

    def test_sampler_selector_uses_live_installed_combo(self):
        selector = self.nodes.SamplerSelector
        options = selector.INPUT_TYPES()["required"]["sampler"][0]
        self.assertIn("custom_sampler", options)
        self.assertIs(
            options,
            sys.modules["comfy.samplers"].KSampler.SAMPLERS,
        )
        self.assertEqual(
            selector.RETURN_TYPES,
            (sys.modules["comfy.samplers"].KSampler.SAMPLERS,),
        )
        self.assertEqual(
            selector().select_sampler("custom_sampler"),
            ("custom_sampler",),
        )

    def test_task_selector_matches_plan_combo(self):
        selector = self.nodes.TaskSelector
        selector_options = selector.INPUT_TYPES()["required"]["task"][0]
        plan_options = self.nodes.SamplingPlanWan22.INPUT_TYPES()["required"][
            "task"
        ][0]
        self.assertIs(selector_options, plan_options)
        self.assertEqual(selector.RETURN_TYPES, (plan_options,))
        self.assertEqual(selector().select_task("I2V"), ("I2V",))

    def test_priority_selector_matches_plan_combo(self):
        selector = self.nodes.PrioritySelector
        selector_options = selector.INPUT_TYPES()["required"]["priority"][0]
        plan_options = self.nodes.SamplingPlanWan22.INPUT_TYPES()["required"][
            "priority"
        ][0]
        self.assertIs(selector_options, plan_options)
        self.assertEqual(selector.RETURN_TYPES, (plan_options,))
        self.assertEqual(
            selector().select_priority("50/50 Split"),
            ("50/50 Split",),
        )

    def test_acceleration_selector_matches_plan_combo(self):
        selector = self.nodes.AccelerationSelector
        options = selector.INPUT_TYPES()["required"]["acceleration"][0]
        plan_options = self.nodes.SamplingPlanWan22.INPUT_TYPES()["required"][
            "acceleration"
        ][0]
        self.assertIs(options, plan_options)
        self.assertEqual(selector.RETURN_TYPES, (plan_options,))
        self.assertEqual(
            selector().select_acceleration("High only"),
            ("High only",),
        )

    def test_step_budget_helper_matches_plan_socket(self):
        helper = self.nodes.StepBudget()
        budget = helper.create_budget(10, 30)[0]
        self.assertEqual(
            budget,
            {
                "type": self.nodes.STEP_BUDGET_TYPE,
                "accelerated_steps": 10,
                "full_steps": 30,
            },
        )
        plan_type = self.nodes.SamplingPlanWan22.INPUT_TYPES()["required"][
            "step_budget"
        ][0]
        self.assertEqual(plan_type, self.nodes.STEP_BUDGET_TYPE)

    def test_acceleration_model_pair_is_lazy_and_handles_none(self):
        planner = self.nodes.SamplingPlanWan22()
        pair = self.nodes.AccelerationModelPair()
        budget = self.nodes.StepBudget().create_budget(10, 30)[0]
        none_result = planner.create_plan(
            FakeModel(),
            "T2V",
            "None",
            budget,
            "simple",
            "Balanced",
        )["result"]
        high_result = planner.create_plan(
            FakeModel(),
            "T2V",
            "High only",
            budget,
            "simple",
            "Balanced",
        )["result"]
        none_state = none_result[1]
        high_state = high_result[1]

        self.assertEqual(len(none_result), 2)
        self.assertEqual(none_state["mode"], "None")
        self.assertFalse(none_state["accelerate_high"])
        self.assertFalse(none_state["accelerate_low"])
        self.assertTrue(high_state["accelerate_high"])
        self.assertFalse(high_state["accelerate_low"])

        schema = pair.INPUT_TYPES()["required"]
        self.assertTrue(schema["base_high"][1]["lazy"])
        self.assertTrue(schema["accelerated_high"][1]["lazy"])
        self.assertTrue(schema["base_cfg"][1]["forceInput"])
        self.assertTrue(schema["accelerated_cfg"][1]["forceInput"])

        self.assertEqual(
            pair.check_lazy_status(none_state),
            ["base_high", "base_low"],
        )
        self.assertEqual(
            pair.check_lazy_status(high_state),
            ["accelerated_high", "base_low"],
        )

        base_high = object()
        accelerated_high = object()
        base_low = object()
        accelerated_low = object()
        self.assertEqual(
            pair.select_models(
                none_state,
                base_high,
                accelerated_high,
                base_low,
                accelerated_low,
                3.5,
                1.0,
            ),
            (base_high, base_low, 3.5, 3.5),
        )
        self.assertEqual(
            pair.select_models(
                high_state,
                base_high,
                accelerated_high,
                base_low,
                accelerated_low,
                3.5,
                1.0,
            ),
            (accelerated_high, base_low, 1.0, 3.5),
        )

        low_state = planner.create_plan(
            FakeModel(),
            "T2V",
            "Low only",
            budget,
            "simple",
            "Balanced",
        )["result"][1]
        both_state = planner.create_plan(
            FakeModel(),
            "T2V",
            "High + Low",
            budget,
            "simple",
            "Balanced",
        )["result"][1]
        self.assertEqual(
            pair.select_models(
                low_state,
                base_high,
                accelerated_high,
                base_low,
                accelerated_low,
                3.5,
                1.0,
            ),
            (base_high, accelerated_low, 3.5, 1.0),
        )
        self.assertEqual(
            pair.select_models(
                both_state,
                base_high,
                accelerated_high,
                base_low,
                accelerated_low,
                3.5,
                1.0,
            ),
            (accelerated_high, accelerated_low, 1.0, 1.0),
        )

    def test_plan_scheduler_widget_uses_same_live_combo(self):
        plan_options = self.nodes.SamplingPlanWan22.INPUT_TYPES()["required"][
            "scheduler"
        ][0]
        self.assertIs(
            plan_options,
            sys.modules["comfy.samplers"].KSampler.SCHEDULERS,
        )

    def test_plan_override_and_breakouts_execute(self):
        model = FakeModel()
        plan_node = self.nodes.SamplingPlanWan22()
        budget = self.nodes.StepBudget().create_budget(10, 30)[0]
        created = plan_node.create_plan(
            model,
            "T2V",
            "High + Low",
            budget,
            "simple",
            "Balanced",
        )
        plan = created["result"][0]
        acceleration_state = created["result"][1]
        self.assertTrue(acceleration_state["accelerate_high"])
        self.assertTrue(acceleration_state["accelerate_low"])

        override_node = self.nodes.StepSplitOverride()
        revised = override_node.override_split(plan, 6)["result"][0]
        self.assertEqual(revised["steps_high"], 6)
        self.assertEqual(revised["steps_low"], plan["steps"] - 6)

        shifted_then_split = override_node.override_split(
            self.nodes.ShiftOverride().override_shift(plan, 7.5)["result"][0],
            6,
        )["result"][0]
        split_then_shift = self.nodes.ShiftOverride().override_shift(
            revised,
            7.5,
        )["result"][0]
        self.assertEqual(
            shifted_then_split["overrides"],
            split_then_shift["overrides"],
        )
        self.assertEqual(
            list(shifted_then_split["sigmas"]),
            list(split_then_shift["sigmas"]),
        )

        sigma_result = self.nodes.SigmaBreakout().breakout(revised)["result"]
        self.assertEqual(len(sigma_result), 8)
        self.assertEqual(len(sigma_result[0]), 7)
        self.assertEqual(
            len(sigma_result[1]),
            revised["steps_low"] + 1,
        )

        model_pair_result = self.nodes.ModelPairBreakout().breakout(
            revised,
            FakeModel(),
            FakeModel(),
        )["result"]
        self.assertEqual(len(model_pair_result), 3)
        self.assertAlmostEqual(model_pair_result[0].shift, revised["shift"])
        self.assertAlmostEqual(model_pair_result[1].shift, revised["shift"])

        exact_plan = plan_node.create_plan(
            FakeModel(),
            "I2V",
            "High + Low",
            budget,
            "simple",
            "Balanced",
        )["result"][0]
        ksampler_result = self.nodes.KSamplerBreakout().breakout(
            exact_plan,
            FakeModel(),
            FakeModel(),
            "euler",
        )["result"]
        self.assertEqual(len(ksampler_result), 9)
        self.assertAlmostEqual(ksampler_result[0].shift, exact_plan["shift"])
        self.assertAlmostEqual(ksampler_result[1].shift, exact_plan["shift"])
        self.assertEqual(ksampler_result[3], exact_plan["steps_high"])
        self.assertEqual(ksampler_result[4], exact_plan["steps_high"])

        with self.assertRaisesRegex(ValueError, "piecewise sigma curve"):
            self.nodes.KSamplerBreakout().breakout(
                plan,
                FakeModel(),
                FakeModel(),
                "euler",
            )
        with self.assertRaisesRegex(ValueError, "changes the core KSampler sigma count"):
            self.nodes.KSamplerBreakout().breakout(
                exact_plan,
                FakeModel(),
                FakeModel(),
                "custom_sampler",
            )

    def test_override_nodes_expose_compact_exact_controls(self):
        split = self.nodes.StepSplitOverride.INPUT_TYPES()["required"]
        shift = self.nodes.ShiftOverride.INPUT_TYPES()["required"]
        self.assertEqual(split["steps_high"][0], "INT")
        self.assertEqual(shift["shift"][0], "FLOAT")
        self.assertEqual(
            self.nodes.ModelPairBreakout.RETURN_NAMES,
            ("model_high", "model_low", "plan"),
        )


if __name__ == "__main__":
    unittest.main()
