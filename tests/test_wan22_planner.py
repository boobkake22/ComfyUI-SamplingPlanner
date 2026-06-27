import math
import unittest

from wan22_planner import (
    ACCELERATION_BOTH,
    ACCELERATION_HIGH,
    ACCELERATION_LOW,
    ACCELERATION_NONE,
    CURVE_PROFILE_COMFYUI,
    CURVE_PROFILE_DISTILLED,
    CURVE_PROFILE_NATIVE,
    PLAN_VERSION,
    PRIORITY_BALANCED,
    PRIORITY_DETAIL,
    PRIORITY_EVEN,
    PRIORITY_MOTION,
    build_plan,
    validate_plan,
)


def shifted_linear_provider(shift, steps, scheduler):
    del scheduler
    values = []
    for index in range(steps + 1):
        timestep = 1.0 - index / steps
        if timestep == 0.0:
            values.append(0.0)
        else:
            values.append(
                shift * timestep / (1.0 + (shift - 1.0) * timestep)
            )
    return values


def insensitive_provider(shift, steps, scheduler):
    del shift, scheduler
    return [1.0 - index / steps for index in range(steps + 1)]


def build(**updates):
    arguments = {
        "model": object(),
        "task": "I2V",
        "acceleration": ACCELERATION_BOTH,
        "accelerated_steps": 10,
        "full_steps": 30,
        "scheduler": "simple",
        "priority": PRIORITY_BALANCED,
        "sigma_provider": shifted_linear_provider,
    }
    arguments.update(updates)
    return build_plan(**arguments)


class ProfileSelectionTests(unittest.TestCase):
    def test_i2v_accelerated_ten_step_golden_curve(self):
        plan = build()
        expected = [
            1.0,
            0.9863013698630136,
            0.9696969696969697,
            0.9491525423728813,
            0.923076923076923,
            0.8888888888888888,
            0.8421052631578948,
            0.7741935483870968,
            0.6666666666666666,
            0.47058823529411764,
            0.0,
        ]
        self.assertEqual(plan["version"], PLAN_VERSION)
        self.assertEqual(plan["curve_profile"], CURVE_PROFILE_COMFYUI)
        self.assertEqual(plan["curve_mode"], "exact")
        self.assertEqual(plan["shift"], 8.0)
        self.assertEqual((plan["steps_high"], plan["steps_low"]), (5, 5))
        self.assertTrue(
            any("matched high/low" in warning for warning in plan["warnings"])
        )
        for actual, wanted in zip(plan["sigmas"], expected):
            self.assertAlmostEqual(actual, wanted, places=12)

    def test_sigma_breakout_slices_match_ksampler_advanced_continuation(self):
        plan = build(priority=PRIORITY_EVEN)
        split = plan["steps_high"]
        sigmas = list(plan["sigmas"])

        # KSampler Advanced high stage:
        #   end_at_step=split, return_with_leftover_noise=enable
        # KSampler Advanced low stage:
        #   start_at_step=split, add_noise=disable
        self.assertEqual(list(plan["sigmas_high"]), sigmas[: split + 1])
        self.assertEqual(list(plan["sigmas_low"]), sigmas[split:])
        self.assertEqual(plan["sigmas_high"][-1], plan["sigmas_low"][0])
        self.assertNotEqual(plan["sigmas_high"][-1], 0.0)

    def test_unaccelerated_uses_task_native_shift(self):
        t2v = build(
            task="T2V",
            acceleration=ACCELERATION_NONE,
            accelerated_steps=4,
            full_steps=30,
        )
        i2v = build(
            task="I2V",
            acceleration=ACCELERATION_NONE,
            accelerated_steps=99,
            full_steps=30,
        )
        self.assertEqual(t2v["curve_profile"], CURVE_PROFILE_NATIVE)
        self.assertEqual(i2v["curve_profile"], CURVE_PROFILE_NATIVE)
        self.assertEqual(t2v["shift"], 12.0)
        self.assertEqual(i2v["shift"], 5.0)

    def test_four_step_acceleration_uses_distilled_profile(self):
        plan = build(accelerated_steps=4)
        self.assertEqual(plan["curve_profile"], CURVE_PROFILE_DISTILLED)
        self.assertEqual(plan["shift"], 5.0)
        self.assertEqual((plan["steps_high"], plan["steps_low"]), (2, 2))
        self.assertEqual(plan["curve_mode"], "exact")

    def test_t2v_ten_step_profile_is_boundary_safe(self):
        plan = build(task="T2V")
        self.assertEqual(plan["shift"], 8.0)
        self.assertTrue(plan["boundary_straddled"])
        self.assertLess(plan["boundary_sigma_after"], 0.875)
        self.assertGreaterEqual(plan["boundary_sigma_before"], 0.875)

    def test_inactive_budget_does_not_change_plan(self):
        both_a = build(full_steps=20)
        both_b = build(full_steps=80)
        none_a = build(
            acceleration=ACCELERATION_NONE,
            accelerated_steps=4,
            full_steps=30,
        )
        none_b = build(
            acceleration=ACCELERATION_NONE,
            accelerated_steps=40,
            full_steps=30,
        )
        self.assertEqual(
            (both_a["steps_high"], both_a["steps_low"], list(both_a["sigmas"])),
            (both_b["steps_high"], both_b["steps_low"], list(both_b["sigmas"])),
        )
        self.assertEqual(
            (none_a["steps_high"], none_a["steps_low"], list(none_a["sigmas"])),
            (none_b["steps_high"], none_b["steps_low"], list(none_b["sigmas"])),
        )


class AllocationAndCurveTests(unittest.TestCase):
    def test_mixed_acceleration_projects_budgets_independently(self):
        high_only = build(
            acceleration=ACCELERATION_HIGH,
            priority=PRIORITY_EVEN,
        )
        low_only = build(
            acceleration=ACCELERATION_LOW,
            priority=PRIORITY_EVEN,
        )
        self.assertEqual(
            (
                high_only["high_budget"],
                high_only["low_budget"],
                high_only["steps_high"],
                high_only["steps_low"],
            ),
            (10, 30, 5, 15),
        )
        self.assertEqual(
            (
                low_only["high_budget"],
                low_only["low_budget"],
                low_only["steps_high"],
                low_only["steps_low"],
            ),
            (30, 10, 15, 5),
        )

    def test_priority_ordering(self):
        motion = build(priority=PRIORITY_MOTION)
        balanced = build(priority=PRIORITY_BALANCED)
        detail = build(priority=PRIORITY_DETAIL)
        self.assertGreaterEqual(motion["steps_high"], balanced["steps_high"])
        self.assertGreaterEqual(balanced["steps_high"], detail["steps_high"])

    def test_odd_even_split_assigns_extra_to_low(self):
        plan = build(accelerated_steps=9, priority=PRIORITY_EVEN)
        self.assertEqual((plan["steps_high"], plan["steps_low"]), (4, 5))
        self.assertTrue(
            any("odd full-range budget" in item.lower() for item in plan["warnings"])
        )

    def test_non_natural_split_uses_piecewise_curve(self):
        plan = build(forced_steps_high=6)
        self.assertEqual(plan["curve_mode"], "piecewise")
        self.assertEqual((plan["steps_high"], plan["steps_low"]), (6, 4))
        self.assertTrue(plan["boundary_straddled"])
        self.assertEqual(len(plan["sigmas"]), 11)
        self.assertEqual(plan["sigmas_high"][-1], plan["sigmas_low"][0])

    def test_shift_insensitive_provider_still_builds_safe_piecewise_curve(self):
        plan = build(sigma_provider=insensitive_provider)
        self.assertEqual(plan["shift"], 8.0)
        self.assertEqual(plan["curve_mode"], "piecewise")
        self.assertTrue(plan["boundary_straddled"])

    def test_short_and_long_valid_providers_are_resampled(self):
        def short_provider(shift, steps, scheduler):
            del shift, steps, scheduler
            return [1.0, 0.0]

        def long_provider(shift, steps, scheduler):
            return shifted_linear_provider(shift, steps + 3, scheduler)

        for provider in (short_provider, long_provider):
            with self.subTest(provider=provider.__name__):
                plan = build(sigma_provider=provider)
                self.assertEqual(plan["curve_mode"], "piecewise")
                self.assertEqual(len(plan["sigmas"]), plan["steps"] + 1)
                self.assertTrue(plan["boundary_straddled"])
                self.assertNotEqual(
                    plan["scheduler_requested_steps"],
                    plan["scheduler_actual_steps"],
                )
                self.assertTrue(
                    any(
                        "scheduler returned" in warning.lower()
                        for warning in plan["warnings"]
                    )
                )

    def test_profile_shift_adjusts_only_when_anchor_has_no_crossing(self):
        def low_start_provider(shift, steps, scheduler):
            del scheduler
            start = min(1.0, shift / 10.0)
            return [start * (1.0 - index / steps) for index in range(steps + 1)]

        plan = build(sigma_provider=low_start_provider)
        self.assertEqual(plan["shift_source"], "profile_adjusted")
        self.assertGreaterEqual(plan["shift"], 9.0)
        self.assertTrue(plan["boundary_straddled"])


class OverrideTests(unittest.TestCase):
    def test_exact_step_override_preserves_effective_total(self):
        plan = build(forced_steps_high=6)
        self.assertEqual(plan["steps"], 10)
        self.assertEqual((plan["steps_high"], plan["steps_low"]), (6, 4))
        self.assertEqual(plan["overrides"], {"steps_high": 6})

    def test_shift_override_rebuilds_share_and_curve(self):
        plan = build(forced_shift=6.5)
        self.assertEqual(plan["shift"], 6.5)
        self.assertEqual(plan["shift_source"], "override")
        self.assertEqual(plan["overrides"], {"shift": 6.5})
        self.assertTrue(plan["boundary_straddled"])

    def test_step_and_shift_overrides_are_order_independent(self):
        first = build(forced_steps_high=6, forced_shift=7.25)
        second = build(forced_shift=7.25, forced_steps_high=6)
        self.assertEqual(first["overrides"], {"steps_high": 6, "shift": 7.25})
        self.assertEqual(first["overrides"], second["overrides"])
        self.assertEqual(first["steps_high"], second["steps_high"])
        self.assertEqual(list(first["sigmas"]), list(second["sigmas"]))

    def test_source_controls_are_sufficient_to_rebuild(self):
        plan = build(forced_steps_high=6, forced_shift=7.25)
        controls = dict(plan["source_controls"])
        rebuilt = build_plan(
            **controls,
            forced_steps_high=plan["overrides"]["steps_high"],
            forced_shift=plan["overrides"]["shift"],
        )
        self.assertEqual(rebuilt["steps_high"], plan["steps_high"])
        self.assertEqual(rebuilt["shift"], plan["shift"])
        self.assertEqual(list(rebuilt["sigmas"]), list(plan["sigmas"]))

    def test_legacy_percentage_override_is_retained(self):
        plan = build(forced_high_range_percent=60)
        self.assertEqual((plan["steps_high"], plan["steps_low"]), (6, 4))
        self.assertEqual(plan["overrides"], {"high_range_percent": 60.0})
        self.assertTrue(plan["manual_split"])

    def test_exact_override_wins_over_legacy_percentage(self):
        plan = build(
            forced_steps_high=6,
            forced_high_range_percent=20,
        )
        self.assertEqual((plan["steps_high"], plan["steps_low"]), (6, 4))
        self.assertEqual(plan["overrides"], {"steps_high": 6})
        self.assertTrue(any("legacy" in item.lower() for item in plan["warnings"]))

    def test_invalid_overrides_are_rejected(self):
        for arguments in (
            {"forced_steps_high": 0},
            {"forced_steps_high": 10},
            {"forced_shift": float("nan")},
            {"forced_shift": 101},
            {"forced_high_range_percent": 100},
        ):
            with self.subTest(arguments=arguments), self.assertRaises(ValueError):
                build(**arguments)


class ValidationAndAdversarialTests(unittest.TestCase):
    def assert_provider_rejected(self, provider):
        with self.assertRaises(ValueError):
            build(sigma_provider=provider)

    def test_nonfinite_providers_are_rejected(self):
        def nan_provider(shift, steps, scheduler):
            values = shifted_linear_provider(shift, steps, scheduler)
            values[2] = math.nan
            return values

        def infinite_provider(shift, steps, scheduler):
            values = shifted_linear_provider(shift, steps, scheduler)
            values[2] = math.inf
            return values

        self.assert_provider_rejected(nan_provider)
        self.assert_provider_rejected(infinite_provider)

    def test_structurally_invalid_providers_are_rejected(self):
        def reversed_provider(shift, steps, scheduler):
            return list(reversed(shifted_linear_provider(shift, steps, scheduler)))

        def duplicate_provider(shift, steps, scheduler):
            values = shifted_linear_provider(shift, steps, scheduler)
            values[2] = values[1]
            return values

        def nonmonotonic_provider(shift, steps, scheduler):
            values = shifted_linear_provider(shift, steps, scheduler)
            values[2] = values[1] + 0.01
            return values

        def missing_zero_provider(shift, steps, scheduler):
            values = shifted_linear_provider(shift, steps, scheduler)
            values[-1] = 0.01
            return values

        def too_short_provider(shift, steps, scheduler):
            del shift, steps, scheduler
            return [0.0]

        for provider in (
            reversed_provider,
            duplicate_provider,
            nonmonotonic_provider,
            missing_zero_provider,
            too_short_provider,
        ):
            with self.subTest(provider=provider.__name__):
                self.assert_provider_rejected(provider)

    def test_validate_plan_rejects_corrupted_invariants(self):
        cases = []
        wrong_length = dict(build())
        wrong_length["steps"] += 1
        cases.append(wrong_length)

        broken_boundary = dict(build())
        broken_boundary["steps_high"] = 4
        broken_boundary["steps_low"] = 6
        broken_boundary["sigmas_high"] = broken_boundary["sigmas"][:5]
        broken_boundary["sigmas_low"] = broken_boundary["sigmas"][4:]
        cases.append(broken_boundary)

        bad_terminal = dict(build())
        bad_terminal["sigmas"] = list(bad_terminal["sigmas"])
        bad_terminal["sigmas"][-1] = 0.01
        cases.append(bad_terminal)

        for plan in cases:
            with self.subTest(), self.assertRaises(ValueError):
                validate_plan(plan)

    def test_requested_and_actual_counts_agree(self):
        plan = build(forced_steps_high=6)
        self.assertEqual(plan["requested_steps"], plan["actual_steps"])
        self.assertEqual(plan["requested_steps_high"], plan["actual_steps_high"])
        self.assertEqual(plan["requested_steps_low"], plan["actual_steps_low"])


if __name__ == "__main__":
    unittest.main()
