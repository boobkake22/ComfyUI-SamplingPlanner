# Comfy sampler parity notes

Retrieved: 2026-06-27.

These notes compare ComfyUI's native `KSamplerAdvanced` path with the modular
`SamplerCustomAdvanced` path for Wan 2.2 two-stage high/low expert sampling.

## Source references

- Official ComfyUI `KSamplerAdvanced` node documentation:
  <https://docs.comfy.org/built-in-nodes/KSamplerAdvanced>
- Official ComfyUI `KSamplerAdvanced` implementation:
  <https://github.com/Comfy-Org/ComfyUI/blob/master/nodes.py>
- Official ComfyUI `KSampler` sigma slicing implementation:
  <https://github.com/Comfy-Org/ComfyUI/blob/master/comfy/samplers.py>
- Official ComfyUI `SamplerCustomAdvanced` implementation:
  <https://github.com/Comfy-Org/ComfyUI/blob/master/comfy_extras/nodes_custom_sampler.py>
- Official ComfyUI `SamplerCustomAdvanced` node documentation:
  <https://docs.comfy.org/built-in-nodes/SamplerCustomAdvanced>

## What KSampler Advanced does

`KSamplerAdvanced` is an all-in-one wrapper. It creates or suppresses initial
noise, constructs the scheduler curve through Comfy's `KSampler`, applies
`start_at_step` and `end_at_step`, and optionally forces the last sigma to zero
when `return_with_leftover_noise` is disabled.

For a legacy two-KSampler high/low Wan branch:

```text
High KSampler:
  add_noise = enable
  start_at_step = 0
  end_at_step = split
  return_with_leftover_noise = enable

Low KSampler:
  add_noise = disable
  start_at_step = split
  end_at_step >= steps
  return_with_leftover_noise = disable
```

The effective sigma slices are:

```text
high sigmas = full_sigmas[:split + 1]
low sigmas  = full_sigmas[split:]
```

The handoff sigma is intentionally present in both slices. Removing it from the
low slice, or forcing the high slice to terminal zero, does not match KSampler
Advanced continuation semantics.

## What SamplerCustomAdvanced does

`SamplerCustomAdvanced` is lower level. It receives an already-built `NOISE`,
`GUIDER`, `SAMPLER`, `SIGMAS`, and `LATENT`, then calls the guider directly with
those objects. That gives the planner exact curve control, but it means the
workflow must recreate the parts KSampler Advanced used to own:

- high stage should use `RandomNoise`;
- low stage should use `DisableNoise`;
- low stage must receive the high stage `output`, not `denoised_output`;
- each stage should use `CFGGuider` for KSampler parity, not `BasicGuider`;
- the guider models must be the plan-shifted high/low expert models;
- the sampler object should come from `KSamplerSelect` with the same sampler
  name used by the legacy KSampler branch;
- the sigmas should come directly from Sampling Planner's Sigma Breakout.

`denoised_output` is an x0 estimate. It is useful for inspection, but it is not
the partially sampled latent at the shared Wan expert handoff sigma.

## Design conclusion

When the plan reports `curve_mode: exact`, prefer KSampler Breakout for
regression testing against known-good two-KSampler workflows. This keeps native
ComfyUI in charge of the schedule slicing and leftover-noise behavior.

Use Sigma Breakout with `SamplerCustomAdvanced` when the plan is piecewise,
mixed-budget, priority-adjusted, or otherwise cannot be represented by ordinary
KSampler Advanced controls.

Sampler-specific caution: Comfy's internal `KSampler.calculate_sigmas` has
special count handling for a small set of samplers such as `dpm_2`,
`dpm_2_ancestral`, `uni_pc`, and `uni_pc_bh2`. The KSampler Breakout node checks
for this and rejects incompatible exact-curve routes. For custom-sampler routes,
Sigma Breakout remains authoritative.
