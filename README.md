# ComfyUI Sampling Planner

Task-aware sampling controls for complex ComfyUI workflows.

The first release focuses on Wan 2.2 split-expert workflows. It converts a small
set of user-facing choices into coordinated shift, step split, and sigma outputs.

The planner uses evidence-backed sampling profiles rather than treating one shift
as universally optimal. See [Wan 2.2 sigma and shift evidence](docs/wan22-sigma-evidence.md)
for the source record and design rationale.

## Wan 2.2 nodes

### Scheduler Selector

This helper reads the live scheduler combo used by the installed core KSampler.
Its output can fan out to:

- the Sampling Plan (Wan 2.2) scheduler input
- the high KSampler scheduler input
- the low KSampler scheduler input
- additional compatible KSampler branches

Scheduler names registered into the core KSampler list by installed extensions
are included automatically when ComfyUI builds the node schema.

Custom scheduler *nodes* that directly output `SIGMAS` are a different ComfyUI
interface and do not have names that can be passed into a KSampler scheduler
combo. Use those through the Sigma Breakout/custom-sampling path instead.

### Sampler Selector

Reads the live sampler combo used by the installed core KSampler. Its output can
fan out to the high and low KSampler `sampler_name` inputs and other compatible
core sampler controls.

Specialized sampler nodes such as Clownshark expose their own algorithm lists;
those are separate registries and are not included in this combo.

### Task Selector

Outputs a task combo compatible with Sampling Plan (Wan 2.2):

- `T2V`
- `I2V`

This is useful when the same workflow mode switch also controls latent/image
inputs and other task-dependent branches.

### Priority Selector

Outputs a priority combo compatible with Sampling Plan (Wan 2.2):

- `Balanced`
- `50/50 Split`
- `Motion / Structure`
- `Detail / Refinement`

### Step Budget

Defines two full-range-equivalent budgets:

- `accelerated_steps`: steps used if the entire denoising range were accelerated
- `full_steps`: steps used if the entire range were unaccelerated

The planner projects the appropriate budget onto each expert's share of the
denoising range. With 10 accelerated steps, 30 full steps, and a 50/50 range
split:

| Acceleration | High stage | Low stage | Effective total |
|---|---:|---:|---:|
| High + Low | 5 accelerated | 5 accelerated | 10 |
| High only | 5 accelerated | 15 full | 20 |
| Low only | 15 full | 5 accelerated | 20 |
| None | 15 full | 15 full | 30 |

### Acceleration Selector

Uses one compact mode:

- `None`
- `High only`
- `Low only`
- `High + Low`

It acts like the other compact selector helpers: one field and one combo output.
Connect that output to the Sampling Plan acceleration input.

### Acceleration Model Pair

This is the recommended replacement for manually bypassing acceleration groups.
It accepts:

- base high model
- accelerated high model
- base low model
- accelerated low model
- base CFG
- accelerated CFG
- bundled acceleration state from Sampling Plan

It lazily requests only the selected model branches. With `None`, neither
acceleration LoRA branch executes. With `High only`, only the accelerated high
branch executes; the low branch remains base.

The same state routes CFG in lockstep with the models:

- unaccelerated stage → `base_cfg`
- accelerated stage → `accelerated_cfg`

The outputs are `cfg_high` and `cfg_low`, ready to connect to the corresponding
samplers. This prevents a base expert from accidentally receiving accelerated
CFG—or the reverse.

Recommended placement:

```text
base high ───────────────┐
base high → accel LoRA ──┤
                         ├─ Acceleration Model Pair → selected high
base low ────────────────┤
base low → accel LoRA ───┘                         → selected low
```

Keep the acceleration groups enabled. Lazy routing prevents unused LoRA branches
from running, so no group bypass synchronization is required.

### Sampling Plan (Wan 2.2)

Managed controls:

- **Task:** T2V or I2V
- **Acceleration:** None, High only, Low only, or High + Low
- **Step Budget:** accelerated and full-range-equivalent steps
- **Scheduler:** ComfyUI sigma scheduler
- **Priority:** Balanced, 50/50, Motion / Structure, or Detail / Refinement

The connected `MODEL` is used to calculate the real ComfyUI sigma schedule. It
can be either Wan expert model.

The planner outputs both the complete sampling plan and an
`acceleration_state` breakout. Connect the latter to Acceleration Model Pair.

The planner uses the official Wan 2.2 boundaries:

- T2V: `0.875`
- I2V: `0.900`

Priority determines the high/low share of the denoising range. Acceleration
determines which budget is projected onto each share. The resulting high and low
step counts are added to produce the effective total schedule.

`Auto` selects an evidence-backed curve profile:

| Configuration | Profile | Shift anchor | Default split policy |
|---|---|---:|---|
| No acceleration, T2V | Wan Native | 12 | Scheduler boundary crossing |
| No acceleration, I2V | Wan Native | 5 | Scheduler boundary crossing |
| Acceleration active, accelerated budget ≤ 4 | LightX2V 4-Step | 5 | 50/50 |
| Acceleration active, accelerated budget > 4 | ComfyUI / YAW | 8 | Scheduler boundary crossing |

The accelerated profile assumes that the selected high- and low-noise models
form a compatible pair. LightX2V's official 4-step LoRA recipes use rank-64
LoRAs for both experts at strength `1.0`. Do not send a full distilled-model
checkpoint through a LoRA loader, and treat strengths above the vendor baseline
as an independent quality variable. The planner can validate the denoising
curve, but an opaque ComfyUI `MODEL` input does not expose enough provenance to
prove that the selected acceleration files are compatible.

The shift anchor controls where the scheduler concentrates its evaluations; it
does not replace or redefine the task's expert boundary. The planner accepts a
discrete handoff that straddles the boundary and does not force one sigma to
equal it. A bounded shift adjustment is used only when the anchored curve cannot
represent a valid crossing. Otherwise the planner preserves the anchor and
constructs a boundary-safe piecewise curve.

### Step Split Override

Adds one managed field: `steps_high`.

Set `steps_high` to `0` to pass the plan through unchanged. This is the
recommended way to leave the node visibly in-chain without applying a manual
split.

Set `steps_high` to a positive value to replace Priority's calculated high-stage
transition count while preserving the plan's effective total. The complete curve
and both sigma slices are rebuilt and validated.

### Progressive 50/50 Sigma Control

No widgets. This is a branch-state signal for progressive transcode/upscale
workflows. Put it inside the low-res → high-res group. When that group is
active, the node tells Sampling Plan to use accelerated 50/50 sigmas; when the
group is muted, the optional signal disappears and Sampling Plan falls back to
normal Low-only accounting.

It connects to Sampling Plan's optional `sigma_override` input; it is not a
pass-through node in the required plan chain. That means the progressive group
can be muted without severing the `WAN22_SAMPLING_PLAN` wire.

Use it with Sampling Plan when:

- Acceleration is `Low only`
- you still want model/CFG routing to remain base high → accelerated low
- but you want the sigma curve to use the accelerated budget split 50/50

Example with accelerated/full budgets `A10/F30`:

| Plan | High steps | Low steps | Model routing |
|---|---:|---:|---|
| Low only + 50/50 priority | 15 | 5 | base high → accelerated low |
| + Progressive 50/50 Sigma Control | 5 | 5 | base high → accelerated low |

Sampling Plan intentionally rejects this control for other acceleration modes.
The control does not manage decode/upscale/re-encode routing or noise routing;
keep those visible in the workflow where the progressive technique actually
lives.

### Shift Override

Adds one managed field: `shift`.

This replaces the Auto profile's shift anchor. It rebuilds the complete plan,
including full sigmas and high/low sigma slices. Downstream Model Pair Breakout
then applies the rebuilt plan's shift to both models, so model shift and sampling
curve cannot diverge.

Step Split Override and Shift Override are order-independent. Each stores its
requested override in the plan and rebuilds from the original planner controls.
Progressive 50/50 Sigma Control is different: it should feed Sampling Plan's
optional `sigma_override` input, before any downstream Step Split or Shift
overrides.

For progressive-upscale style workflows, a clear ordering is:

```text
Progressive 50/50 Sigma Control ─┐
                                  ├→ Sampling Plan → Step Split Override → Sigma Breakout
normal planner controls ──────────┘
```

With that order, `steps_high = 0` keeps the accelerated 50/50 sigma plan, while
any positive `steps_high` value manually overrides the split within that
accelerated total.

### Model Pair Breakout

Accepts the plan plus the selected high- and low-noise expert models. It clones
and patches both models with the plan's exact shift, then outputs the synchronized
model pair.

Use this after Acceleration Model Pair and before guiders or KSamplers:

```text
Acceleration Model Pair → Model Pair Breakout → high/low samplers
Sampling Plan ────────────┘
```

Model Pair Breakout has no user-facing widgets. Shift is owned by the plan,
including any Shift Override.

### KSampler Breakout

Connect the plan, high/low expert models, and the sampler selected for the
KSampler branch. The node exposes compatible KSampler Advanced controls:

- total steps
- high `end_at_step`
- low `start_at_step`
- high and low step counts
- shift

For a standard two-KSampler Advanced branch:

```text
model_high      -> High KSampler model
model_low       -> Low KSampler model
steps           -> both KSampler steps
high_end_step   -> High KSampler end_at_step
low_start_step  -> Low KSampler start_at_step
```

Keep the scheduler selected in both KSamplers the same as the Sampling Plan.
The Scheduler Selector is intended to drive all three scheduler inputs, and the
Sampler Selector should drive the KSamplers and KSampler Breakout.

KSampler Breakout is available only when core KSampler can reproduce the exact
planned curve from the selected sampler, scheduler, step count, and shift. Plans
using piecewise-resampled sigmas are not KSampler-representable; the node reports
that incompatibility and directs the workflow to Sigma Breakout instead of
silently approximating the curve.

For parity testing against older Wan 2.2 workflows, prefer this KSampler
Breakout path whenever the plan reports `curve_mode: exact`. It lets ComfyUI's
native KSampler Advanced implementation own `start_at_step`, `end_at_step`, and
`return_with_leftover_noise`, which is the closest match to legacy two-KSampler
branches.

### MoE Sampler Breakout

Adapts the plan for single-node MoE samplers such as `WanMoeKSampler`
(stduhpf/ComfyUI-WanMoeKSampler). Connect the plan and the high/low expert
models (typically from Acceleration Model Pair):

```text
model_high -> MoE sampler model_high_noise (patched with the plan's shift)
model_low  -> MoE sampler model_low_noise  (patched with the plan's shift)
steps      -> MoE sampler steps
shift      -> MoE sampler sigma_shift
boundary   -> MoE sampler boundary (the plan's task boundary: T2V 0.875 / I2V 0.900)
```

Drive the MoE sampler's `sampler_name` and `scheduler` inputs from the same
Sampler Selector and Scheduler Selector that feed the Sampling Plan so the
sampler's internally generated curve matches the planned curve.

Unlike KSampler Breakout, this node never rejects a plan: the MoE sampler
rebuilds a native scheduler curve from steps and shift and hands off experts
where that curve crosses the boundary sigma, so exact representability is not
required. Because the planner solves shift against the same task boundary, an
`exact` plan's handoff lands on the planned split step; for `piecewise` plans
the curve is approximated best-effort at the planned shift (the split may land
a step away from the plan's allocation) and the node's summary notes the
approximation. Step Split Override and priority choices influence this path
only through the solved shift.

### Sigma Breakout

Exposes:

- high sigmas
- low sigmas
- complete sigmas
- total/high/low step counts
- shift

The high schedule ends at the same sigma where the low schedule begins.

Sigma Breakout is the authoritative sigma-output path for every plan. It
preserves the exact validated curve, including mixed-budget and
priority-adjusted piecewise curves that ordinary KSampler controls cannot
express. Use it when KSampler Breakout reports that the plan cannot be
represented by native KSampler Advanced controls.

For `SamplerCustomAdvanced`, continue the two expert stages as follows:

```text
RandomNoise → high SamplerCustomAdvanced → output → low SamplerCustomAdvanced → output
                                                   ↑
                                              DisableNoise
```

Use the high sampler's `output`, not `denoised_output`, as the low-stage latent.
`denoised_output` is an x0 prediction intended for preview or inspection, not
the partially sampled latent at the shared handoff sigma.

For KSampler parity, use `CFGGuider` for each custom-sampler stage, feed the
same positive/negative conditioning and CFG values that the KSampler Advanced
nodes used, use the same `KSamplerSelect` sampler, and do not insert a separate
`BasicScheduler`/scheduler node between the plan and the samplers. Sigma
Breakout already emits the high and low sigma slices.

See [Comfy sampler parity notes](docs/comfy-sampler-parity.md) for the source
comparison between KSampler Advanced and SamplerCustomAdvanced.

## Priority behaviour

- **Balanced:** profile allocation.
- **50/50 Split:** equal high/low range allocation. Executed step counts may
  differ when the stages use different budgets.
- **Motion / Structure:** moves roughly 10% of the range to the high-noise expert.
- **Detail / Refinement:** moves roughly 10% of the range to low-noise refinement.

At least one step is preserved for each expert.

## Scheduler notes

Some ComfyUI schedulers respond very little to ModelSamplingSD3 shift. The plan
validates each generated curve for finite, nonnegative, descending sigmas, a
terminal zero, usable stage lengths, and a shared handoff. Requested scheduler
steps and actual sigma transitions are tracked separately for schedulers whose
output cardinality differs from `steps + 1`.

When an installed scheduler cannot produce a safe anchored crossing, the planner
uses a validated piecewise curve or reports a descriptive error. It does not
emit an extreme shift or a malformed schedule.

## Installation

Place this directory under `ComfyUI/custom_nodes` and restart ComfyUI.

The nodes appear under:

```text
sampling / Sampling Planner / Wan 2.2
```

## Development tests

The planner tests do not require ComfyUI or PyTorch:

```bash
python3 -m unittest discover -s tests -v
```
