# Wan 2.2 sigma and shift evidence

Retrieved: 2026-06-23

This note records the evidence used by Sampling Planner's Wan 2.2 `Auto`
profiles. It separates official model settings, official ComfyUI workflow
conventions, community implementations, and project-specific empirical results.

## What the Wan graph establishes

**Official Wan source.** The [Wan 2.2 architecture article](https://wan.video/blog/wan2.2)
describes the model's high-noise and low-noise experts in signal-to-noise terms.
The expert boundary belongs to the trained MoE architecture. The article's graph
does not specify one universally optimal inference shift or require a sampled
sigma to equal the boundary.

That last sentence is a Sampling Planner interpretation of the graph, not a
quoted Wan recommendation. Shift changes the distribution of inference
evaluations along the denoising trajectory; it does not move the trained expert
boundary.

Source revision: web page, no repository commit exposed.

## Evidence record

### Official Wan model configuration

**Classification:** Official model configuration.

At commit [`fa96d0963f1aef3b0f9dd312296d54b13e2da538`](https://github.com/Wan-Video/Wan2.2/commit/fa96d0963f1aef3b0f9dd312296d54b13e2da538):

- [T2V A14B configuration](https://github.com/Wan-Video/Wan2.2/blob/fa96d0963f1aef3b0f9dd312296d54b13e2da538/wan/configs/wan_t2v_A14B.py)
  sets `sample_shift = 12.0`, `sample_steps = 40`, and `boundary = 0.875`.
- [I2V A14B configuration](https://github.com/Wan-Video/Wan2.2/blob/fa96d0963f1aef3b0f9dd312296d54b13e2da538/wan/configs/wan_i2v_A14B.py)
  sets `sample_shift = 5.0`, `sample_steps = 40`, and `boundary = 0.900`.

These are the authoritative native task defaults. They establish that shift is
task- and recipe-dependent even though both tasks use the same split-expert
architecture.

### Official ComfyUI standard workflows

**Classification:** Official ComfyUI workflow convention.

At commit [`40fc612cdf058af9440a9262ef293002832cf9ea`](https://github.com/Comfy-Org/workflow_templates/commit/40fc612cdf058af9440a9262ef293002832cf9ea),
the official [T2V](https://github.com/Comfy-Org/workflow_templates/blob/40fc612cdf058af9440a9262ef293002832cf9ea/templates/video_wan2_2_14B_t2v.json)
and [I2V](https://github.com/Comfy-Org/workflow_templates/blob/40fc612cdf058af9440a9262ef293002832cf9ea/templates/video_wan2_2_14B_i2v.json)
templates use:

- Shift `8` on both expert models.
- `20` Euler/simple steps.
- A `10` high-noise / `10` low-noise split.

This supports shift 8 as an established ComfyUI profile. It does not establish
shift 8 as a universal mathematical optimum.

### Official ComfyUI LightX2V workflows

**Classification:** Official ComfyUI accelerated-workflow convention.

Commit [`7a924ade29e2dd140ef3ddeb21ca915b9f635f4f`](https://github.com/Comfy-Org/workflow_templates/commit/7a924ade29e2dd140ef3ddeb21ca915b9f635f4f)
added LightX2V branches to the official [T2V](https://github.com/Comfy-Org/workflow_templates/blob/7a924ade29e2dd140ef3ddeb21ca915b9f635f4f/templates/video_wan2_2_14B_t2v.json)
and [I2V](https://github.com/Comfy-Org/workflow_templates/blob/7a924ade29e2dd140ef3ddeb21ca915b9f635f4f/templates/video_wan2_2_14B_i2v.json)
templates. Their accelerated branches use:

- Shift `5` on both expert models.
- `4` Euler/simple steps.
- A `2` high-noise / `2` low-noise split.
- CFG `1` and LoRA strength `1`.

The same templates retain their 20-step, shift-8 standard branches. This is
direct evidence that acceleration recipe and step budget can select a different
valid inference profile without changing the model's trained boundary.

### Official LightX2V acceleration model pairing

**Classification:** Official LightX2V model documentation.

At Hugging Face revision
[`570044187a5219776ef30a5c60c6f76428a3a10a`](https://huggingface.co/lightx2v/Wan2.2-Distill-Loras/tree/570044187a5219776ef30a5c60c6f76428a3a10a),
the [Wan2.2 Distill LoRA model card](https://huggingface.co/lightx2v/Wan2.2-Distill-Loras/blob/570044187a5219776ef30a5c60c6f76428a3a10a/README.md)
documents a matched pair of rank-64 I2V LoRAs:

- `wan2.2_i2v_A14b_high_noise_lora_rank64_lightx2v_4step_xxx.safetensors`
- `wan2.2_i2v_A14b_low_noise_lora_rank64_lightx2v_4step_xxx.safetensors`

Its merge examples use LoRA strength `1.0` for both experts. The separate
`Wan2.2-Distill-Models` repository contains full distilled checkpoints with
similar names but without `_lora_rank64`; those are model checkpoints, not LoRA
inputs.

At Hugging Face revision
[`18bccf8884ec0a078eed79785eb4ef13ea16ce1e`](https://huggingface.co/lightx2v/Wan2.2-Lightning/tree/18bccf8884ec0a078eed79785eb4ef13ea16ce1e),
LightX2V's [native ComfyUI I2V workflow](https://huggingface.co/lightx2v/Wan2.2-Lightning/blob/18bccf8884ec0a078eed79785eb4ef13ea16ce1e/Wan2.2-I2V-A14B-4steps-lora-rank64-Seko-V1/Wan2.2-I2V-A14B-4steps-lora-rank64-Seko-V1-NativeComfy.json)
uses both expert LoRAs at strength `1`, shift `5`, four Euler/simple steps, and a
2/2 split.

This evidence constrains the planner's promise: sigma validation can guarantee a
well-formed curve, but it cannot make an incompatible or over-strength
acceleration model pair numerically stable. Extended step profiles must begin
from a matched vendor-supported model pair.

### WanMoeKSampler

**Classification:** Community implementation; corroborating evidence, not an
official Wan or ComfyUI default.

At commit [`e5b2576c73d02f5991711275d1c202828133c035`](https://github.com/stduhpf/ComfyUI-WanMoeKSampler/commit/e5b2576c73d02f5991711275d1c202828133c035),
[`nodes.py`](https://github.com/stduhpf/ComfyUI-WanMoeKSampler/blob/e5b2576c73d02f5991711275d1c202828133c035/nodes.py)
defaults `sigma_shift` to `8.0`, documents boundaries `0.875` for T2V and `0.9`
for I2V, and chooses the expert transition by mapping the scheduler's sigmas
back to diffusion timesteps.

This corroborates the distinction between denoising-step position and the
trained diffusion-timestep boundary. Its pinned [T2V example](https://github.com/stduhpf/ComfyUI-WanMoeKSampler/blob/e5b2576c73d02f5991711275d1c202828133c035/workflows/Wan%20MoE%20T2V.json)
uses shift `12`, while its [I2V example](https://github.com/stduhpf/ComfyUI-WanMoeKSampler/blob/e5b2576c73d02f5991711275d1c202828133c035/workflows/Wan%20MoE%20I2V.json)
uses shift `5`. The examples therefore demonstrate that native task shifts and
the node's community default are both used.

### YAW validation anchor

**Classification:** Sampling Planner/YAW empirical evidence.

The project has a validated accelerated operating point of:

- I2V, High + Low acceleration.
- `10` total Euler/simple steps.
- `5` high-noise / `5` low-noise steps.
- Shift `8`.

The local reference artifact is
`workflows/YAW_2.2_T2V+I2V_v0_39.json` from the adjacent `comfyui-wan`
workspace, SHA-256
`bbda6edcd004d26ef090074f44361d8845f2a8069e5cfe8f1b89e73d4a856137`.
It is untracked in that workspace and therefore has no public URL or Git commit.
The operating point is recorded as project testing evidence, not an upstream
recommendation.

## Planner policy derived from the evidence

`Auto` uses profile anchors instead of searching for a shift that makes one
sample equal the expert boundary:

| Configuration | Profile | Shift anchor | Allocation |
|---|---|---:|---|
| Unaccelerated T2V | Wan Native | 12 | Scheduler boundary crossing |
| Unaccelerated I2V | Wan Native | 5 | Scheduler boundary crossing |
| Acceleration active, accelerated budget ≤ 4 | LightX2V 4-Step | 5 | 50/50 |
| Acceleration active, accelerated budget > 4 | ComfyUI / YAW | 8 | Scheduler boundary crossing |

The boundary remains fixed at `0.875` for T2V and `0.900` for I2V. A valid
discrete handoff straddles that boundary; the planner does not require exact
equality. Manual Shift Override remains available and rebuilds both model
patching and sigma outputs together.

The resulting policy is deliberately modest: these profiles are defensible
baselines supported by published configurations and known-good workflows. They
are not a claim that quality can be maximized from the MoE boundary alone.
