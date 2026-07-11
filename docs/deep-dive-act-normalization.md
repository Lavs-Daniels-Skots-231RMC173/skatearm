# Debugging deep-dive — the ACT policy that reached for garbage

> A short, honest bug story from the [ACT visuomotor pipeline](../README.md).
> The training curve was textbook. The policy still failed. The bug wasn't in the model —
> it was in the contract between **training** and **inference**.

## TL;DR

| | |
|---|---|
| **Symptom** | Trained ACT hit a clean **L1 ≈ 0.070**, but closed-loop rollout drove the arms **~0.65 m** from the targets — *worse than not moving at all*. |
| **Wrong turn** | I blamed the VAE first. A red herring. |
| **Root cause** | In `lerobot` 0.6.0, ACT has **no built-in normalization** — `select_action` expects *normalized* input and returns *normalized* output. I fed it raw radians + raw pixels and applied its raw output to the joints. |
| **Fix** | Wrap inference in the LeRobot pre/post processors: `post(policy.select_action(pre(obs)))`. |
| **Result** | Mean reach error **0.65 m → ~5 cm**. |
| **Lesson** | A perfect loss curve only proves the net learned the mapping *in normalized space*. When training is clean but inference fails, suspect the **plumbing**, not the weights. |

## The setup

A scripted damped-least-squares IK "expert" glides both arms to random reachable targets in the
MuJoCo twin; those episodes become a LeRobotDataset v3.0; an ACT policy (ResNet-18 + Transformer,
~52 M params) is behaviour-cloned from them on a 4 GB laptop GPU. At inference the policy only ever
sees the rendered camera frame + the 14-DoF joint state — never the target coordinates.

Training went beautifully. L1 action loss fell from ~0.78 to **0.070** over 20k steps; the curve was
clean and monotonic. By every training-time signal, the policy had learned the task.

## The symptom

Then I rolled it out closed-loop — each step: render the frame, feed the policy, apply its action,
step the sim, repeat. The arms didn't reach. They drifted to a fixed, wrong pose and sat there.
Final end-effector error averaged **~0.65 m** — literally *worse than doing nothing*, since the home
pose is already closer to the targets than that.

A model with 0.070 training loss should not miss by two thirds of a metre. Something between
"training" and "inference" was broken.

## Wrong turn: blaming the VAE

ACT is a CVAE — at training it encodes the action chunk into a latent; at inference the latent is set
to zero (the prior mean). My first instinct was the fashionable suspect: *the VAE latent is degenerate
on such a small dataset, so the decoder produces junk at latent = 0.*

So I retrained with `--policy.use_vae=false`. That's a defensible choice for a small, deterministic
reach — a deterministic policy removes the train/inference latent gap. But it **did not fix the reach
error.** The arms still missed by half a metre. The VAE was a red herring; I'd chased the
interesting-looking suspect instead of the boring one.

## Finding the real bug: the model was fine all along

Before touching anything else, I wrote a short check: load the trained policy, take real frames from
the dataset, and compare the policy's predicted action to the ground-truth action — **using the same
processors the training entrypoint builds.** Predicted-vs-ground-truth L1 came back at **0.02–0.04**.
The network had learned the mapping essentially perfectly.

That was the turning point. If the policy predicts correctly *when called the way training calls it*,
the weights are not the problem — my rollout loop was calling it a **different** way.

## Root cause: normalization moved out of the policy

Older LeRobot normalized observations and un-normalized actions **inside** the policy `forward`. In
`lerobot` 0.6.0 that was refactored out: normalization now lives in separate **processor** objects
built by `make_pre_post_processors(...)`. The ACT `nn.Module` itself is a pure *normalized → normalized*
function. It assumes:

- observations arrive already normalized (state standardized to the dataset mean/std, images to
  ImageNet stats), and
- its output is still in normalized action space, to be un-normalized by the post-processor.

My rollout did neither. I passed raw joint angles and raw `0–255` pixels straight in, and wrote the raw
output straight to `d.qpos`. The network saw out-of-distribution inputs and I mis-scaled its outputs —
so of course the arms flew to a fixed, meaningless pose. And it failed **silently**: no error, no NaN,
just confidently wrong motion.

```python
# WRONG — raw obs in, raw action out (silently wrong)
action = policy.select_action(obs)          # obs never normalized; action still in normalized space
d.qpos[ARM_IDX] = action.cpu().numpy()      # applying normalized numbers as if they were radians

# RIGHT — normalize in, un-normalize out
pre, post = make_pre_post_processors(
    policy_cfg=policy.config,
    pretrained_path=CKPT,
    preprocessor_overrides={"device_processor": {"device": device}},
)
with torch.no_grad():
    action = post(policy.select_action(pre(obs)))   # pre normalizes obs, post un-normalizes action
d.qpos[ARM_IDX] = action.squeeze(0).cpu().numpy()
```

The entire bug was the missing `pre(...)` and `post(...)`.

## Result

| Rollout (16 unseen episodes) | Before | After |
|---|---|---|
| Mean reach error (right / left) | ~0.65 m | **5.1 / 5.2 cm** |
| Both hands within 8 cm | 0 % | **75 %** |

Same weights, same checkpoint. The only change was calling the policy through the processors it was
trained with.

## Why it's easy to miss

- **It fails silently.** Un-normalized inputs don't raise — they just push the network off its training
  distribution and the outputs are quietly wrong.
- **The contract is implicit.** The training CLI builds the processors for you, so if you write your own
  rollout loop you have to *know* to rebuild them. Nothing forces you to.
- **The training curve lies to you** — in the good way. It only certifies the mapping in normalized
  space, which is not the space your raw sim observations live in.

## Lesson

When a model trains cleanly but fails at inference, don't start with the model. Start with the
**train/inference contract**: normalization, image stats, device, dtype. The fastest triage is to score
the trained policy against ground truth *with the training-time processors* — if that's good (here
0.02–0.04), the weights are fine and the bug is in your plumbing. It cost me a wrong VAE detour to
relearn a boring truth: **most "the model is broken" bugs are actually "I called the model wrong" bugs.**

---

*Code: [`tools/skate_commander/examples/act_reach/rollout_act.py`](../tools/skate_commander/examples/act_reach/rollout_act.py) · pipeline overview in the [main README](../README.md).*
