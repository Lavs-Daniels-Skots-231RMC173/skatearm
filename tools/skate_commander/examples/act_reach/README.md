# ACT reach example — cockpit → policy

Scripts behind the **ACT visuomotor pipeline** section of the top-level
[`README.md`](../../../../README.md#act-pipeline): render a bimanual-reach
**LeRobotDataset v3.0** from the Skate Commander MuJoCo twin, train an **ACT**
policy, and roll it out closed-loop.

## Requirements

- The `skt_v3` model directory — set `SKT_DIR`, or place it at `../../skate_teleop/skt_v3`.
- `pip install 'lerobot[training]' mujoco pillow` and a CUDA GPU for training.
- Offscreen render backend: `MUJOCO_GL=osmesa` (CPU) or `egl`.

## Usage (run from this directory)

```bash
# 1 · dataset  ->  ../../lerobot_datasets/reach_act
MUJOCO_GL=osmesa python gen_reach_dataset.py 40 256

# 2 · train ACT (deterministic, batch 4, 20k steps, ~32 min on an RTX 3050)
lerobot-train \
  --dataset.repo_id=skate/reach_act \
  --dataset.root=../../lerobot_datasets/reach_act \
  --policy.type=act --policy.use_vae=false --policy.device=cuda \
  --batch_size=4 --steps=20000 --save_freq=5000 \
  --output_dir=../../act_reach

# 3 · rollout (CPU inference is fine)
MUJOCO_GL=osmesa python rollout_act.py ../../act_reach/checkpoints/020000/pretrained_model 6

# 4 · reproduce the README's headline table + chart from the committed per-seed rollouts
python aggregate_reach.py eval_data/seed*.json eval_data/baseline.json
#   regenerate from scratch: train seeds with --seed 0/1/2 --output_dir=../../act_seedN,
#   rollout each with ACT_TMP=eval/seedN, then  ACT_TMP=eval/base python baseline_reach.py 24
```

Env overrides: `SKT_DIR` (model dir), `REACH_DATASET` (dataset out), `ACT_TMP` (scratch / GIF out).

## Notes

- **Inference must normalize** — wrap the policy with `make_pre_post_processors(...)`:
  `preprocessor(obs) → select_action → postprocessor(action)`. Skip it and the model
  receives un-normalized inputs and the arms go to garbage.
- **`use_vae=false` is deliberate** — a deterministic policy fits the deterministic reach and
  avoids the VAE train/inference gap on a small (40-demo) dataset.
- **The headline is reproducible in-repo** — `baseline_reach.py` (no-vision baseline) and
  `aggregate_reach.py` rebuild the mean ± std table and the accuracy chart from the per-seed
  rollouts; `eval_data/` holds the raw 3-seed + baseline JSON.
- **The rollout is a *kinematic* position-replay** — each predicted pose is applied via forward
  kinematics (`d.qpos` + `mj_forward`), not actuator dynamics or contact, so the eval validates
  the in-distribution visuomotor mapping, not dynamic control or sim-to-real.
- Training outputs (`../../act_reach/`, `../../act_smoke/`) and the dataset
  (`../../lerobot_datasets/`) are git-ignored.
