# muscular-skeleton-system


This repository contains the MuJoCo simulation model and Python scripts for a
pneumatic-muscle-driven musculoskeletal arm. The project focuses on structural
design, switched-state pneumatic muscle modeling, motion range analysis, and a
preliminary Gymnasium/MuJoCo environment for future reinforcement learning
control.

The most important handover point is that the XML files mainly define the
geometry, tendons, joints, and actuator interfaces. The actual pneumatic muscle
force is computed in Python by a custom switched-state model, then applied to
MuJoCo through `data.qfrc_applied`.

## Project Overview

The simulated system is inspired by a human arm:

- `anchor_ring`: fixed external ring around the shoulder region.
- `rod`: upper arm segment, connected through a proximal ball joint.
- `elbow`: elbow joint body.
- `forearm`: lower arm segment, connected through a hinge joint.
- `forearm_tip`: distal reference body used for endpoint tracking.
- 41 muscle/tendon actuators in total:
  - `muscle1` to `muscle25`: shoulder/upper-arm muscle group.
  - `elbow_front_flexor1` to `elbow_front_flexor8`: anterior elbow group.
  - `elbow_behind_flexor1` to `elbow_behind_flexor8`: posterior elbow group.

The current recommended model/script pair is:

```text
system_model_n=6_010.xml
view_switched_state_mujoco_n=6_010.py
```

This pair corresponds to the latest working full-arm visualization script in the
repository and is the best starting point for a new user.

## Repository Structure

```text
.
├── README.md
├── custom_env_v4.py
├── system_model_n=0.xml
├── system_model_n=4.xml
├── system_model_n=4_008.xml
├── system_model_n=4_010.xml
├── system_model_n=6_010.xml
├── system_model_n=6_012.xml
├── system_model_n=8.xml
├── view_switched_state_mujoco_n=0.py
├── view_switched_state_mujoco_n=4.py
├── view_switched_state_mujoco_n=4_008.py
├── view_switched_state_mujoco_n=4_010.py
├── view_switched_state_mujoco_n=6_010.py
├── view_switched_state_mujoco_n=6_012.py
└── view_switched_state_mujoco_n=8.py
```

## Environment Setup

Recommended: Python 3.9 or newer.

```bash
python -m venv .venv
source .venv/bin/activate
pip install mujoco numpy imageio gymnasium
```

On headless Linux servers, MuJoCo rendering may require:

```bash
export MUJOCO_GL=egl
```

## Quick Start

```bash
git clone https://github.com/twanblm03/muscular-skeleton-system.git
cd muscular-skeleton-system
```

List all actuators:

```bash
python view_switched_state_mujoco_n=6_010.py --list-actuators
```

Run a single-muscle activation demo:

```bash
python view_switched_state_mujoco_n=6_010.py --activate muscle1
```

Run grouped activation tests:

```bash
python view_switched_state_mujoco_n=6_010.py --activate upper_front
python view_switched_state_mujoco_n=6_010.py --activate upper_back
python view_switched_state_mujoco_n=6_010.py --activate lower_front
python view_switched_state_mujoco_n=6_010.py --activate lower_back
```

Save the final simulation image:

```bash
python view_switched_state_mujoco_n=6_010.py \
  --activate upper_back \
  --save-final-image \
  --final-image-path final_state.png
```

## Viewer Script Arguments

| Argument | Meaning |
| --- | --- |
| `--xml` | Path to the MuJoCo XML model. Defaults to `system_model_n=6_010.xml`. |
| `--activate` | Comma-separated actuator names, actuator indices, or preset group names. |
| `--pre-steps` | Initial steps with all muscles inactive. Default: `800`. |
| `--total-steps` | Total simulation steps. Default: `8000`. |
| `--render-every` | Viewer synchronization interval. |
| `--print-every` | Console status print interval. |
| `--tracked-body` | Body whose world position is printed. Default: `elbow`. |
| `--active-ctrl` | Control value applied to active muscles after `pre-steps`. Default: `1.0`. |
| `--no-realtime-sleep` | Disable real-time sleeping and run faster. |
| `--list-actuators` | Print actuator indices and names, then exit. |
| `--save-final-image` | Save a PNG of the final MuJoCo state. |
| `--final-image-path` | Output path for the final PNG. |

Available activation presets:

| Preset | Actuators |
| --- | --- |
| `upper_front` | `muscle1` to `muscle16` |
| `upper_back` | `muscle17` to `muscle25` |
| `lower_front` | `elbow_front_flexor1` to `elbow_front_flexor8` |
| `lower_back` | `elbow_behind_flexor1` to `elbow_behind_flexor8` |

Manual examples:

```bash
python view_switched_state_mujoco_n=6_010.py --activate muscle1,muscle16
python view_switched_state_mujoco_n=6_010.py --activate 0,15
```

## Model Details

The XML model defines:

- fixed anchor ring and fixed anchor sites;
- small connector bodies under anchor sites, each with a limited ball joint;
- upper-arm body `rod`;
- shoulder-like ball joint `rod_joint`;
- elbow body `elbow`;
- hinge joint `elbow_flex`;
- forearm body and `forearm_tip`;
- spatial tendons for all muscle paths;
- MuJoCo muscle actuator interfaces with `ctrlrange="0 1"`.

The XML actuator force is intentionally set to zero:

```xml
<muscle force="0" timeconst="0.1 0.1" scale="1"/>
```

This is because the built-in MuJoCo muscle actuator is used only as an interface
for geometry, tendon length, velocity, and moment-arm information. The true
pneumatic muscle mechanics are handled in Python.

## Switched-State Pneumatic Muscle Model

Each muscle has two states:

- State 1: inactive/passive stretch state.
- State 2: activated/retraction state under negative pressure.

In the Python script, each muscle type stores:

```python
L0_state1, k_state1, c_state1
L0_state2, k_state2, c_state2
F_active_max
FORCE_SIGN
```

The switching rule is:

- if `ctrl <= ACT_THRESHOLD`, use State 1 and generate passive force only;
- if `ctrl > ACT_THRESHOLD`, use State 2 and generate activated contraction force.

At every simulation step, the script:

1. reads current actuator length and velocity from MuJoCo;
2. computes the muscle scalar force through `switched_state_force()`;
3. obtains the actuator moment-arm row from `data.actuator_moment`;
4. maps scalar tendon force into generalized force;
5. accumulates the result in `data.qfrc_applied`;
6. advances the MuJoCo simulation with `mujoco.mj_step()`.

This keeps tendon routing and moment arms configuration-dependent, instead of
applying manually guessed joint torques.

## Geometry Variants

| XML | Script | Notes |
| --- | --- | --- |
| `system_model_n=0.xml` | `view_switched_state_mujoco_n=0.py` | Baseline geometry variant. |
| `system_model_n=4.xml` | `view_switched_state_mujoco_n=4.py` | Geometry variant with a different shoulder joint offset. |
| `system_model_n=4_008.xml` | `view_switched_state_mujoco_n=4_008.py` | Geometry variant. |
| `system_model_n=4_010.xml` | `view_switched_state_mujoco_n=4_010.py` | Geometry variant. |
| `system_model_n=6_010.xml` | `view_switched_state_mujoco_n=6_010.py` | Recommended current full-arm version. |
| `system_model_n=6_012.xml` | `view_switched_state_mujoco_n=6_012.py` | Geometry variant. |
| `system_model_n=8.xml` | `view_switched_state_mujoco_n=8.py` | Geometry variant with larger shoulder joint offset. |

For handover, start from `n=6_010`. Treat the other versions as previous
geometry candidates unless you are specifically reproducing parameter comparison
experiments.

## Reinforcement Learning Environment

`custom_env_v4.py` defines a preliminary Gymnasium environment:

- class name: `RobExampleEnv`;
- base class: `gymnasium.envs.mujoco.MujocoEnv`;
- action space: `MultiBinary(self.muscle_count)`;
- observation vector currently has 16 values:
  - shoulder ball-joint position quaternion: 4;
  - shoulder angular velocity: 3;
  - `forearm_tip` position: 3;
  - vector from tip to target: 3;
  - target position: 3;
- reward terms:
  - distance penalty to target;
  - action-change penalty;
  - action sparsity penalty;
  - optional stability reward is computed but not included in the current final reward.

Important: this file currently loads:

```python
xml_path = os.path.join(current_dir, "system_model.xml")
```

However, the repository does not currently contain `system_model.xml`. Before RL
training, either:

```bash
cp system_model_n=6_010.xml system_model.xml
```

or change `custom_env_v4.py` to load the intended XML file explicitly.

Also note that the RL environment currently uses MuJoCo's normal
`do_simulation(a, frame_skip)` path. It does not yet include the full external
switched-state muscle force law from `view_switched_state_mujoco_n=6_010.py`.
If the goal is to train a controller for the same pneumatic muscle model, the next
step should be to port the switched-state force computation into the environment's
`step()` method.

## Suggested Handover Workflow

1. Run `--list-actuators` to confirm the model loads.
2. Run single-muscle tests such as `--activate muscle1`.
3. Run grouped tests such as `--activate upper_front` and `--activate lower_front`.
4. Compare final `elbow` and `forearm_tip` positions printed by the script.
5. Save final screenshots for documentation when needed.
6. Only after the viewer script is understood, update `custom_env_v4.py` for RL.

## Known Issues and TODOs

- `custom_env_v4.py` refers to `system_model.xml`, which is missing from the
  repository.
- The viewer scripts and XML files are stored in a flat directory. A future cleanup
  could place XML models under `models/` and scripts under `scripts/`.
- There is no `requirements.txt` or `pyproject.toml` yet. Adding one would make
  setup easier.
- The RL environment has not yet been fully synchronized with the switched-state
  muscle force implementation.
- Some geometry variants are not documented in code comments. Keep `n=6_010` as
  the default handover version unless reproducing older tests.

## Recommended Next Steps

Add a `requirements.txt`:

```text
mujoco
numpy
imageio
gymnasium
```

Rename or symlink the selected XML model for RL:

```bash
ln -s system_model_n=6_010.xml system_model.xml
```

Refactor the switched-state muscle model into a shared module, for example
`muscle_model.py`, so both the viewer and RL environment use the same force law.

Add a short training script once the RL environment is synchronized with the
custom muscle dynamics.

## Citation / Thesis Context

This code supports the project:

```text
Structure Design and Optimization of a Muscular-Skeleton System
```

The thesis studies a MuJoCo-based musculoskeletal robotic arm driven by
origami-inspired pneumatic muscles. The design uses an anchor ring, upper arm,
forearm, antagonistic muscle groups, and a switched-state muscle model to analyze
single-muscle actuation, passive antagonist behavior, workspace characteristics, and
local structural optimization.
