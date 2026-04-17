import argparse
import os
import time
from typing import Dict, List, Optional
import numpy as np
import pathlib
import imageio.v2 as imageio

# import sys
# import mujoco
# print("Python exe:", sys.executable)
# print("MuJoCo version:", mujoco.__version__)
# print("NumPy version:", np.__version__)

'''
python view_switched_state_mujoco_n=6_010.py --activate muscle1
python view_switched_state_mujoco_n=6_010.py --activate upper_front
python view_switched_state_mujoco_n=6_010.py --activate upper_back
python view_switched_state_mujoco_n=6_010.py --activate lower_front
'''

'''
python view_switched_state_mujoco_n=6_010.py \
  --activate upper_back \
  --save-final-image \
  --final-image-path final_state.png
'''

try:
    import mujoco
    from mujoco import viewer as mj_viewer
except Exception as e:
    raise ImportError(
        "This script requires the Python package 'mujoco'. "
        "Install it in the environment where you run the simulation."
    ) from e

DEFAULT_XML_PATH = os.path.join(os.path.dirname(__file__), "system_model_n=6_010.xml")
ACT_THRESHOLD = 1e-6
DEFAULT_PRE_STEPS = 800
DEFAULT_TOTAL_STEPS = 8000
DEFAULT_RENDER_EVERY = 1
DEFAULT_PRINT_EVERY = 7999
DEFAULT_REALTIME_SLEEP = True
DEFAULT_ACTIVE_CTRL = 1.0

TYPE_PARAM_TEMPLATES = {
    # upper arm muscle--front(tricep)
    # d2=130
    "A": {"L0_state1": 0.1097, "k_state1": 11.5, "c_state1": 1.0000,
          "L0_state2": 0.0655, "k_state2": 22.5, "c_state2": 1.0000,
          "F_active_max": 2.4, "FORCE_SIGN": -1.0},
    "B": {"L0_state1": 0.1204, "k_state1": 11.5, "c_state1": 1.0000,
          "L0_state2": 0.0719, "k_state2": 22.5, "c_state2": 1.0000,
          "F_active_max": 2.4, "FORCE_SIGN": -1.0},
    "C": {"L0_state1": 0.1301, "k_state1": 11.5, "c_state1": 1.0000,
          "L0_state2": 0.0777, "k_state2": 22.5, "c_state2": 1.0000,
          "F_active_max": 2.4, "FORCE_SIGN": -1.0},
    "D": {"L0_state1": 0.1386, "k_state1": 11.5, "c_state1": 1.0000,
          "L0_state2": 0.0827, "k_state2": 22.5, "c_state2": 1.0000,
          "F_active_max": 2.4, "FORCE_SIGN": -1.0},
    "E": {"L0_state1": 0.1457, "k_state1": 11.5, "c_state1": 1.0000,
          "L0_state2": 0.087, "k_state2": 22.5, "c_state2": 1.0000,
          "F_active_max": 2.4, "FORCE_SIGN": -1.0},
    "F": {"L0_state1": 0.1511, "k_state1": 11.5, "c_state1": 1.0000,
          "L0_state2": 0.0902, "k_state2": 22.5, "c_state2": 1.0000,
          "F_active_max": 2.4, "FORCE_SIGN": -1.0},
    "G": {"L0_state1": 0.1549, "k_state1": 11.5, "c_state1": 1.0000,
          "L0_state2": 0.0924, "k_state2": 22.5, "c_state2": 1.0000,
          "F_active_max": 2.4, "FORCE_SIGN": -1.0},
    "H": {"L0_state1": 0.1567, "k_state1": 11.5, "c_state1": 1.0000,
          "L0_state2": 0.0936, "k_state2": 22.5, "c_state2": 1.0000,
          "F_active_max": 2.4, "FORCE_SIGN": -1.0},

    # upper arm muscle--behind(bicep)
    # d1=240
    "I": {"L0_state1": 0.1708, "k_state1": 11.5, "c_state1": 1.0000,
          "L0_state2": 0.102, "k_state2": 22.5, "c_state2": 1.0000,
          "F_active_max": 2.4, "FORCE_SIGN": -1.0},
    "J": {"L0_state1": 0.162, "k_state1": 11.5, "c_state1": 1.0000,
          "L0_state2": 0.0967, "k_state2": 22.5, "c_state2": 1.0000,
          "F_active_max": 2.4, "FORCE_SIGN": -1.0},
    "K": {"L0_state1": 0.1553, "k_state1": 11.5, "c_state1": 1.0000,
          "L0_state2": 0.0927, "k_state2": 22.5, "c_state2": 1.0000,
          "F_active_max": 2.4, "FORCE_SIGN": -1.0},
    "L": {"L0_state1": 0.151, "k_state1": 11.5, "c_state1": 1.0000,
          "L0_state2": 0.0901, "k_state2": 22.5, "c_state2": 1.0000,
          "F_active_max": 2.4, "FORCE_SIGN": -1.0},
    "M": {"L0_state1": 0.1495, "k_state1": 11.5, "c_state1": 1.0000,
          "L0_state2": 0.0893, "k_state2": 22.5, "c_state2": 1.0000,
          "F_active_max": 2.4, "FORCE_SIGN": -1.0},

    # lower arm muscle
    "N": {"L0_state1": 0.1460, "k_state1": 11.5, "c_state1": 1.0000,
          "L0_state2": 0.0872, "k_state2": 22.5, "c_state2": 1.0000,
          "F_active_max": 2.4, "FORCE_SIGN": -1.0},
    "O": {"L0_state1": 0.0660, "k_state1": 11.5, "c_state1": 1.0000,
          "L0_state2": 0.0394, "k_state2": 22.5, "c_state2": 1.0000,
          "F_active_max": 2.4, "FORCE_SIGN": -1.0},
}

MUSCLE_TYPE_MAP = {
    "muscle1": "A", "muscle2": "B", "muscle3": "C", "muscle4": "D",
    "muscle5": "E", "muscle6": "F", "muscle7": "G", "muscle8": "H",
    "muscle9": "H", "muscle10": "G", "muscle11": "F", "muscle12": "E",
    "muscle13": "D", "muscle14": "C", "muscle15": "B", "muscle16": "A",
    "muscle17": "I", "muscle18": "J", "muscle19": "K", "muscle20": "L", "muscle21": "M",
    "muscle22": "L", "muscle23": "K", "muscle24": "J", "muscle25": "I",
    "elbow_front_flexor1": "N", "elbow_front_flexor2": "N", "elbow_front_flexor3": "N", "elbow_front_flexor4": "N",
    "elbow_front_flexor5": "N", "elbow_front_flexor6": "N", "elbow_front_flexor7": "N", "elbow_front_flexor8": "N",
    "elbow_behind_flexor1": "O", "elbow_behind_flexor2": "O", "elbow_behind_flexor3": "O", "elbow_behind_flexor4": "O",
    "elbow_behind_flexor5": "O", "elbow_behind_flexor6": "O", "elbow_behind_flexor7": "O", "elbow_behind_flexor8": "O",
}

PRESET_GROUPS = {
    "upper_front": [f"muscle{i}" for i in range(1, 17)],
    "upper_back": [f"muscle{i}" for i in range(17, 26)],
    "lower_front": [f"elbow_front_flexor{i}" for i in range(1, 9)],
    "lower_back": [f"elbow_behind_flexor{i}" for i in range(1, 9)],
}


def validate_state_difference(params: Dict[str, float], actuator_name: str):
    same = (
        np.isclose(params["L0_state1"], params["L0_state2"]) and
        np.isclose(params["k_state1"], params["k_state2"]) and
        np.isclose(params["c_state1"], params["c_state2"]) and
        np.isclose(params["F_active_max"], 0.0)
    )
    if same:
        raise ValueError(f"{actuator_name}: state1 and state2 are effectively identical.")


def build_actuator_param_dict(model: mujoco.MjModel) -> Dict[str, Dict[str, float]]:
    actuator_params: Dict[str, Dict[str, float]] = {}
    for act_id in range(model.nu):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, act_id)
        if name is None:
            raise ValueError(f"Actuator id={act_id} has no name.")
        if name not in MUSCLE_TYPE_MAP:
            raise ValueError(f"Actuator '{name}' missing in MUSCLE_TYPE_MAP.")
        type_name = MUSCLE_TYPE_MAP[name]
        if type_name not in TYPE_PARAM_TEMPLATES:
            raise ValueError(f"Type '{type_name}' for actuator '{name}' missing in TYPE_PARAM_TEMPLATES.")
        params = dict(TYPE_PARAM_TEMPLATES[type_name])
        validate_state_difference(params, name)
        actuator_params[name] = params
    return actuator_params


def switched_state_force(length: float, velocity: float, ctrl: float, params: Dict[str, float]):
    if ctrl <= ACT_THRESHOLD:
        l0 = params["L0_state1"]
        k = params["k_state1"]
        c = params["c_state1"]
        state_flag = 1
        active_force = 0.0
        passive_extension = max(length - l0, 0.0)
        passive_force = k * passive_extension + c * max(velocity, 0.0)
    else:
        l0 = params["L0_state2"]
        k = params["k_state2"]
        c = params["c_state2"]
        state_flag = 2
        active_extension = max(length - l0, 0.0)
        active_force = k * active_extension + c * max(velocity, 0.0)
        passive_force = 0.0

    total_force = params["FORCE_SIGN"] * (active_force + passive_force)
    return float(total_force), float(active_force), float(passive_force), int(state_flag)


def actuator_generalized_force_row(model: mujoco.MjModel, data: mujoco.MjData, actuator_id: int) -> np.ndarray:
    nv = int(model.nv)
    nu = int(model.nu)
    moment = np.asarray(data.actuator_moment)

    #print(data.actuator_moment)

    if moment.ndim == 2:
        rows, cols = moment.shape
        if rows == nu and cols >= nv:
            return moment[actuator_id, :nv].copy()
        if rows >= nv and cols == nu:
            return moment[:nv, actuator_id].copy()
        if rows == nv and cols >= nu:
            return moment[:, actuator_id].copy()
        raise ValueError(f"Unexpected 2D actuator_moment shape {moment.shape}; expected compatibility with (nu={nu}, nv={nv}).")

    if moment.ndim == 1:
        if moment.size % nu == 0:
            cols = moment.size // nu
            mat = moment.reshape(nu, cols)
            if cols >= nv:
                return mat[actuator_id, :nv].copy()
        if moment.size % nv == 0:
            rows = moment.size // nv
            mat = moment.reshape(rows, nv)
            if rows > actuator_id:
                return mat[actuator_id, :].copy()
        raise ValueError(f"Unexpected flat actuator_moment size {moment.size}; cannot map to (nu={nu}, nv={nv}).")

    raise ValueError(f"Unsupported actuator_moment ndim={moment.ndim}")


def expand_actuator_tokens(model: mujoco.MjModel, tokens: List[str]) -> List[str]:
    actuator_names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i) for i in range(model.nu)]
    actuator_set = set(actuator_names)
    result: List[str] = []
    for token in tokens:
        token = token.strip()
        if not token:
            continue
        if token in PRESET_GROUPS:
            expanded = PRESET_GROUPS[token]
            for name in expanded:
                if name not in actuator_set:
                    raise ValueError(f"Preset group '{token}' contains actuator '{name}', but it is not found in the current model.")
                result.append(name)
            continue
        if token.isdigit():
            idx = int(token)
            if idx < 0 or idx >= model.nu:
                raise ValueError(f"Actuator index {idx} out of range [0, {model.nu - 1}].")
            result.append(actuator_names[idx])
            continue
        if token not in actuator_set:
            raise ValueError(f"Unknown actuator token: {token}")
        result.append(token)
    deduped = []
    seen = set()
    for name in result:
        if name not in seen:
            deduped.append(name)
            seen.add(name)
    return deduped


def parse_activate_argument(model: mujoco.MjModel, activate_arg: str) -> List[str]:
    if activate_arg is None or not activate_arg.strip():
        return ["muscle1"]
    tokens = [x.strip() for x in activate_arg.split(",")]
    return expand_actuator_tokens(model, tokens)


def make_action_vector(model: mujoco.MjModel, active_names: List[str], active_ctrl: float) -> np.ndarray:
    action = np.zeros(model.nu, dtype=float)
    active_set = set(active_names)
    for act_id in range(model.nu):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, act_id)
        if name in active_set:
            action[act_id] = active_ctrl
    return action


def get_body_pos(model: mujoco.MjModel, data: mujoco.MjData, body_name: str) -> np.ndarray:
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    if body_id < 0:
        raise ValueError(f"Body '{body_name}' not found in model.")
    return data.xpos[body_id].copy()


def angle_deg(vec: np.ndarray, ref_vec: np.ndarray) -> float:
    a = np.asarray(vec, dtype=float)
    b = np.asarray(ref_vec, dtype=float)
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return float("nan")
    cosang = np.clip(np.dot(a, b) / (na * nb), -1.0, 1.0)
    return float(np.degrees(np.arccos(cosang)))


def resolve_upper_origin(model: mujoco.MjModel, data: mujoco.MjData, upper_origin_body: Optional[str], upper_origin_xyz: Optional[np.ndarray]) -> np.ndarray:
    if upper_origin_xyz is not None:
        return np.asarray(upper_origin_xyz, dtype=float)
    if upper_origin_body:
        return get_body_pos(model, data, upper_origin_body)
    return np.array([-0.025, 0.0, -0.06], dtype=float)


def summarize_step(step: int, model: mujoco.MjModel, data: mujoco.MjData, force_info: List[dict], active_names: List[str], tracked_body: str):
    tracked_pos = None
    try:
        tracked_pos = get_body_pos(model, data, tracked_body)
    except Exception:
        tracked_pos = None

    print(f"\n[step {step:05d}] time={data.time:.4f}s")
    if tracked_pos is not None:
        print(f"  tracked_body={tracked_body} pos={np.array2string(tracked_pos, precision=5)}")
    for item in force_info:
        mark = "*" if item["name"] in active_names else " "
        print(
            f" {mark} {item['name']:<24s} state={item['state']} "
            f"ctrl={item['ctrl']:.2f} len={item['length']:.5f} vel={item['velocity']:.5f} "
            f"total={item['total_force']:.5f} active={item['active_force']:.5f} passive={item['passive_force']:.5f}"
        )


def print_final_pose_summary(model: mujoco.MjModel, data: mujoco.MjData, upper_origin_init: np.ndarray):
    try:
        elbow = get_body_pos(model, data, "elbow")
        tip = get_body_pos(model, data, "forearm_tip")
    except Exception as e:
        print(f"\nFinal pose summary unavailable: {e}")
        return

    upper_vec_final = elbow - upper_origin_init
    forearm_vec_final = tip - elbow

    mujoco.mj_forward(model, data)
    # assume current data is final; use saved initial vectors from reset-like geometry
    # initial vectors are reconstructed from qpos0 via a temporary data object
    d0 = mujoco.MjData(model)
    mujoco.mj_resetData(model, d0)
    mujoco.mj_forward(model, d0)
    elbow0 = get_body_pos(model, d0, "elbow")
    tip0 = get_body_pos(model, d0, "forearm_tip")
    upper_vec_init = elbow0 - upper_origin_init
    forearm_vec_init = tip0 - elbow0

    print("\n===== Final pose summary =====")
    print(f"upper_origin      = {np.array2string(upper_origin_init, precision=6)}")
    print(f"final elbow       = {np.array2string(elbow, precision=6)}")
    print(f"final forearm_tip = {np.array2string(tip, precision=6)}")
    print(f"upperarm vector   = {np.array2string(upper_vec_final, precision=6)}")
    print(f"forearm vector    = {np.array2string(forearm_vec_final, precision=6)}")
    print(f"upperarm angle wrt initial  = {angle_deg(upper_vec_final, [0.0, 0.0, -1]):.6f} deg")
    print(f"forearm angle wrt initial   = {angle_deg(forearm_vec_final, upper_vec_final):.6f} deg")

def get_offscreen_buffer_size(model: mujoco.MjModel) -> tuple:
    """
    Return the offscreen framebuffer size defined by the model.
    Different MuJoCo bindings may expose the field slightly differently.
    """
    candidates = [
        ("vis", "global_", "offwidth"),
        ("vis", "global", "offwidth"),
    ]

    offwidth = None
    offheight = None

    for path in candidates:
        try:
            obj = model
            for key in path[:-1]:
                obj = getattr(obj, key)
            offwidth = getattr(obj, path[-1])
            # height field is on the same object
            offheight = getattr(getattr(model, path[0]), path[1]).offheight
            break
        except Exception:
            continue

    if offwidth is None or offheight is None:
        # conservative fallback matching the error message you saw
        offwidth, offheight = 640, 480

    return int(offwidth), int(offheight)

def save_final_screenshot(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    output_path: str,
    width: int = 1280,
    height: int = 960,
) -> bool:
    """
    Save a rendered image of the final MuJoCo state to a PNG file.

    This function automatically clamps the requested size to the XML-defined
    offscreen framebuffer size, so it will not crash when width/height are too large.
    Returns True on success, False on failure.
    """
    output_path = str(pathlib.Path(output_path).expanduser().resolve())
    parent_dir = os.path.dirname(output_path)
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)

    max_width, max_height = get_offscreen_buffer_size(model)
    safe_width = max(1, min(int(width), max_width))
    safe_height = max(1, min(int(height), max_height))

    if safe_width != width or safe_height != height:
        print(
            f"Requested final image size {width}x{height} exceeds offscreen framebuffer "
            f"{max_width}x{max_height}. Using {safe_width}x{safe_height} instead."
        )

    renderer = None
    try:
        mujoco.mj_forward(model, data)

        renderer = mujoco.Renderer(model, height=safe_height, width=safe_width)
        renderer.update_scene(data)
        pixels = renderer.render()

        imageio.imwrite(output_path, pixels)
        print(f"Final screenshot saved to: {output_path}")
        return True

    except Exception as e:
        print(f"Failed to save final screenshot: {e}")
        print(
            "Tip: either use a smaller --final-image-width/--final-image-height, "
            "or enlarge the XML offscreen framebuffer via:\n"
            "<visual>\n"
            "  <global offwidth=\"1600\" offheight=\"1200\"/>\n"
            "</visual>"
        )
        return False

    finally:
        if renderer is not None:
            try:
                renderer.close()
            except Exception:
                pass

def run_viewer(
    xml_path: str,
    active_names: List[str],
    pre_steps: int,
    total_steps: int,
    render_every: int,
    print_every: int,
    tracked_body: str,
    realtime_sleep: bool,
    active_ctrl: float,
    upper_origin_body: Optional[str],
    upper_origin_xyz: Optional[np.ndarray],
    save_final_image: bool,
    final_image_path: str,
    final_image_width: int,
    final_image_height: int,):
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)

    actuator_params = build_actuator_param_dict(model)
    actuator_names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i) for i in range(model.nu)]
    prev_lengths = np.asarray(data.actuator_length).copy()
    upper_origin_init = resolve_upper_origin(model, data, upper_origin_body, upper_origin_xyz)

    print(f"Model loaded: {os.path.basename(xml_path)}")
    print(f"nu={model.nu}, nv={model.nv}, timestep={model.opt.timestep:.6f}")
    print(f"Active actuators after pre-steps: {active_names}")
    print(f"pre_steps={pre_steps}, total_steps={total_steps}")
    print("Preset names available:", ", ".join(PRESET_GROUPS.keys()))

    viewer = None
    try:
        viewer = mj_viewer.launch_passive(model, data)
    except Exception as e:
        print(f"Viewer launch failed: {e}")
        print("Continuing without on-screen viewer.")

    try:
        for step in range(total_steps):
            if viewer is not None and hasattr(viewer, "is_running") and not viewer.is_running():
                print("Viewer closed by user. Stopping simulation loop.")
                break

            data.ctrl[:] = 0.0
            data.qfrc_applied[:] = 0.0
            mujoco.mj_forward(model, data)

            current_lengths = np.asarray(data.actuator_length).copy()
            try:
                current_velocities = np.asarray(data.actuator_velocity).copy()
            except Exception:
                dt = max(model.opt.timestep, 1e-9)
                current_velocities = (current_lengths - prev_lengths) / dt

            action = np.zeros(model.nu, dtype=float) if step < pre_steps else make_action_vector(model, active_names, active_ctrl)

            force_info = []
            for act_id, act_name in enumerate(actuator_names):
                ctrl = float(action[act_id])
                params = actuator_params[act_name]
                total_force, active_force, passive_force, state_flag = switched_state_force(
                    float(current_lengths[act_id]),
                    float(current_velocities[act_id]),
                    ctrl,
                    params,
                )
                moment_row = actuator_generalized_force_row(model, data, act_id)
                data.qfrc_applied[:] += moment_row * total_force
                force_info.append({
                    "name": act_name,
                    "ctrl": ctrl,
                    "state": state_flag,
                    "length": float(current_lengths[act_id]),
                    "velocity": float(current_velocities[act_id]),
                    "total_force": total_force,
                    "active_force": active_force,
                    "passive_force": passive_force,
                })

            mujoco.mj_step(model, data)
            prev_lengths = current_lengths

            if step % print_every == 0:
                active_now = active_names if step >= pre_steps else []
                summarize_step(step, model, data, force_info, active_now, tracked_body)

            if viewer is not None and step % render_every == 0:
                viewer.sync()

            if realtime_sleep:
                time.sleep(model.opt.timestep)
    finally:
        try:
            print_final_pose_summary(model, data, upper_origin_init)
            if save_final_image:
                save_final_screenshot(
                    model=model,
                    data=data,
                    output_path=final_image_path,
                    width=final_image_width,
                    height=final_image_height,
                )
        finally:
            try:
                viewer.close()
            except Exception:
                pass

def parse_xyz_arg(text: Optional[str]) -> Optional[np.ndarray]:
    if text is None:
        return None
    parts = [p.strip() for p in text.split(",")]
    if len(parts) != 3:
        raise ValueError("--upper-origin-xyz must be in the form x,y,z")
    return np.array([float(x) for x in parts], dtype=float)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Visualize the switched-state muscle model in MuJoCo.")
    parser.add_argument("--xml", default=DEFAULT_XML_PATH, help="Path to MuJoCo XML model.")
    parser.add_argument("--activate", default="muscle1",
                        help="Comma-separated actuator names, indices, or preset names. Examples: 'muscle1,muscle16', '0,15', 'upper_front'.")
    parser.add_argument("--pre-steps", type=int, default=DEFAULT_PRE_STEPS, help="Initial steps with all muscles in state1.")
    parser.add_argument("--total-steps", type=int, default=DEFAULT_TOTAL_STEPS, help="Total simulation steps.")
    parser.add_argument("--render-every", type=int, default=DEFAULT_RENDER_EVERY, help="Viewer sync interval.")
    parser.add_argument("--print-every", type=int, default=DEFAULT_PRINT_EVERY, help="Console print interval.")
    parser.add_argument("--tracked-body", default="elbow", help="Body name whose world position is printed.")
    parser.add_argument("--active-ctrl", type=float, default=DEFAULT_ACTIVE_CTRL, help="Control value applied after pre-steps.")
    parser.add_argument("--upper-origin-body", default=None, help="Body name used as upperarm origin for final angle calculation.")
    parser.add_argument("--upper-origin-xyz", default=None, help="World coordinate x,y,z used as upperarm origin.")
    parser.add_argument("--no-realtime-sleep", action="store_true", help="Disable real-time sleeping.")
    parser.add_argument("--list-actuators", action="store_true", help="Print actuator indices and names, then exit.")
    parser.add_argument("--save-final-image", action="store_true", help="Save a PNG image of the final simulation state.")
    parser.add_argument("--final-image-path", default="final_musculoskeletal_state.png", help="Path to save the final PNG image.")
    parser.add_argument("--final-image-width", type=int, default=1280, help="Width of the saved final image.")
    parser.add_argument("--final-image-height", type=int, default=960, help="Height of the saved final image.")
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    model = mujoco.MjModel.from_xml_path(args.xml)
    if args.list_actuators:
        print(f"Actuators in {os.path.basename(args.xml)}:")
        for i in range(model.nu):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
            print(f"  {i:2d}: {name}")
        return

    active_names = parse_activate_argument(model, args.activate)
    print("Resolved active actuators:", active_names)
    run_viewer(
        xml_path=args.xml,
        active_names=active_names,
        pre_steps=args.pre_steps,
        total_steps=args.total_steps,
        render_every=max(1, args.render_every),
        print_every=max(1, args.print_every),
        tracked_body=args.tracked_body,
        realtime_sleep=not args.no_realtime_sleep,
        active_ctrl=float(args.active_ctrl),
        upper_origin_body=args.upper_origin_body,
        upper_origin_xyz=parse_xyz_arg(args.upper_origin_xyz),
        save_final_image=args.save_final_image,
        final_image_path=args.final_image_path,
        final_image_width=max(1, args.final_image_width),
        final_image_height=max(1, args.final_image_height),
    )

if __name__ == "__main__":
    main()
