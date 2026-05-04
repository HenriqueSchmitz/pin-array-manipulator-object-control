import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import numpy as np

from rl_common import setup, load_model
from constants import RUN_NAME, MAX_XY_STEP
from train import make_env


PATHS = setup(__file__, f"test_{RUN_NAME}", RUN_NAME)

# Choose one:
MODE = "policy"
# MODE = "oracle"
# MODE = "zero"
# MODE = "constant_x"
# MODE = "constant_neg_x"
# MODE = "constant_y"
# MODE = "constant_neg_y"


def fmt(x, default=0.0, digits=4):
    try:
        return f"{float(x): .{digits}f}"
    except Exception:
        return f"{float(default): .{digits}f}"


def arr(x, digits=4):
    if x is None:
        return None
    return np.round(np.asarray(x, dtype=np.float32), digits)


def xy_delta(object_pose, target):
    if object_pose is None or target is None:
        return None

    object_pose = np.asarray(object_pose, dtype=np.float32)
    target = np.asarray(target, dtype=np.float32)

    return target[:2] - object_pose[:2]


def make_oracle_action(info):
    """
    Simple hand-coded controller.

    If this cannot reduce XY error, PPO has no chance.
    """
    action = np.zeros(6, dtype=np.float32)

    object_pose = info.get("current_object_pose")
    target = info.get("target")

    if object_pose is None or target is None:
        return action

    dxy = xy_delta(object_pose, target)
    norm = float(np.linalg.norm(dxy))

    if norm > 1e-8:
        # Full-scale command toward target.
        action[:2] = dxy / norm

    return np.clip(action, -1.0, 1.0).astype(np.float32)


def make_constant_action(mode):
    action = np.zeros(6, dtype=np.float32)

    if mode == "constant_x":
        action[0] = 1.0
    elif mode == "constant_neg_x":
        action[0] = -1.0
    elif mode == "constant_y":
        action[1] = 1.0
    elif mode == "constant_neg_y":
        action[1] = -1.0

    return action.astype(np.float32)


def print_header(info, obs):
    print("model:", PATHS.stable_model_path)
    print("mode:", MODE)
    print("target:", arr(info.get("target")))
    print("base_generator_target:", arr(info.get("base_generator_target")))
    print("synthetic_target_used:", info.get("synthetic_target_used"))
    print("curriculum_mode:", info.get("curriculum_mode"))
    print("initial_xy_error:", fmt(info.get("initial_xy_error")))
    print("initial_rotation_error:", fmt(info.get("initial_rotation_error")))
    print("object_pose:", arr(info.get("current_object_pose")))
    print("obs_shape:", np.asarray(obs).shape)
    print("MAX_XY_STEP:", MAX_XY_STEP)
    print()


def choose_action(model, obs, info):
    if MODE == "policy":
        action, _ = model.predict(obs, deterministic=True)
        return np.asarray(action, dtype=np.float32).reshape(-1)

    if MODE == "oracle":
        return make_oracle_action(info)

    if MODE == "zero":
        return np.zeros(6, dtype=np.float32)

    if MODE in (
        "constant_x",
        "constant_neg_x",
        "constant_y",
        "constant_neg_y",
    ):
        return make_constant_action(MODE)

    raise ValueError(f"Unknown MODE: {MODE}")


def main():
    env = make_env(render_mode="human")

    model = None
    if MODE == "policy":
        model = load_model(PATHS, env=env)

    obs, info = env.reset()

    print_header(info, obs)

    total_reward = 0.0

    last_info = info
    last_action = None

    try:
        for step in range(2000):
            action = choose_action(model, obs, info)

            obs, reward, terminated, truncated, info = env.step(action)

            reward = float(reward)
            total_reward += reward

            raw_action = np.asarray(action, dtype=np.float32).reshape(-1)

            object_pose = info.get("current_object_pose")
            target = info.get("target")
            dxy = xy_delta(object_pose, target)

            last_info = info
            last_action = raw_action

            print(
                f"step={step + 1:04d} | "
                f"r={reward:8.3f} | "
                f"tot={total_reward:9.3f} | "
                f"xy={fmt(info.get('current_xy_error'))} | "
                f"dxy={arr(dxy, 4)} | "
                f"prog={fmt(info.get('progress_xy'), digits=6)} | "
                f"toward={fmt(info.get('movement_toward_goal'), digits=6)} | "
                f"cmd_along={fmt(info.get('actual_motion_along_command'), digits=6)} | "
                f"target_along={fmt(info.get('actual_motion_toward_target'), digits=6)} | "
                f"move_norm={fmt(info.get('actual_xy_motion_norm'), digits=6)} | "
                f"cmd_norm={fmt(info.get('command_xy_norm'), digits=6)} | "
                f"dx={fmt(info.get('chosen_dx'), digits=5)} | "
                f"dy={fmt(info.get('chosen_dy'), digits=5)} | "
                f"dz={fmt(info.get('chosen_dz'), digits=5)} | "
                f"raw_dz={fmt(info.get('raw_chosen_dz'), digits=5)} | "
                f"dr={fmt(info.get('chosen_drotvec_norm'), digits=5)} | "
                f"rot={fmt(info.get('current_rotation_error'))} | "
                f"z={fmt(info.get('object_z'), digits=4)} | "
                f"oob={info.get('object_out_of_bounds')} | "
                f"fell={info.get('object_fell')} | "
                f"raw={np.round(raw_action, 3)} | "
                f"success={info.get('success')} | "
                f"failure={info.get('failure')} | "
                f"base_term={info.get('base_terminated')} | "
                f"base_trunc={info.get('base_truncated')}"
            )

            if terminated or truncated:
                print()
                print(
                    "done:",
                    {
                        "terminated": bool(terminated),
                        "truncated": bool(truncated),
                        "success": info.get("success"),
                        "failure": info.get("failure"),
                        "xy_error": info.get("current_xy_error"),
                        "rotation_error": info.get("current_rotation_error"),
                        "target": arr(target),
                        "object_pose": arr(object_pose),
                        "target_minus_object_xy": arr(dxy),
                        "raw_action": arr(raw_action, 4),
                        "total_reward": total_reward,
                    },
                )
                break

    except KeyboardInterrupt:
        print("Stopped.")

    finally:
        print()
        print(
            "final snapshot:",
            {
                "xy_error": last_info.get("current_xy_error"),
                "rotation_error": last_info.get("current_rotation_error"),
                "target": arr(last_info.get("target")),
                "object_pose": arr(last_info.get("current_object_pose")),
                "target_minus_object_xy": arr(
                    xy_delta(
                        last_info.get("current_object_pose"),
                        last_info.get("target"),
                    )
                ),
                "raw_action": arr(last_action, 4),
                "total_reward": total_reward,
            },
        )

        env.close()

    print("total_reward:", total_reward)


if __name__ == "__main__":
    main()