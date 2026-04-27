import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from rl_common import setup, load_model

PATHS = setup(__file__, "test_6dof_twist_v3", "ppo_6dof_twist_v3")

import numpy as np

from train_6dof import make_env


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


def main():
    env = make_env(render_mode="human")
    model = load_model(PATHS, env=env)

    obs, info = env.reset()
    total_reward = 0.0

    print("model:", PATHS.stable_model_path)
    print("target:", arr(info.get("target")))
    print("initial_xy_error:", fmt(info.get("initial_translation_error")))
    print("initial_rotation_error:", fmt(info.get("initial_rotation_error")))
    print("synthetic_target_used:", info.get("synthetic_target_used"))
    print("obs_shape:", np.asarray(obs).shape)
    print()

    last_info = info
    last_object_pose = None
    last_target = info.get("target")
    last_raw_action = None

    try:
        for step in range(2000):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)

            reward = float(reward)
            total_reward += reward

            raw_action = np.asarray(action, dtype=np.float32).reshape(-1)
            object_pose = info.get("current_object_pose")
            target = info.get("target")
            dxy = xy_delta(object_pose, target)

            last_info = info
            last_object_pose = object_pose
            last_target = target
            last_raw_action = raw_action

            print(
                f"step={step + 1:04d} | "
                f"reward={reward:8.3f} | "
                f"total={total_reward:9.3f} | "
                f"xy={fmt(info.get('current_translation_error'))} | "
                f"rot={fmt(info.get('current_rotation_error'))} | "
                f"dxy={arr(dxy, 4)} | "
                f"xy_prog={fmt(info.get('translation_progress'), digits=6)} | "
                f"rot_prog={fmt(info.get('rotation_progress'), digits=6)} | "
                f"xy_gate={fmt(info.get('translation_gate'), digits=4)} | "
                f"dx={fmt(info.get('chosen_dx'))} | "
                f"dy={fmt(info.get('chosen_dy'))} | "
                f"dz={fmt(info.get('chosen_dz'), digits=5)} | "
                f"dr_norm={fmt(info.get('chosen_drotvec_norm'), digits=5)} | "
                f"rpy=({fmt(info.get('executed_roll'))},"
                f"{fmt(info.get('executed_pitch'))},"
                f"{fmt(info.get('executed_yaw'))}) | "
                f"pin_hold_n={info.get('held_previous_pin_count')} | "
                f"min_pin_raw={fmt(info.get('min_pin_before_floor'))} | "
                f"raw={np.round(raw_action, 3)} | "
                f"success={info.get('success')} | "
                f"failure={info.get('failure')} | "
                f"base_term={info.get('base_terminated')}"
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
                        "xy_error": info.get("current_translation_error"),
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
        if last_info is not None:
            print()
            print(
                "final snapshot:",
                {
                    "xy_error": last_info.get("current_translation_error"),
                    "rotation_error": last_info.get("current_rotation_error"),
                    "target": arr(last_target),
                    "object_pose": arr(last_object_pose),
                    "target_minus_object_xy": arr(xy_delta(last_object_pose, last_target)),
                    "raw_action": arr(last_raw_action, 4),
                    "total_reward": total_reward,
                },
            )

        env.close()

    print("total_reward:", total_reward)


if __name__ == "__main__":
    main()