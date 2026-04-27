import argparse
import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from rl_common import setup

import numpy as np
from stable_baselines3 import PPO

from train_6dof_staged import (
    XYZ_MODEL_NAME,
    RPY_MODEL_NAME,
    RESIDUAL_MODEL_NAME,
    make_env,
)


def fmt(x, default=0.0, digits=4):
    try:
        return f"{float(x): .{digits}f}"
    except Exception:
        return f"{float(default): .{digits}f}"


def arr(x, digits=4):
    if x is None:
        return None
    return np.round(np.asarray(x, dtype=np.float32), digits)


def model_name_for_stage(stage):
    if stage == "xyz":
        return XYZ_MODEL_NAME
    if stage == "rpy":
        return RPY_MODEL_NAME
    if stage == "residual":
        return RESIDUAL_MODEL_NAME
    raise ValueError(stage)


def load_stage_model(stage, env):
    model_name = model_name_for_stage(stage)
    paths = setup(__file__, f"test_{model_name}", model_name)
    print("loading model:", paths.stable_model_path)
    return PPO.load(paths.stable_model_path, env=env, device="cpu")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage",
        choices=["xyz", "rpy", "residual"],
        required=True,
    )
    parser.add_argument("--steps", type=int, default=800)
    parser.add_argument("--deterministic", action="store_true", default=True)
    args = parser.parse_args()

    xyz_model_path = None
    rpy_model_path = None

    if args.stage in ["rpy", "residual"]:
        xyz_paths = setup(__file__, "test_xyz_dependency", XYZ_MODEL_NAME)
        xyz_model_path = str(xyz_paths.stable_model_path)

    if args.stage == "residual":
        rpy_paths = setup(__file__, "test_rpy_dependency", RPY_MODEL_NAME)
        rpy_model_path = str(rpy_paths.stable_model_path)

    env = make_env(
        stage=args.stage,
        xyz_model_path=xyz_model_path,
        rpy_model_path=rpy_model_path,
        render_mode="human",
    )

    model = load_stage_model(args.stage, env)

    obs, info = env.reset()
    total_reward = 0.0

    print("stage:", args.stage)
    print("target:", arr(info.get("target")))
    print("initial_translation_error:", info.get("initial_translation_error"))
    print("initial_rotation_error:", info.get("initial_rotation_error"))
    print()

    try:
        for step in range(args.steps):
            action, _ = model.predict(obs, deterministic=args.deterministic)
            obs, reward, terminated, truncated, info = env.step(action)

            reward = float(reward)
            total_reward += reward

            raw_action = np.asarray(action, dtype=np.float32).reshape(-1)
            object_pose = info.get("current_object_pose")
            z = object_pose[2] if object_pose is not None else 0.0

            print(
                f"step={step + 1:04d} | "
                f"reward={reward:8.3f} | "
                f"total={total_reward:9.3f} | "
                f"trans={fmt(info.get('current_translation_error'))} | "
                f"rot={fmt(info.get('current_rotation_error'))} | "
                f"z={fmt(z)} | "
                f"z_err={fmt(info.get('z_error'))} | "
                f"trans_prog={fmt(info.get('translation_progress'), digits=6)} | "
                f"rot_prog={fmt(info.get('rotation_progress'), digits=6)} | "
                f"xy_toward={fmt(info.get('actual_xy_toward_goal'), digits=6)} | "
                f"xy_speed={fmt(info.get('actual_xy_speed'), digits=6)} | "
                f"dx={fmt(info.get('chosen_dx'))} | "
                f"dy={fmt(info.get('chosen_dy'))} | "
                f"dz={fmt(info.get('chosen_dz'))} | "
                f"droll={fmt(info.get('chosen_droll'))} | "
                f"dpitch={fmt(info.get('chosen_dpitch'))} | "
                f"dyaw={fmt(info.get('chosen_dyaw'))} | "
                f"action={np.round(raw_action, 3)} | "
                f"action6={arr(info.get('raw_action_6d'), 3)} | "
                f"success={info.get('success')} | "
                f"failure={info.get('failure')} | "
                f"fell={info.get('object_fell')} | "
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
                        "object_fell": info.get("object_fell"),
                        "object_out_of_bounds": info.get("object_out_of_bounds"),
                        "translation_error": info.get("current_translation_error"),
                        "rotation_error": info.get("current_rotation_error"),
                        "target": arr(info.get("target")),
                        "object_pose": arr(object_pose),
                        "executed_waypoint": arr(info.get("executed_waypoint")),
                        "pose_delta_command": arr(info.get("pose_delta_command"), 5),
                        "raw_action": arr(raw_action, 4),
                        "raw_action_6d": arr(info.get("raw_action_6d"), 4),
                    },
                )
                break

    except KeyboardInterrupt:
        print("Stopped.")

    finally:
        env.close()

    print("total_reward:", total_reward)


if __name__ == "__main__":
    main()