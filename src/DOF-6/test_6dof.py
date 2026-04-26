import sys
from pathlib import Path
SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
from rl_common import setup, load_model
PATHS = setup(__file__, "test_6dof", "ppo_6dof_residual")

import numpy as np

from train_6dof import make_env


def main():
    env = make_env(render_mode="human")
    model = load_model(PATHS, env=env)

    obs, info = env.reset()
    total_reward = 0.0

    try:
        for step in range(2000):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)

            total_reward += float(reward)

            action = np.asarray(action).reshape(-1)

            print(
                f"step={step + 1:04d} | "
                f"reward={float(reward):8.3f} | "
                f"total={total_reward:9.3f} | "
                f"trans_dist={info.get('current_translation_error', 0.0):.4f} | "
                f"rot_dist={info.get('current_rotation_error', 0.0):.4f} | "
                f"trans_prog={info.get('translation_progress', 0.0): .6f} | "
                f"rot_prog={info.get('rotation_progress', 0.0): .6f} | "
                f"dx={action[0]: .3f} | "
                f"dy={action[1]: .3f} | "
                f"dz={action[2]: .3f} | "
                f"dr={action[3]: .3f} | "
                f"dp={action[4]: .3f} | "
                f"dyaw={action[5]: .3f} | "
                f"success={info.get('success')} | "
                f"base_trunc={info.get('base_truncated')}"
            )

            if terminated or truncated:
                print(
                    "done:",
                    {
                        "terminated": terminated,
                        "truncated": truncated,
                        "success": info.get("success"),
                        "translation_error": info.get("current_translation_error"),
                        "rotation_error": info.get("current_rotation_error"),
                        "target": info.get("target"),
                        "object_pose": info.get("current_object_pose"),
                        "executed_waypoint": info.get("executed_waypoint"),
                        "pose_residual": info.get("pose_residual"),
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