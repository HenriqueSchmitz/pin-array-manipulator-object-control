from stable_baselines3 import PPO

from train_rl_pose import make_env


MODEL_PATH = "ppo_simple_waypoint"


def main():
    env = make_env(render_mode="human")
    model = PPO.load(MODEL_PATH, env=env)

    obs, info = env.reset()
    total_reward = 0.0

    try:
        for step in range(2000):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)

            total_reward += float(reward)

            print(
                f"step={step + 1:04d} | "
                f"reward={float(reward):8.3f} | "
                f"total={total_reward:9.3f} | "
                f"dist={info.get('current_translation_error', 0.0):.4f} | "
                f"progress={info.get('translation_progress', 0.0): .6f} | "
                f"forward={info.get('chosen_forward', 0.0):.4f} | "
                f"lateral={info.get('chosen_lateral', 0.0): .4f} | "
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
                        "dist": info.get("current_translation_error"),
                        "target": info.get("target"),
                        "object_pos": info.get("current_object_pos"),
                        "executed_waypoint": info.get("executed_waypoint"),
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