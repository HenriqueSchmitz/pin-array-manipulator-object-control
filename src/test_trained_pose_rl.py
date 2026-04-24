from stable_baselines3 import PPO

from train_rl_pose import make_env


def main():
    env = make_env()
    env.unwrapped.render_mode = "human"

    # Must load with the same env so SB3 validates the correct obs/action spaces.
    model = PPO.load("ppo_subgoal_pin_array", env=env)
    
    obs, info = env.reset()
    total_reward = 0.0

    try:
        for step in range(5000):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)

            total_reward += float(reward)

            print(
                f"step={step + 1}, "
                f"reward={float(reward):.4f}, "
                f"total={total_reward:.4f}, "
                f"dist={info.get('current_translation_error', None)}, "
                f"success={info.get('success', info.get('is_success', None))}"
            )

            if terminated or truncated:
                print("done:", {"terminated": terminated, "truncated": truncated})
                break

    except KeyboardInterrupt:
        print("Stopped.")

    finally:
        env.close()

    print("total_reward:", total_reward)


if __name__ == "__main__":
    main()