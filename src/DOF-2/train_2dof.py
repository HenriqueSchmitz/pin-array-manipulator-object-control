import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from rl_common import setup, save_model

PATHS = setup(__file__, "ppo_2dof", "ppo_2dof_simple_waypoint")

import numpy as np
from gymnasium import spaces

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv

from pin_array_manipulator_object_control.environment.composite_control_env import CompositeControlEnv
from pin_array_manipulator_object_control.manipulator.observation import PinArrayEnvObservation
from pin_array_manipulator_object_control.manipulator.pin_array_manipulator import PinArrayManipulatorConfig
from pin_array_manipulator_object_control.objects.ball import Ball
from pin_array_manipulator_object_control.rewards.distance_3d import Distance3DRewardModel
from pin_array_manipulator_object_control.routines.multi_target_generator import MultiTargetGenerator


BASE_SEEK_SPEED = 5e-4
MIN_SEEK_SPEED = 1e-4


class DirectXYRLEnv(CompositeControlEnv):
    """
    No-cheat 2DOF env.

    PPO directly chooses global XY waypoint deltas.
    Henrique's CompositeControlEnv is still used only as the pose-to-pin controller.
    """

    def __init__(self, *args, manipulator_config: PinArrayManipulatorConfig, **kwargs):
        self.wrapper_max_episode_steps = int(kwargs.pop("max_episode_steps", 2000))
        super().__init__(*args, manipulator_config=manipulator_config, **kwargs)

        self.config = manipulator_config
        self.raw_obs_dim = int(np.prod(self.observation_space.shape))

        # [
        #   rel_x, rel_y, dist,
        #   goal_dir_x, goal_dir_y,
        #   object_vel_x, object_vel_y,
        #   prev_move_x, prev_move_y,
        #   last_action_x, last_action_y,
        #   normalized_time
        # ]
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(12,),
            dtype=np.float32,
        )

        # PPO chooses raw global dx, dy waypoint command.
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(2,),
            dtype=np.float32,
        )

        self.max_xy_step = 0.012

        self.base_seek_speed = BASE_SEEK_SPEED
        self.min_seek_speed = MIN_SEEK_SPEED

        self.success_radius = 0.01

        self.step_count = 0
        self.current_raw_obs = None
        self.target_array = None

        self.prev_object_pos = None
        self.prev_dist = None
        self.prev_move_xy = np.zeros(2, dtype=np.float32)
        self.last_action = np.zeros(2, dtype=np.float32)

    def reset(self, **kwargs):
        raw_obs, info = super().reset(**kwargs)

        self.step_count = 0
        self.current_raw_obs = self._trim_raw_obs(raw_obs)
        self.target_array = np.asarray(info["target"], dtype=np.float32).copy()

        object_pos = self._object_pos(self.current_raw_obs)

        self.prev_object_pos = object_pos.copy()
        self.prev_dist = self._xy_distance(object_pos, self.target_array)
        self.prev_move_xy = np.zeros(2, dtype=np.float32)
        self.last_action = np.zeros(2, dtype=np.float32)

        out_info = dict(info)
        out_info["target"] = self.target_array.copy()
        out_info["initial_distance"] = self.prev_dist
        out_info["current_translation_error"] = self.prev_dist
        out_info["distance_to_target"] = self.prev_dist
        out_info["success"] = self.prev_dist <= self.success_radius
        out_info["is_success"] = out_info["success"]

        return self._compact_obs(self.current_raw_obs), out_info

    def step(self, action):
        self.step_count += 1

        policy_action = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
        self.last_action = policy_action.copy()

        composite_action, debug_info = self._action_to_composite_action(policy_action)

        if self.render_mode == "human":
            try:
                self.update_debug_visuals(composite_action[2:])
            except Exception:
                pass

        raw_obs, base_reward, base_terminated, base_truncated, info = super().step(composite_action)

        self.current_raw_obs = self._trim_raw_obs(raw_obs)

        if "target" in info:
            self.target_array = np.asarray(info["target"], dtype=np.float32).copy()

        reward, reward_info = self._compute_reward(self.current_raw_obs, policy_action)

        success = bool(reward_info["success"])
        timeout = self.step_count >= self.wrapper_max_episode_steps

        terminated = bool(success or base_terminated)
        truncated = bool(base_truncated or (timeout and not terminated))

        self._update_reward_state(reward_info)

        obs = self._compact_obs(self.current_raw_obs)

        out_info = dict(info)
        out_info.update(debug_info)
        out_info.update(reward_info)
        out_info["target"] = self.target_array.copy()
        out_info["is_success"] = success
        out_info["base_reward"] = float(base_reward)
        out_info["base_terminated"] = bool(base_terminated)
        out_info["base_truncated"] = bool(base_truncated)
        out_info["TimeLimit.truncated"] = bool(timeout and not terminated)

        return obs, reward, terminated, truncated, out_info

    def _action_to_composite_action(self, policy_action):
        obs_obj = self._parse_obs(self.current_raw_obs)

        current_pose = obs_obj.object_pose.array().astype(np.float32)
        current_pos = current_pose[:3]

        delta_xy = policy_action * self.max_xy_step

        waypoint = current_pose.copy()
        waypoint[0] = current_pos[0] + delta_xy[0]
        waypoint[1] = current_pos[1] + delta_xy[1]

        # Keep z + orientation unchanged. This is honest 2DOF XY control.
        waypoint[2:] = current_pose[2:]

        composite_action = np.concatenate(
            [
                np.array([self.base_seek_speed, self.min_seek_speed], dtype=np.float32),
                waypoint.astype(np.float32),
            ]
        ).astype(np.float32)

        return composite_action, {
            "executed_waypoint": waypoint.copy(),
            "chosen_dx": float(delta_xy[0]),
            "chosen_dy": float(delta_xy[1]),
            "raw_action": policy_action.copy(),
        }

    def _compute_reward(self, raw_obs, policy_action):
        object_pos = self._object_pos(raw_obs)

        curr_dist = self._xy_distance(object_pos, self.target_array)
        progress = float(self.prev_dist - curr_dist)

        move_vec = object_pos - self.prev_object_pos
        move_xy = move_vec[:2]
        move_xy_norm = float(np.linalg.norm(move_xy))

        goal_vec = self.target_array[:2] - object_pos[:2]
        goal_dist = float(np.linalg.norm(goal_vec))

        if goal_dist > 1e-8:
            goal_dir = goal_vec / goal_dist
        else:
            goal_dir = np.zeros(2, dtype=np.float32)

        # This rewards actual object movement toward the goal,
        # not whether the commanded action points at the goal.
        movement_toward_goal = float(np.dot(move_xy, goal_dir))

        success = curr_dist <= self.success_radius

        progress_reward = 80.0 * progress
        toward_goal_reward = 20.0 * movement_toward_goal
        distance_penalty = -0.4 * curr_dist
        action_penalty = -0.01 * float(np.linalg.norm(policy_action))
        step_penalty = -0.002

        stuck_penalty = 0.0
        if curr_dist > self.success_radius and move_xy_norm < 1e-5:
            stuck_penalty = -0.01

        terminal_reward = 10.0 if success else 0.0

        reward_unclipped = (
            progress_reward
            + toward_goal_reward
            + distance_penalty
            + action_penalty
            + step_penalty
            + stuck_penalty
            + terminal_reward
        )

        reward = float(np.clip(reward_unclipped, -5.0, 15.0))

        return reward, {
            "success": bool(success),
            "failure": False,
            "object_out_of_bounds": False,
            "object_fell": False,

            "current_pose_error": curr_dist,
            "pose_error": curr_dist,
            "current_translation_error": curr_dist,
            "distance_to_target": curr_dist,
            "current_distance": curr_dist,

            "translation_progress": progress,
            "progress_xy": progress,
            "movement_toward_goal": movement_toward_goal,

            "progress_reward": float(progress_reward),
            "toward_goal_reward": float(toward_goal_reward),
            "distance_penalty": float(distance_penalty),
            "action_penalty": float(action_penalty),
            "step_penalty": float(step_penalty),
            "stuck_penalty": float(stuck_penalty),
            "terminal_reward": float(terminal_reward),
            "reward_unclipped": float(reward_unclipped),

            "move_norm": float(np.linalg.norm(move_vec)),
            "move_xy_norm": move_xy_norm,
            "z_movement": abs(float(move_vec[2])),

            "current_object_pos": object_pos.copy(),
        }

    def _compact_obs(self, raw_obs):
        obs_obj = self._parse_obs(raw_obs)

        object_pose = obs_obj.object_pose.array().astype(np.float32)
        object_pos = object_pose[:3]
        object_vel = obs_obj.object_velocity.array().astype(np.float32)

        rel_xy = self.target_array[:2].astype(np.float32) - object_pos[:2]
        dist = float(np.linalg.norm(rel_xy))

        if dist > 1e-8:
            goal_dir = rel_xy / dist
        else:
            goal_dir = np.zeros(2, dtype=np.float32)

        normalized_time = float(self.step_count) / float(max(1, self.wrapper_max_episode_steps))

        return np.array(
            [
                rel_xy[0],
                rel_xy[1],
                dist,
                goal_dir[0],
                goal_dir[1],
                object_vel[0],
                object_vel[1],
                self.prev_move_xy[0],
                self.prev_move_xy[1],
                self.last_action[0],
                self.last_action[1],
                normalized_time,
            ],
            dtype=np.float32,
        )

    def _update_reward_state(self, reward_info):
        new_pos = reward_info["current_object_pos"].copy()
        self.prev_move_xy = (new_pos[:2] - self.prev_object_pos[:2]).astype(np.float32)
        self.prev_object_pos = new_pos
        self.prev_dist = float(reward_info["current_translation_error"])

    def _object_pos(self, raw_obs):
        obs_obj = self._parse_obs(raw_obs)
        return obs_obj.object_pose.array().astype(np.float32)[:3].copy()

    def _trim_raw_obs(self, obs):
        raw = np.asarray(obs, dtype=np.float32).reshape(-1)
        return raw[: self.raw_obs_dim].copy()

    def _parse_obs(self, obs):
        raw = self._trim_raw_obs(obs)
        return PinArrayEnvObservation.from_array(raw, self.config.pins_per_side)

    @staticmethod
    def _xy_distance(a, b):
        a = np.asarray(a, dtype=np.float32)
        b = np.asarray(b, dtype=np.float32)
        return float(np.linalg.norm(a[:2] - b[:2]))


def make_env(render_mode=None):
    config = PinArrayManipulatorConfig(
        manipulator_size=1.0,
        pins_per_side=15,
        pin_height=0.15,
        actuation_length=0.1,
        pin_spacing=0.001,
        has_wall=True,
        rounded_pins=True,
    )

    ball = Ball(diameter=0.1, starting_z=0.2)
    reward_model = Distance3DRewardModel(manipulator_config=config)

    target_generator = MultiTargetGenerator(
        simulation_object=ball,
        manipulator_config=config,
    )

    env = DirectXYRLEnv(
        simulation_object=ball,
        target_generator=target_generator,
        reward_model=reward_model,
        manipulator_config=config,
        render_mode=render_mode,
        max_episode_steps=2000,
    )

    return Monitor(env)


def make_one_env(rank, render_mode=None):
    def _init():
        env = make_env(render_mode=render_mode)
        env.action_space.seed(rank)
        env.observation_space.seed(rank)
        return env

    return _init


def main():
    n_envs = 8

    env = SubprocVecEnv([make_one_env(i) for i in range(n_envs)])

    print("run dir:", PATHS.run_dir)
    print("obs space:", env.observation_space)
    print("action space:", env.action_space)

    policy_kwargs = dict(
        net_arch=dict(
            pi=[256, 256],
            vf=[256, 256],
        ),
        log_std_init=-0.5,
    )

    model = PPO(
        "MlpPolicy",
        env,
        policy_kwargs=policy_kwargs,
        verbose=1,
        tensorboard_log=PATHS.tensorboard_log,
        learning_rate=1e-4,
        n_steps=4096 // n_envs,
        batch_size=512,
        n_epochs=8,
        gamma=0.98,
        gae_lambda=0.95,
        clip_range=0.15,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        target_kl=0.03,
        device="cpu",
    )

    try:
        model.learn(total_timesteps=200_000)
        save_model(model, PATHS, "ppo_2dof_simple_waypoint")
    finally:
        env.close()


if __name__ == "__main__":
    main()