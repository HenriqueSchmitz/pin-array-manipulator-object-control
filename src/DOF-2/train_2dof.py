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
from pin_array_manipulator_object_control.objects.object import Pose
from pin_array_manipulator_object_control.rewards.distance_3d import Distance3DRewardModel
from pin_array_manipulator_object_control.routines.multi_target_generator import MultiTargetGenerator


BASE_SEEK_SPEED = 5e-4
MIN_SEEK_SPEED = 1e-4


def define_movement_limit(number_of_pin_sizes: float, config: PinArrayManipulatorConfig) -> float:
    spaces_per_side = config.pins_per_side - 1
    manipulator_size_no_spaces = config.manipulator_size - config.pin_spacing * spaces_per_side
    pin_size = manipulator_size_no_spaces / config.pins_per_side
    return pin_size * number_of_pin_sizes


def calculate_incremental_target(
    observation: np.ndarray,
    target: np.ndarray,
    config: PinArrayManipulatorConfig,
    max_step_in_pin_sizes: float,
) -> Pose:
    obs_obj = PinArrayEnvObservation.from_array(observation, config.pins_per_side)
    final_target_pose = Pose.from_array(target)

    translation = obs_obj.object_pose.translation_to(final_target_pose)
    distance = translation.length()
    max_step = define_movement_limit(max_step_in_pin_sizes, config)

    if distance <= max_step:
        return obs_obj.object_pose + translation

    return obs_obj.object_pose + translation.resize(max_step)


class ResidualWaypointRLEnv(CompositeControlEnv):
    def __init__(self, *args, manipulator_config: PinArrayManipulatorConfig, **kwargs):
        self.wrapper_max_episode_steps = int(kwargs.pop("max_episode_steps", 2000))
        super().__init__(*args, manipulator_config=manipulator_config, **kwargs)

        self.config = manipulator_config
        self.raw_obs_dim = int(np.prod(self.observation_space.shape))

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(10,),
            dtype=np.float32,
        )

        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(2,),
            dtype=np.float32,
        )

        self.max_step_in_pin_sizes = 0.4
        self.max_forward_residual = 0.004
        self.max_lateral_residual = 0.004

        self.base_seek_speed = BASE_SEEK_SPEED
        self.min_seek_speed = MIN_SEEK_SPEED

        self.success_radius = 0.01

        self.step_count = 0
        self.current_raw_obs = None

        # Important: do NOT call this self.current_target.
        # CompositeControlEnv / PinArrayEnv owns self.current_target and expects a Pose.
        self.target_array = None

        self.prev_object_pos = None
        self.prev_dist = None
        self.prev_move_xy = np.zeros(2, dtype=np.float32)

    def reset(self, **kwargs):
        raw_obs, info = super().reset(**kwargs)

        self.step_count = 0
        self.current_raw_obs = self._trim_raw_obs(raw_obs)

        self.target_array = np.asarray(info["target"], dtype=np.float32).copy()

        object_pos = self._object_pos(self.current_raw_obs)

        self.prev_object_pos = object_pos.copy()
        self.prev_dist = self._xy_distance(object_pos, self.target_array)
        self.prev_move_xy = np.zeros(2, dtype=np.float32)

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

        reward, reward_info = self._compute_reward(self.current_raw_obs)

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
        raw_obs = self.current_raw_obs
        obs_obj = self._parse_obs(raw_obs)

        current_pose = obs_obj.object_pose.array().astype(np.float32)
        current_xy = current_pose[:2]

        target = self.target_array.copy()

        nominal_waypoint = calculate_incremental_target(
            observation=raw_obs,
            target=target,
            config=self.config,
            max_step_in_pin_sizes=self.max_step_in_pin_sizes,
        ).array().astype(np.float32)

        goal_vec = target[:2] - current_xy
        goal_dist = float(np.linalg.norm(goal_vec))

        if goal_dist > 1e-8:
            goal_dir = goal_vec / goal_dist
        else:
            goal_dir = np.zeros(2, dtype=np.float32)

        tangent_dir = np.array([-goal_dir[1], goal_dir[0]], dtype=np.float32)

        forward_residual = self.max_forward_residual * float(policy_action[0])
        lateral_residual = self.max_lateral_residual * float(policy_action[1])

        residual_xy = forward_residual * goal_dir + lateral_residual * tangent_dir

        waypoint = nominal_waypoint.copy()
        waypoint[0] += residual_xy[0]
        waypoint[1] += residual_xy[1]
        waypoint[2:] = nominal_waypoint[2:]

        composite_action = np.concatenate(
            [
                np.array([self.base_seek_speed, self.min_seek_speed], dtype=np.float32),
                waypoint.astype(np.float32),
            ]
        ).astype(np.float32)

        return composite_action, {
            "executed_waypoint": waypoint.copy(),
            "nominal_waypoint": nominal_waypoint.copy(),
            "goal_dist_at_plan": goal_dist,
            "chosen_forward": float(forward_residual),
            "chosen_lateral": float(lateral_residual),
            "raw_action": policy_action.copy(),
        }

    def _compute_reward(self, raw_obs):
        object_pos = self._object_pos(raw_obs)

        curr_dist = self._xy_distance(object_pos, self.target_array)
        progress = float(self.prev_dist - curr_dist)

        move_vec = object_pos - self.prev_object_pos
        move_xy_norm = float(np.linalg.norm(move_vec[:2]))

        success = curr_dist <= self.success_radius

        progress_reward = 50.0 * progress
        distance_penalty = -0.5 * curr_dist
        step_penalty = -0.002
        terminal_reward = 10.0 if success else 0.0

        reward_unclipped = (
            progress_reward
            + distance_penalty
            + step_penalty
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

            "progress_reward": float(progress_reward),
            "distance_penalty": float(distance_penalty),
            "step_penalty": float(step_penalty),
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

    env = ResidualWaypointRLEnv(
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

    print("obs space:", env.observation_space)
    print("action space:", env.action_space)

    policy_kwargs = dict(
        net_arch=dict(
            pi=[128, 128],
            vf=[128, 128],
        ),
        log_std_init=-1.0,
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
        ent_coef=0.005,
        vf_coef=0.5,
        max_grad_norm=0.5,
        target_kl=0.03,
        device="cpu",
    )

    model.learn(total_timesteps=10)
    save_model(model, PATHS, "ppo_2dof_simple_waypoint")

    env.close()


if __name__ == "__main__":
    main()