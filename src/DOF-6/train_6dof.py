import sys
from pathlib import Path
SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
from rl_common import setup, save_model
PATHS = setup(__file__, "ppo_6dof", "ppo_6dof_residual")

from datetime import datetime
from pathlib import Path

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


class Residual6DOFRLEnv(CompositeControlEnv):
    def __init__(self, *args, manipulator_config: PinArrayManipulatorConfig, **kwargs):
        self.wrapper_max_episode_steps = int(kwargs.pop("max_episode_steps", 2000))
        super().__init__(*args, manipulator_config=manipulator_config, **kwargs)

        self.config = manipulator_config
        self.raw_obs_dim = int(np.prod(self.observation_space.shape))

        # Observation:
        # rel_pose[6],
        # translation_dist[1],
        # translation_dir[3],
        # object_velocity[6],
        # prev_pose_delta[6],
        # normalized_time[1]
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(23,),
            dtype=np.float32,
        )

        # Full 6DOF residual:
        # [dx, dy, dz, droll, dpitch, dyaw]
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(6,),
            dtype=np.float32,
        )

        self.max_step_in_pin_sizes = 0.4

        # Assumes pose rotation units are radians.
        # Keep these small. This is residual control, not raw pose control.
        self.max_xyz_residual = np.array([0.004, 0.004, 0.002], dtype=np.float32)
        self.max_rpy_residual = np.array([0.035, 0.035, 0.06], dtype=np.float32)

        self.base_seek_speed = BASE_SEEK_SPEED
        self.min_seek_speed = MIN_SEEK_SPEED

        self.success_translation_radius = 0.01
        self.success_rotation_radius = 0.15

        self.step_count = 0
        self.current_raw_obs = None

        # Do NOT call this self.current_target.
        # Base env owns self.current_target and expects a Pose.
        self.target_array = None

        self.prev_object_pose = None
        self.prev_translation_error = None
        self.prev_rotation_error = None
        self.prev_pose_delta = np.zeros(6, dtype=np.float32)

    def reset(self, **kwargs):
        raw_obs, info = super().reset(**kwargs)

        self.step_count = 0
        self.current_raw_obs = self._trim_raw_obs(raw_obs)
        self.target_array = np.asarray(info["target"], dtype=np.float32).copy()

        object_pose = self._object_pose_array(self.current_raw_obs)

        self.prev_object_pose = object_pose.copy()
        self.prev_translation_error = self._translation_error(object_pose, self.target_array)
        self.prev_rotation_error = self._rotation_error(object_pose, self.target_array)
        self.prev_pose_delta = np.zeros(6, dtype=np.float32)

        out_info = dict(info)
        out_info["target"] = self.target_array.copy()
        out_info["initial_translation_error"] = self.prev_translation_error
        out_info["initial_rotation_error"] = self.prev_rotation_error
        out_info["current_translation_error"] = self.prev_translation_error
        out_info["current_rotation_error"] = self.prev_rotation_error
        out_info["success"] = self._is_success(
            self.prev_translation_error,
            self.prev_rotation_error,
        )
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
        object_pose = self._object_pose_array(raw_obs)
        target = self.target_array.copy()

        nominal_waypoint = calculate_incremental_target(
            observation=raw_obs,
            target=target,
            config=self.config,
            max_step_in_pin_sizes=self.max_step_in_pin_sizes,
        ).array().astype(np.float32)

        xyz_residual = policy_action[:3] * self.max_xyz_residual
        rpy_residual = policy_action[3:] * self.max_rpy_residual

        pose_residual = np.concatenate([xyz_residual, rpy_residual]).astype(np.float32)

        waypoint = nominal_waypoint.copy()
        waypoint += pose_residual

        composite_action = np.concatenate(
            [
                np.array([self.base_seek_speed, self.min_seek_speed], dtype=np.float32),
                waypoint.astype(np.float32),
            ]
        ).astype(np.float32)

        return composite_action, {
            "executed_waypoint": waypoint.copy(),
            "nominal_waypoint": nominal_waypoint.copy(),
            "pose_residual": pose_residual.copy(),
            "xyz_residual": xyz_residual.copy(),
            "rpy_residual": rpy_residual.copy(),
            "raw_action": policy_action.copy(),
            "goal_translation_dist_at_plan": self._translation_error(object_pose, target),
            "goal_rotation_dist_at_plan": self._rotation_error(object_pose, target),
        }

    def _compute_reward(self, raw_obs):
        object_pose = self._object_pose_array(raw_obs)

        translation_error = self._translation_error(object_pose, self.target_array)
        rotation_error = self._rotation_error(object_pose, self.target_array)

        translation_progress = float(self.prev_translation_error - translation_error)
        rotation_progress = float(self.prev_rotation_error - rotation_error)

        pose_delta = object_pose - self.prev_object_pose
        translation_move_norm = float(np.linalg.norm(pose_delta[:3]))
        rotation_move_norm = float(np.linalg.norm(pose_delta[3:]))

        success = self._is_success(translation_error, rotation_error)

        translation_progress_reward = 50.0 * translation_progress
        rotation_progress_reward = 2.0 * rotation_progress

        translation_penalty = -0.5 * translation_error
        rotation_penalty = -0.05 * rotation_error

        step_penalty = -0.002
        terminal_reward = 10.0 if success else 0.0

        reward_unclipped = (
            translation_progress_reward
            + rotation_progress_reward
            + translation_penalty
            + rotation_penalty
            + step_penalty
            + terminal_reward
        )

        reward = float(np.clip(reward_unclipped, -5.0, 15.0))

        return reward, {
            "success": bool(success),
            "failure": False,
            "object_out_of_bounds": False,
            "object_fell": False,

            "current_pose_error": translation_error,
            "pose_error": translation_error,

            "current_translation_error": translation_error,
            "distance_to_target": translation_error,
            "current_distance": translation_error,

            "current_rotation_error": rotation_error,
            "rotation_error": rotation_error,

            "translation_progress": translation_progress,
            "progress_xy": translation_progress,
            "rotation_progress": rotation_progress,

            "translation_progress_reward": float(translation_progress_reward),
            "rotation_progress_reward": float(rotation_progress_reward),
            "translation_penalty": float(translation_penalty),
            "rotation_penalty": float(rotation_penalty),
            "step_penalty": float(step_penalty),
            "terminal_reward": float(terminal_reward),
            "reward_unclipped": float(reward_unclipped),

            "move_norm": float(np.linalg.norm(pose_delta)),
            "translation_move_norm": translation_move_norm,
            "rotation_move_norm": rotation_move_norm,
            "z_movement": abs(float(pose_delta[2])),

            "current_object_pose": object_pose.copy(),
            "current_object_pos": object_pose[:3].copy(),
        }

    def _compact_obs(self, raw_obs):
        obs_obj = self._parse_obs(raw_obs)

        object_pose = obs_obj.object_pose.array().astype(np.float32)
        object_vel = obs_obj.object_velocity.array().astype(np.float32)

        rel_pose = self.target_array.astype(np.float32) - object_pose

        rel_xyz = rel_pose[:3]
        translation_dist = float(np.linalg.norm(rel_xyz))

        if translation_dist > 1e-8:
            translation_dir = rel_xyz / translation_dist
        else:
            translation_dir = np.zeros(3, dtype=np.float32)

        normalized_time = float(self.step_count) / float(max(1, self.wrapper_max_episode_steps))

        return np.concatenate(
            [
                rel_pose.astype(np.float32),                         # 6
                np.array([translation_dist], dtype=np.float32),       # 1
                translation_dir.astype(np.float32),                   # 3
                object_vel.astype(np.float32),                        # 6
                self.prev_pose_delta.astype(np.float32),              # 6
                np.array([normalized_time], dtype=np.float32),        # 1
            ]
        ).astype(np.float32)

    def _update_reward_state(self, reward_info):
        new_pose = reward_info["current_object_pose"].copy()

        self.prev_pose_delta = (new_pose - self.prev_object_pose).astype(np.float32)
        self.prev_object_pose = new_pose

        self.prev_translation_error = float(reward_info["current_translation_error"])
        self.prev_rotation_error = float(reward_info["current_rotation_error"])

    def _is_success(self, translation_error, rotation_error):
        return (
            translation_error <= self.success_translation_radius
            and rotation_error <= self.success_rotation_radius
        )

    def _object_pose_array(self, raw_obs):
        obs_obj = self._parse_obs(raw_obs)
        return obs_obj.object_pose.array().astype(np.float32).copy()

    @staticmethod
    def _translation_error(object_pose, target_pose):
        object_pose = np.asarray(object_pose, dtype=np.float32)
        target_pose = np.asarray(target_pose, dtype=np.float32)
        return float(np.linalg.norm(object_pose[:3] - target_pose[:3]))

    @staticmethod
    def _rotation_error(object_pose, target_pose):
        object_pose = np.asarray(object_pose, dtype=np.float32)
        target_pose = np.asarray(target_pose, dtype=np.float32)
        return float(np.linalg.norm(object_pose[3:] - target_pose[3:]))

    def _trim_raw_obs(self, obs):
        raw = np.asarray(obs, dtype=np.float32).reshape(-1)
        return raw[: self.raw_obs_dim].copy()

    def _parse_obs(self, obs):
        raw = self._trim_raw_obs(obs)
        return PinArrayEnvObservation.from_array(raw, self.config.pins_per_side)


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

    obj = Ball(diameter=0.1, starting_z=0.2)

    reward_model = Distance3DRewardModel(manipulator_config=config)

    target_generator = MultiTargetGenerator(
        simulation_object=obj,
        manipulator_config=config,
    )

    env = Residual6DOFRLEnv(
        simulation_object=obj,
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
        log_std_init=-1.2,
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

    model.learn(total_timesteps=200_000)
    save_model(model, PATHS, "ppo_6dof_residual")
    env.close()


if __name__ == "__main__":
    main()