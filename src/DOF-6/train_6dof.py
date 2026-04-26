import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from rl_common import setup, save_model

PATHS = setup(__file__, "ppo_6dof_direct", "ppo_6dof_direct")

import numpy as np
from gymnasium import spaces
from scipy.spatial.transform import Rotation as R

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


class Direct6DOFContactRLEnv(CompositeControlEnv):
    def __init__(self, *args, manipulator_config: PinArrayManipulatorConfig, **kwargs):
        self.wrapper_max_episode_steps = int(kwargs.pop("max_episode_steps", 2000))
        super().__init__(*args, manipulator_config=manipulator_config, **kwargs)

        self.config = manipulator_config
        self.raw_obs_dim = int(np.prod(self.observation_space.shape))
        self.pin_xy = self._pin_xy_grid(manipulator_config)

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(37,),
            dtype=np.float32,
        )

        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(6,),
            dtype=np.float32,
        )

        # Rotations are degrees in Henrique's controller.
        self.max_xyz_step = np.array([0.02, 0.02, 0.001], dtype=np.float32)
        self.max_rpy_step = np.array([0.0, 0.0, 0.0], dtype=np.float32)

        self.base_seek_speed = BASE_SEEK_SPEED
        self.min_seek_speed = MIN_SEEK_SPEED

        self.success_translation_radius = 0.012
        self.success_rotation_radius = 6.0

        # Prevent the learned "sink / collapse" exploit.
        self.failure_z_floor = 0.015
        self.failure_translation_radius = 100.0
        self.failure_rotation_radius = 355.0

        self.step_count = 0
        self.current_raw_obs = None
        self.target_array = None

        self.prev_object_pose = None
        self.prev_translation_error = None
        self.prev_rotation_error = None

        self.prev_pose_delta = np.zeros(6, dtype=np.float32)
        self.last_action = np.zeros(6, dtype=np.float32)

    def reset(self, **kwargs):
        raw_obs, info = super().reset(**kwargs)

        self.step_count = 0
        self.current_raw_obs = self._trim_raw_obs(raw_obs)

        # Freeze target for this wrapper episode.
        self.target_array = np.asarray(info["target"], dtype=np.float32).copy()

        object_pose = self._object_pose_array(self.current_raw_obs)

        self.prev_object_pose = object_pose.copy()
        self.prev_translation_error = self._translation_error(object_pose, self.target_array)
        self.prev_rotation_error = self._rotation_error(object_pose, self.target_array)

        self.prev_pose_delta = np.zeros(6, dtype=np.float32)
        self.last_action = np.zeros(6, dtype=np.float32)

        success = self._is_success(self.prev_translation_error, self.prev_rotation_error)

        out_info = dict(info)
        out_info["target"] = self.target_array.copy()
        out_info["initial_translation_error"] = self.prev_translation_error
        out_info["initial_rotation_error"] = self.prev_rotation_error
        out_info["current_translation_error"] = self.prev_translation_error
        out_info["current_rotation_error"] = self.prev_rotation_error
        out_info["success"] = bool(success)
        out_info["is_success"] = bool(success)
        out_info["failure"] = False

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

        reward, reward_info = self._compute_reward(self.current_raw_obs, policy_action)

        success = bool(reward_info["success"])
        failure = bool(reward_info["failure"])
        timeout = self.step_count >= self.wrapper_max_episode_steps

        # Failure must dominate success. This prevents fake Monitor success_rate.
        if failure:
            success = False
            reward_info["success"] = False
            reward_info["is_success"] = False
            reward -= 50.0
            reward_info["failure_penalty"] = -5.0
        else:
            reward_info["failure_penalty"] = 0.0

        if base_terminated or base_truncated:
            reward -= 10.0
            reward_info["base_failure_penalty"] = -2.0
        else:
            reward_info["base_failure_penalty"] = 0.0

        terminated = bool(success or failure)
        truncated = bool(timeout and not terminated)

        self._update_reward_state(reward_info)

        obs = self._compact_obs(self.current_raw_obs)

        out_info = dict(info)
        out_info.update(debug_info)
        out_info.update(reward_info)

        out_info["target"] = self.target_array.copy()
        out_info["success"] = bool(success)
        out_info["is_success"] = bool(success)
        out_info["failure"] = bool(failure)

        out_info["base_reward"] = float(base_reward)
        out_info["base_terminated"] = bool(base_terminated)
        out_info["base_truncated"] = bool(base_truncated)
        out_info["TimeLimit.truncated"] = bool(truncated)

        return obs, float(reward), terminated, truncated, out_info

    def _action_to_composite_action(self, policy_action):
        current_pose = self._object_pose_array(self.current_raw_obs)

        delta_xyz = policy_action[:3] * self.max_xyz_step
        delta_rpy = policy_action[3:] * self.max_rpy_step

        pose_delta = np.concatenate([delta_xyz, delta_rpy]).astype(np.float32)

        waypoint = current_pose.copy()
        waypoint += pose_delta

        composite_action = np.concatenate(
            [
                np.array([self.base_seek_speed, self.min_seek_speed], dtype=np.float32),
                waypoint.astype(np.float32),
            ]
        ).astype(np.float32)

        return composite_action, {
            "executed_waypoint": waypoint.copy(),
            "chosen_dx": float(delta_xyz[0]),
            "chosen_dy": float(delta_xyz[1]),
            "chosen_dz": float(delta_xyz[2]),
            "chosen_droll": float(delta_rpy[0]),
            "chosen_dpitch": float(delta_rpy[1]),
            "chosen_dyaw": float(delta_rpy[2]),
            "pose_delta_command": pose_delta.copy(),
            "raw_action": policy_action.copy(),
        }

    def _compute_reward(self, raw_obs, policy_action):
        object_pose = self._object_pose_array(raw_obs)

        translation_error = self._translation_error(object_pose, self.target_array)
        rotation_error = self._rotation_error(object_pose, self.target_array)

        translation_progress = float(self.prev_translation_error - translation_error)
        rotation_progress = float(self.prev_rotation_error - rotation_error)

        pose_delta = object_pose - self.prev_object_pose
        move_xyz = pose_delta[:3]
        move_rpy = pose_delta[3:]

        z_error = abs(float(object_pose[2] - self.target_array[2]))

        success = self._is_success(translation_error, rotation_error)
        failure = self._is_failure(object_pose, translation_error, rotation_error)

        # Reward only what actually happened to the object, not what was commanded.
        translation_progress_reward = 120.0 * translation_progress

        # Do not care about rotation until translation is reasonably close.
        if translation_error < 0.08:
            rotation_progress_reward = 0.25 * rotation_progress
            rotation_penalty = -0.003 * rotation_error
        else:
            rotation_progress_reward = 0.0
            rotation_penalty = 0.0

        # Keep distance pressure moderate. Huge absolute penalties create constant clipping.
        translation_penalty = -0.75 * translation_error
        z_penalty = -1.0 * z_error

        action_penalty = -0.0005 * float(np.linalg.norm(policy_action))
        step_penalty = -0.0005

        move_norm = float(np.linalg.norm(move_xyz))

        stuck_penalty = 0.0
        if translation_error > self.success_translation_radius and move_norm < 1e-6:
            stuck_penalty = -0.01

        weak_progress_penalty = 0.0
        if translation_error > 0.05 and translation_progress < 5e-5:
            weak_progress_penalty = -0.01

        terminal_reward = 15.0 if success else 0.0

        reward_unclipped = (
            translation_progress_reward
            + rotation_progress_reward
            + translation_penalty
            + rotation_penalty
            + z_penalty
            + action_penalty
            + step_penalty
            + stuck_penalty
            + weak_progress_penalty
            + terminal_reward
        )

        reward = float(np.clip(reward_unclipped, -2.0, 10.0))

        return reward, {
            "success": bool(success),
            "is_success": bool(success),
            "failure": bool(failure),
            "object_out_of_bounds": bool(translation_error > self.failure_translation_radius),
            "object_fell": bool(object_pose[2] < self.failure_z_floor),

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
            "z_penalty": float(z_penalty),
            "action_penalty": float(action_penalty),
            "step_penalty": float(step_penalty),
            "stuck_penalty": float(stuck_penalty),
            "weak_progress_penalty": float(weak_progress_penalty),
            "terminal_reward": float(terminal_reward),
            "reward_unclipped": float(reward_unclipped),

            "move_norm": float(np.linalg.norm(pose_delta)),
            "translation_move_norm": float(np.linalg.norm(move_xyz)),
            "rotation_move_norm": float(np.linalg.norm(move_rpy)),
            "z_movement": abs(float(pose_delta[2])),
            "z_error": float(z_error),

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

        contact_features = self._contact_features(obs_obj)
        normalized_time = float(self.step_count) / float(max(1, self.wrapper_max_episode_steps))

        return np.concatenate(
            [
                rel_pose.astype(np.float32),
                np.array([translation_dist], dtype=np.float32),
                translation_dir.astype(np.float32),
                object_vel.astype(np.float32),
                self.prev_pose_delta.astype(np.float32),
                self.last_action.astype(np.float32),
                contact_features.astype(np.float32),
                np.array([normalized_time], dtype=np.float32),
            ]
        ).astype(np.float32)

    def _contact_features(self, obs_obj):
        forces = np.asarray(obs_obj.pin_forces, dtype=np.float32).reshape(-1)
        abs_forces = np.abs(forces)

        contact_mask = abs_forces > 1e-8
        contact_count = float(np.sum(contact_mask))
        contact_frac = contact_count / float(max(1, abs_forces.size))

        force_sum = float(np.sum(abs_forces))
        force_max = float(np.max(abs_forces)) if abs_forces.size else 0.0

        xy = self.pin_xy

        if contact_count > 0:
            contact_xy = xy[contact_mask]
            contact_centroid = np.mean(contact_xy, axis=0)
            contact_spread = np.std(contact_xy, axis=0)
        else:
            contact_centroid = np.zeros(2, dtype=np.float32)
            contact_spread = np.zeros(2, dtype=np.float32)

        if force_sum > 1e-8:
            weights = abs_forces / force_sum
            force_centroid = np.sum(xy * weights[:, None], axis=0)
        else:
            force_centroid = np.zeros(2, dtype=np.float32)

        centroid_gap = float(np.linalg.norm(force_centroid - contact_centroid))

        return np.array(
            [
                contact_frac,
                force_sum,
                force_max,
                contact_centroid[0],
                contact_centroid[1],
                contact_spread[0],
                contact_spread[1],
                centroid_gap,
            ],
            dtype=np.float32,
        )

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

    def _is_failure(self, object_pose, translation_error, rotation_error):
        if object_pose[2] < self.failure_z_floor:
            return True
        if translation_error > self.failure_translation_radius:
            return True
        if rotation_error > self.failure_rotation_radius:
            return True
        return False

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

        r_obj = R.from_euler("xyz", object_pose[3:], degrees=True)
        r_tgt = R.from_euler("xyz", target_pose[3:], degrees=True)
        r_err = r_tgt * r_obj.inv()

        return float(np.degrees(r_err.magnitude()))

    def _trim_raw_obs(self, obs):
        raw = np.asarray(obs, dtype=np.float32).reshape(-1)
        return raw[: self.raw_obs_dim].copy()

    def _parse_obs(self, obs):
        raw = self._trim_raw_obs(obs)
        return PinArrayEnvObservation.from_array(raw, self.config.pins_per_side)

    def _pin_xy_grid(self, config):
        spaces_per_side = config.pins_per_side - 1
        pin_size = (config.manipulator_size - config.pin_spacing * spaces_per_side) / config.pins_per_side
        pin_size_spaced = pin_size + config.pin_spacing

        coords = np.linspace(
            -(config.pins_per_side - 1) / 2 * pin_size_spaced,
            (config.pins_per_side - 1) / 2 * pin_size_spaced,
            config.pins_per_side,
        )

        pin_x, pin_y = np.meshgrid(coords, coords, indexing="ij")
        return np.stack([pin_x.flatten(), pin_y.flatten()], axis=1).astype(np.float32)


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

    env = Direct6DOFContactRLEnv(
        simulation_object=obj,
        target_generator=target_generator,
        reward_model=reward_model,
        manipulator_config=config,
        render_mode=render_mode,
        max_episode_steps=500,
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
        log_std_init=-1.6,
    )

    model = PPO(
        "MlpPolicy",
        env,
        policy_kwargs=policy_kwargs,
        verbose=1,
        tensorboard_log=PATHS.tensorboard_log,
        learning_rate=3e-5,
        n_steps=4096 // n_envs,
        batch_size=512,
        n_epochs=8,
        gamma=0.98,
        gae_lambda=0.95,
        clip_range=0.10,
        ent_coef=0.002,
        vf_coef=0.5,
        max_grad_norm=0.5,
        target_kl=0.02,
        device="cpu",
    )

    try:
        model.learn(total_timesteps=1_000_000)
        save_model(model, PATHS, "ppo_6dof_direct")
    finally:
        env.close()


if __name__ == "__main__":
    main()