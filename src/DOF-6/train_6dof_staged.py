import argparse
import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from rl_common import setup, save_model

import numpy as np
from gymnasium import spaces
from scipy.spatial.transform import Rotation as R

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from pin_array_manipulator_object_control.environment.composite_control_env import CompositeControlEnv
from pin_array_manipulator_object_control.manipulator.observation import PinArrayEnvObservation
from pin_array_manipulator_object_control.manipulator.pin_array_manipulator import PinArrayManipulatorConfig
from pin_array_manipulator_object_control.objects.ball import Ball
from pin_array_manipulator_object_control.rewards.distance_3d import Distance3DRewardModel
from pin_array_manipulator_object_control.routines.multi_target_generator import MultiTargetGenerator


BASE_SEEK_SPEED = 5e-4
MIN_SEEK_SPEED = 1e-4


XYZ_MODEL_NAME = "ppo_6dof_xyz"
RPY_MODEL_NAME = "ppo_6dof_rpy"
RESIDUAL_MODEL_NAME = "ppo_6dof_residual"


class Staged6DOFBaseEnv(CompositeControlEnv):
    def __init__(self, *args, manipulator_config: PinArrayManipulatorConfig, **kwargs):
        self.wrapper_max_episode_steps = int(kwargs.pop("max_episode_steps", 800))
        super().__init__(*args, manipulator_config=manipulator_config, **kwargs)

        self.config = manipulator_config
        self.raw_obs_dim = int(np.prod(self.observation_space.shape))

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(29,),
            dtype=np.float32,
        )

        self.max_xyz_step = np.array([0.04, 0.04, 0.0010], dtype=np.float32)
        self.max_rpy_step = np.array([0.25, 0.25, 0.50], dtype=np.float32)

        self.base_seek_speed = BASE_SEEK_SPEED
        self.min_seek_speed = MIN_SEEK_SPEED

        self.success_translation_radius = 0.012
        self.success_rotation_radius = 6.0

        self.failure_z_floor = 0.012
        self.failure_translation_radius = 0.50
        self.failure_rotation_radius = 120.0

        self.step_count = 0
        self.current_raw_obs = None
        self.target_array = None

        self.prev_object_pose = None
        self.prev_translation_error = None
        self.prev_rotation_error = None

        self.prev_pose_delta = np.zeros(6, dtype=np.float32)
        self.last_action_6d = np.zeros(6, dtype=np.float32)

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
        self.last_action_6d = np.zeros(6, dtype=np.float32)

        out_info = dict(info)
        out_info["target"] = self.target_array.copy()
        out_info["initial_translation_error"] = self.prev_translation_error
        out_info["initial_rotation_error"] = self.prev_rotation_error
        out_info["current_translation_error"] = self.prev_translation_error
        out_info["current_rotation_error"] = self.prev_rotation_error
        out_info["success"] = False
        out_info["is_success"] = False
        out_info["failure"] = False

        return self._compact_obs(self.current_raw_obs), out_info

    def _step_full_action(self, action_6d):
        self.step_count += 1

        action_6d = np.clip(np.asarray(action_6d, dtype=np.float32), -1.0, 1.0)
        self.last_action_6d = action_6d.copy()

        composite_action, debug_info = self._action_to_composite_action(action_6d)

        if self.render_mode == "human":
            try:
                self.update_debug_visuals(composite_action[2:])
            except Exception:
                pass

        raw_obs, base_reward, base_terminated, base_truncated, info = super().step(composite_action)
        self.current_raw_obs = self._trim_raw_obs(raw_obs)

        reward, reward_info = self._compute_reward(self.current_raw_obs, action_6d)

        success = bool(reward_info["success"])
        failure = bool(reward_info["failure"])
        timeout = self.step_count >= self.wrapper_max_episode_steps

        if failure:
            success = False
            reward = -5.0
            reward_info["success"] = False
            reward_info["is_success"] = False
            reward_info["failure_penalty"] = -5.0
        else:
            reward_info["failure_penalty"] = 0.0

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

    def _action_to_composite_action(self, action_6d):
        current_pose = self._object_pose_array(self.current_raw_obs)

        delta_xyz = action_6d[:3] * self.max_xyz_step
        delta_rpy = action_6d[3:] * self.max_rpy_step

        min_command_z = self.target_array[2] - 0.005
        proposed_z = current_pose[2] + delta_xyz[2]
        if proposed_z < min_command_z:
            delta_xyz[2] = min_command_z - current_pose[2]

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
            "raw_action_6d": action_6d.copy(),
            "z_guard_active": bool(proposed_z < min_command_z),
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
                rel_pose.astype(np.float32),
                np.array([translation_dist], dtype=np.float32),
                translation_dir.astype(np.float32),
                object_vel.astype(np.float32),
                self.prev_pose_delta.astype(np.float32),
                self.last_action_6d.astype(np.float32),
                np.array([normalized_time], dtype=np.float32),
            ]
        ).astype(np.float32)

    def _update_reward_state(self, reward_info):
        new_pose = reward_info["current_object_pose"].copy()
        self.prev_pose_delta = (new_pose - self.prev_object_pose).astype(np.float32)
        self.prev_object_pose = new_pose
        self.prev_translation_error = float(reward_info["current_translation_error"])
        self.prev_rotation_error = float(reward_info["current_rotation_error"])

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
    def _xy_error(object_pose, target_pose):
        object_pose = np.asarray(object_pose, dtype=np.float32)
        target_pose = np.asarray(target_pose, dtype=np.float32)
        return float(np.linalg.norm(object_pose[:2] - target_pose[:2]))

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


class XYZTransportEnv(Staged6DOFBaseEnv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)

    def step(self, xyz_action):
        action_6d = np.zeros(6, dtype=np.float32)
        action_6d[:3] = np.asarray(xyz_action, dtype=np.float32)
        return self._step_full_action(action_6d)

    def _compute_reward(self, raw_obs, action_6d):
        object_pose = self._object_pose_array(raw_obs)

        translation_error = self._translation_error(object_pose, self.target_array)
        xy_error = self._xy_error(object_pose, self.target_array)
        z_error = abs(float(object_pose[2] - self.target_array[2]))
        rotation_error = self._rotation_error(object_pose, self.target_array)

        translation_progress = float(self.prev_translation_error - translation_error)

        pose_delta = object_pose - self.prev_object_pose
        move_xyz = pose_delta[:3]
        move_xy = move_xyz[:2]

        goal_vec_xy = self.target_array[:2] - object_pose[:2]
        goal_dist_xy = float(np.linalg.norm(goal_vec_xy))
        if goal_dist_xy > 1e-8:
            goal_dir_xy = goal_vec_xy / goal_dist_xy
        else:
            goal_dir_xy = np.zeros(2, dtype=np.float32)

        actual_xy_toward_goal = float(np.dot(move_xy, goal_dir_xy))
        actual_xy_speed = float(np.linalg.norm(move_xy))

        success = translation_error <= self.success_translation_radius
        failure = self._is_failure(object_pose, translation_error, rotation_error)

        reward_unclipped = (
            250.0 * translation_progress
            + 800.0 * actual_xy_toward_goal
            - 1.0 * translation_error
            - 2.0 * max(0.0, float(self.target_array[2] - object_pose[2]))
            - 0.001
        )

        if translation_error > 0.05 and actual_xy_speed < 2e-5:
            reward_unclipped -= 1.0

        terminal_reward = 25.0 if success else 0.0
        reward_unclipped += terminal_reward

        reward = float(np.clip(reward_unclipped, -5.0, 10.0))

        return reward, {
            "success": bool(success),
            "is_success": bool(success),
            "failure": bool(failure),
            "object_fell": bool(object_pose[2] < self.failure_z_floor),
            "object_out_of_bounds": bool(translation_error > self.failure_translation_radius),
            "current_translation_error": translation_error,
            "current_rotation_error": rotation_error,
            "translation_progress": translation_progress,
            "rotation_progress": 0.0,
            "xy_error": xy_error,
            "z_error": z_error,
            "actual_xy_toward_goal": actual_xy_toward_goal,
            "actual_xy_speed": actual_xy_speed,
            "reward_unclipped": float(reward_unclipped),
            "terminal_reward": float(terminal_reward),
            "current_object_pose": object_pose.copy(),
            "current_object_pos": object_pose[:3].copy(),
        }


class RPYRotationEnv(Staged6DOFBaseEnv):
    def __init__(self, *args, xyz_model_path: str, **kwargs):
        super().__init__(*args, **kwargs)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
        self.xyz_model = PPO.load(xyz_model_path, device="cpu")

    def step(self, rpy_action):
        obs = self._compact_obs(self.current_raw_obs)
        xyz_action, _ = self.xyz_model.predict(obs, deterministic=True)

        action_6d = np.zeros(6, dtype=np.float32)
        action_6d[:3] = np.asarray(xyz_action, dtype=np.float32).reshape(-1)[:3]
        action_6d[3:] = np.asarray(rpy_action, dtype=np.float32).reshape(-1)[:3]

        return self._step_full_action(action_6d)

    def _compute_reward(self, raw_obs, action_6d):
        object_pose = self._object_pose_array(raw_obs)

        translation_error = self._translation_error(object_pose, self.target_array)
        rotation_error = self._rotation_error(object_pose, self.target_array)

        translation_progress = float(self.prev_translation_error - translation_error)
        rotation_progress = float(self.prev_rotation_error - rotation_error)

        pose_delta = object_pose - self.prev_object_pose
        z_below_target = max(0.0, float(self.target_array[2] - object_pose[2]))

        success = (
            translation_error <= self.success_translation_radius
            and rotation_error <= self.success_rotation_radius
        )
        failure = self._is_failure(object_pose, translation_error, rotation_error)

        reward_unclipped = (
            0.35 * rotation_progress
            - 0.010 * rotation_error
            - 2.0 * max(0.0, translation_error - self.success_translation_radius)
            - 3.0 * z_below_target
            - 0.001
        )

        # Rotation policy may not destroy translation.
        if translation_progress < -1e-4:
            reward_unclipped -= 0.5

        terminal_reward = 50.0 if success else 0.0
        reward_unclipped += terminal_reward

        reward = float(np.clip(reward_unclipped, -5.0, 10.0))

        return reward, {
            "success": bool(success),
            "is_success": bool(success),
            "failure": bool(failure),
            "object_fell": bool(object_pose[2] < self.failure_z_floor),
            "object_out_of_bounds": bool(translation_error > self.failure_translation_radius),
            "current_translation_error": translation_error,
            "current_rotation_error": rotation_error,
            "translation_progress": translation_progress,
            "rotation_progress": rotation_progress,
            "reward_unclipped": float(reward_unclipped),
            "terminal_reward": float(terminal_reward),
            "current_object_pose": object_pose.copy(),
            "current_object_pos": object_pose[:3].copy(),
        }


class Residual6DOFEnv(Staged6DOFBaseEnv):
    def __init__(self, *args, xyz_model_path: str, rpy_model_path: str, **kwargs):
        super().__init__(*args, **kwargs)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(6,), dtype=np.float32)
        self.xyz_model = PPO.load(xyz_model_path, device="cpu")
        self.rpy_model = PPO.load(rpy_model_path, device="cpu")
        self.residual_scale = 0.25

    def step(self, residual_action):
        obs = self._compact_obs(self.current_raw_obs)

        xyz_action, _ = self.xyz_model.predict(obs, deterministic=True)
        rpy_action, _ = self.rpy_model.predict(obs, deterministic=True)

        base_action = np.zeros(6, dtype=np.float32)
        base_action[:3] = np.asarray(xyz_action, dtype=np.float32).reshape(-1)[:3]
        base_action[3:] = np.asarray(rpy_action, dtype=np.float32).reshape(-1)[:3]

        residual_action = np.asarray(residual_action, dtype=np.float32).reshape(-1)[:6]
        action_6d = np.clip(base_action + self.residual_scale * residual_action, -1.0, 1.0)

        return self._step_full_action(action_6d)

    def _compute_reward(self, raw_obs, action_6d):
        object_pose = self._object_pose_array(raw_obs)

        translation_error = self._translation_error(object_pose, self.target_array)
        rotation_error = self._rotation_error(object_pose, self.target_array)

        translation_progress = float(self.prev_translation_error - translation_error)
        rotation_progress = float(self.prev_rotation_error - rotation_error)

        pose_delta = object_pose - self.prev_object_pose
        move_xy = pose_delta[:2]

        goal_vec_xy = self.target_array[:2] - object_pose[:2]
        goal_dist_xy = float(np.linalg.norm(goal_vec_xy))
        if goal_dist_xy > 1e-8:
            goal_dir_xy = goal_vec_xy / goal_dist_xy
        else:
            goal_dir_xy = np.zeros(2, dtype=np.float32)

        actual_xy_toward_goal = float(np.dot(move_xy, goal_dir_xy))
        z_below_target = max(0.0, float(self.target_array[2] - object_pose[2]))

        success = (
            translation_error <= self.success_translation_radius
            and rotation_error <= self.success_rotation_radius
        )
        failure = self._is_failure(object_pose, translation_error, rotation_error)

        reward_unclipped = (
            200.0 * translation_progress
            + 400.0 * actual_xy_toward_goal
            + 0.20 * rotation_progress
            - 1.0 * translation_error
            - 0.004 * rotation_error
            - 3.0 * z_below_target
            - 0.001
        )

        terminal_reward = 75.0 if success else 0.0
        reward_unclipped += terminal_reward

        reward = float(np.clip(reward_unclipped, -5.0, 10.0))

        return reward, {
            "success": bool(success),
            "is_success": bool(success),
            "failure": bool(failure),
            "object_fell": bool(object_pose[2] < self.failure_z_floor),
            "object_out_of_bounds": bool(translation_error > self.failure_translation_radius),
            "current_translation_error": translation_error,
            "current_rotation_error": rotation_error,
            "translation_progress": translation_progress,
            "rotation_progress": rotation_progress,
            "actual_xy_toward_goal": actual_xy_toward_goal,
            "reward_unclipped": float(reward_unclipped),
            "terminal_reward": float(terminal_reward),
            "current_object_pose": object_pose.copy(),
            "current_object_pos": object_pose[:3].copy(),
        }


def build_base_kwargs(render_mode=None):
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

    return dict(
        simulation_object=obj,
        target_generator=target_generator,
        reward_model=reward_model,
        manipulator_config=config,
        render_mode=render_mode,
        max_episode_steps=800,
    )


def make_env(stage, xyz_model_path=None, rpy_model_path=None, render_mode=None):
    kwargs = build_base_kwargs(render_mode=render_mode)

    if stage == "xyz":
        env = XYZTransportEnv(**kwargs)
    elif stage == "rpy":
        env = RPYRotationEnv(**kwargs, xyz_model_path=xyz_model_path)
    elif stage == "residual":
        env = Residual6DOFEnv(**kwargs, xyz_model_path=xyz_model_path, rpy_model_path=rpy_model_path)
    else:
        raise ValueError(f"Unknown stage: {stage}")

    return Monitor(env)


def make_one_env(stage, xyz_model_path=None, rpy_model_path=None, render_mode=None):
    def _init():
        return make_env(
            stage=stage,
            xyz_model_path=xyz_model_path,
            rpy_model_path=rpy_model_path,
            render_mode=render_mode,
        )

    return _init


def train(stage):
    if stage == "xyz":
        paths = setup(__file__, XYZ_MODEL_NAME, XYZ_MODEL_NAME)
        xyz_model_path = None
        rpy_model_path = None
        total_timesteps = 400_000
        log_std_init = -0.5
        ent_coef = 0.002

    elif stage == "rpy":
        paths = setup(__file__, RPY_MODEL_NAME, RPY_MODEL_NAME)
        xyz_paths = setup(__file__, XYZ_MODEL_NAME, XYZ_MODEL_NAME)
        xyz_model_path = str(xyz_paths.stable_model_path)
        rpy_model_path = None
        total_timesteps = 400_000
        log_std_init = -0.8
        ent_coef = 0.001

    elif stage == "residual":
        paths = setup(__file__, RESIDUAL_MODEL_NAME, RESIDUAL_MODEL_NAME)
        xyz_paths = setup(__file__, XYZ_MODEL_NAME, XYZ_MODEL_NAME)
        rpy_paths = setup(__file__, RPY_MODEL_NAME, RPY_MODEL_NAME)
        xyz_model_path = str(xyz_paths.stable_model_path)
        rpy_model_path = str(rpy_paths.stable_model_path)
        total_timesteps = 300_000
        log_std_init = -1.2
        ent_coef = 0.0005

    else:
        raise ValueError(stage)

    env = DummyVecEnv(
        [
            make_one_env(
                stage=stage,
                xyz_model_path=xyz_model_path,
                rpy_model_path=rpy_model_path,
            )
        ]
    )

    print("stage:", stage)
    print("run dir:", paths.run_dir)
    print("model path:", paths.stable_model_path)
    print("obs space:", env.observation_space)
    print("action space:", env.action_space)

    policy_kwargs = dict(
        net_arch=dict(
            pi=[256, 256],
            vf=[256, 256],
        ),
        log_std_init=log_std_init,
    )

    model = PPO(
        "MlpPolicy",
        env,
        policy_kwargs=policy_kwargs,
        verbose=1,
        tensorboard_log=paths.tensorboard_log,
        learning_rate=7e-5,
        n_steps=4096,
        batch_size=512,
        n_epochs=8,
        gamma=0.98,
        gae_lambda=0.95,
        clip_range=0.12,
        ent_coef=ent_coef,
        vf_coef=0.5,
        max_grad_norm=0.5,
        target_kl=0.025,
        device="cpu",
    )

    try:
        model.learn(total_timesteps=total_timesteps)
        save_model(model, paths, paths.model_name)
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage",
        choices=["xyz", "rpy", "residual"],
        required=True,
        help="Training stage to run.",
    )
    args = parser.parse_args()
    train(args.stage)


if __name__ == "__main__":
    main()