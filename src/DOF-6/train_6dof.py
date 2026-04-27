import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from rl_common import setup, save_model

PATHS = setup(__file__, "ppo_6dof_twist_v3", "ppo_6dof_twist_v3")

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

N_ENVS = 8
TOTAL_TIMESTEPS = 1_000_000


class ReliableSphere6DOFEnv(CompositeControlEnv):
    """
    Reliable sphere baseline.

    Action is still 6D:
        [dx, dy, dz_unused, drotvec_x, drotvec_y, drotvec_z]

    But for a sphere:
        - XY is the real task.
        - Z is held at current object height.
        - Rotation is logged but not required for success.
        - Rotation command is frozen by default because sphere orientation is not reliable.
    """

    def __init__(self, *args, manipulator_config: PinArrayManipulatorConfig, **kwargs):
        self.wrapper_max_episode_steps = int(kwargs.pop("max_episode_steps", 2000))
        super().__init__(*args, manipulator_config=manipulator_config, **kwargs)

        self.config = manipulator_config
        self.raw_obs_dim = int(np.prod(self.observation_space.shape))

        # [
        #   rel_xy: 2
        #   dist: 1
        #   goal_dir_xy: 2
        #   object_vel_xy: 2
        #   prev_move_xy: 2
        #   last_action: 6
        #   normalized_time: 1
        #   rot_error_vec: 3
        #   rot_error_rad: 1
        # ]
        # total = 20
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(20,),
            dtype=np.float32,
        )

        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(6,),
            dtype=np.float32,
        )

        self.max_xy_step = 0.012

        self.base_seek_speed = BASE_SEEK_SPEED
        self.min_seek_speed = MIN_SEEK_SPEED

        self.success_radius = 0.018
        self.failure_radius = 0.75
        self.success_rot_deg = 15.0
        self.failure_rot_deg = 170.0

        self.step_count = 0
        self.current_raw_obs = None
        self.target_array = None

        self.prev_object_pose = None
        self.prev_object_pos = None
        self.prev_dist = None
        self.prev_rot_error_deg = None

        self.prev_move_xy = np.zeros(2, dtype=np.float32)
        self.last_action = np.zeros(6, dtype=np.float32)

    def reset(self, **kwargs):
        raw_obs, info = super().reset(**kwargs)

        self.step_count = 0
        self.current_raw_obs = self._trim_raw_obs(raw_obs)

        object_pose = self._object_pose(self.current_raw_obs)

        # Use generator target XY if useful, but override Z for sphere center sanity.
        self.target_array = np.asarray(info["target"], dtype=np.float32).copy()
        self.target_array[2] = object_pose[2]

        # If generator returns trivial target, synthesize a fixed nontrivial XY target.
        init_dist = self._xy_distance(object_pose, self.target_array)
        if init_dist < 0.05:
            self.target_array = object_pose.copy()
            self.target_array[:2] = np.clip(
                object_pose[:2] + np.array([0.12, -0.08], dtype=np.float32),
                -0.35,
                0.35,
            )
            self.target_array[2] = object_pose[2]
            r_obj = R.from_euler("xyz", object_pose[3:], degrees=True)
            r_delta = R.from_euler("xyz", [20.0, -10.0, 25.0], degrees=True)
            self.target_array[3:] = (r_delta * r_obj).as_euler("xyz", degrees=True).astype(np.float32)

        self.prev_object_pose = object_pose.copy()
        self.prev_object_pos = object_pose[:3].copy()
        self.prev_dist = self._xy_distance(object_pose, self.target_array)
        self.prev_rot_error_deg = self._rotation_error_deg(object_pose, self.target_array)

        self.prev_move_xy = np.zeros(2, dtype=np.float32)
        self.last_action = np.zeros(6, dtype=np.float32)

        out_info = dict(info)
        out_info["target"] = self.target_array.copy()
        out_info["initial_distance"] = self.prev_dist
        out_info["initial_translation_error"] = self.prev_dist
        out_info["initial_rotation_error"] = self.prev_rot_error_deg
        out_info["current_translation_error"] = self.prev_dist
        out_info["current_rotation_error"] = self.prev_rot_error_deg
        out_info["distance_to_target"] = self.prev_dist
        out_info["success"] = False
        out_info["is_success"] = False
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

        raw_obs, base_reward, base_terminated, base_truncated, info = super().step(
            composite_action
        )

        self.current_raw_obs = self._trim_raw_obs(raw_obs)

        # Keep target z synced to current sphere center. XY target remains fixed.
        object_pose = self._object_pose(self.current_raw_obs)
        self.target_array[2] = object_pose[2]

        reward, reward_info = self._compute_reward(self.current_raw_obs, policy_action)

        success = bool(reward_info["success"])
        failure = bool(reward_info["failure"])
        timeout = self.step_count >= self.wrapper_max_episode_steps

        # Important: ignore base_terminated for sphere wrapper.
        # Composite base termination often fires for reasons unrelated to this task.
        terminated = bool(success or failure)
        truncated = bool(timeout and not terminated)

        self._update_reward_state(reward_info)

        obs = self._compact_obs(self.current_raw_obs)

        out_info = dict(info)
        out_info.update(debug_info)
        out_info.update(reward_info)
        out_info["target"] = self.target_array.copy()
        out_info["success"] = success
        out_info["is_success"] = success
        out_info["failure"] = failure
        out_info["base_reward"] = float(base_reward)
        out_info["base_terminated"] = bool(base_terminated)
        out_info["base_truncated"] = bool(base_truncated)
        out_info["TimeLimit.truncated"] = bool(truncated)

        return obs, float(reward), terminated, truncated, out_info

    def _action_to_composite_action(self, policy_action):
        current_pose = self._object_pose(self.current_raw_obs)
        current_pos = current_pose[:3]

        delta_xy = policy_action[:2] * self.max_xy_step
        delta_z = float(policy_action[2]) * 0.0005

        waypoint = current_pose.copy()
        waypoint[0] = current_pos[0] + delta_xy[0]
        waypoint[1] = current_pos[1] + delta_xy[1]

        # Small local Z command only.
        waypoint[2] = np.clip(
            current_pose[2] + delta_z,
            current_pose[2] - 0.002,
            current_pose[2] + 0.002,
        )

        # Keep rotation frozen for this first test.
        delta_rotvec = policy_action[3:] * np.array([0.05, 0.05, 0.05], dtype=np.float32)
        r_current = R.from_euler("xyz", current_pose[3:], degrees=True)
        r_delta = R.from_rotvec(delta_rotvec.astype(np.float64))
        r_new = r_delta * r_current
        waypoint[3:] = r_new.as_euler("xyz", degrees=True).astype(np.float32)

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
            "chosen_dz": float(delta_z),
            "chosen_drotvec_x": float(delta_rotvec[0]),
            "chosen_drotvec_y": float(delta_rotvec[1]),
            "chosen_drotvec_z": float(delta_rotvec[2]),
            "chosen_drotvec_norm": float(np.linalg.norm(delta_rotvec)),
            "executed_roll": float(waypoint[3]),
            "executed_pitch": float(waypoint[4]),
            "executed_yaw": float(waypoint[5]),
            "raw_action": policy_action.copy(),
        }

    def _compute_reward(self, raw_obs, policy_action):
        object_pose = self._object_pose(raw_obs)
        object_pos = object_pose[:3]

        curr_dist = self._xy_distance(object_pose, self.target_array)
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

        movement_toward_goal = float(np.dot(move_xy, goal_dir))

        rot_error_deg = self._rotation_error_deg(object_pose, self.target_array)
        rot_progress_deg = float(self.prev_rot_error_deg - rot_error_deg)

        success = bool(
            curr_dist <= self.success_radius
            and rot_error_deg <= self.success_rot_deg
        )
        failure = bool(curr_dist > self.failure_radius or rot_error_deg > self.failure_rot_deg)

        progress_reward = 80.0 * progress
        toward_goal_reward = 20.0 * movement_toward_goal
        distance_penalty = -0.4 * curr_dist
        action_penalty = -0.01 * float(np.linalg.norm(policy_action[:2]))
        step_penalty = -0.002

        stuck_penalty = 0.0
        if curr_dist > self.success_radius and move_xy_norm < 1e-5:
            stuck_penalty = -0.01

        # Rotation is diagnostic only for sphere. Penalize major worsening very lightly.
        xy_gate = float(np.exp(-curr_dist / 0.06))
        rotation_progress_reward = 0.015 * xy_gate * rot_progress_deg
        rotation_worsening_penalty = -0.004 * xy_gate * max(0.0, -rot_progress_deg)
        rotation_error_penalty = -0.001 * xy_gate * rot_error_deg

        terminal_reward = 10.0 if success else 0.0
        failure_penalty = -10.0 if failure else 0.0

        reward_unclipped = (
            progress_reward
            + toward_goal_reward
            + distance_penalty
            + action_penalty
            + step_penalty
            + stuck_penalty
            + rotation_worsening_penalty
            + terminal_reward
            + failure_penalty
            + rotation_progress_reward
            + rotation_worsening_penalty
        )

        reward = float(np.clip(reward_unclipped, -5.0, 15.0))

        return reward, {
            "success": success,
            "failure": failure,
            "object_out_of_bounds": failure,
            "object_fell": False,

            "current_pose_error": curr_dist,
            "pose_error": curr_dist,
            "current_translation_error": curr_dist,
            "distance_to_target": curr_dist,
            "current_distance": curr_dist,
            "current_rotation_error": rot_error_deg,
            "rotation_error": rot_error_deg,

            "translation_progress": progress,
            "progress_xy": progress,
            "movement_toward_goal": movement_toward_goal,
            "rotation_progress": rot_progress_deg,

            "progress_reward": float(progress_reward),
            "toward_goal_reward": float(toward_goal_reward),
            "distance_penalty": float(distance_penalty),
            "action_penalty": float(action_penalty),
            "step_penalty": float(step_penalty),
            "stuck_penalty": float(stuck_penalty),
            "rotation_worsening_penalty": float(rotation_worsening_penalty),
            "terminal_reward": float(terminal_reward),
            "failure_penalty": float(failure_penalty),
            "reward_unclipped": float(reward_unclipped),

            "move_norm": float(np.linalg.norm(move_vec)),
            "move_xy_norm": move_xy_norm,
            "z_movement": abs(float(move_vec[2])),

            "current_object_pose": object_pose.copy(),
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

        rot_error_vec = self._rotation_error_vec(object_pose, self.target_array)
        rot_error_rad = float(np.linalg.norm(rot_error_vec))

        normalized_time = float(self.step_count) / float(
            max(1, self.wrapper_max_episode_steps)
        )

        return np.concatenate(
            [
                rel_xy.astype(np.float32),
                np.array([dist], dtype=np.float32),
                goal_dir.astype(np.float32),
                object_vel[:2].astype(np.float32),
                self.prev_move_xy.astype(np.float32),
                self.last_action.astype(np.float32),
                np.array([normalized_time], dtype=np.float32),
                rot_error_vec.astype(np.float32),
                np.array([rot_error_rad], dtype=np.float32),
            ]
        ).astype(np.float32)

    def _update_reward_state(self, reward_info):
        new_pose = reward_info["current_object_pose"].copy()
        new_pos = new_pose[:3].copy()

        self.prev_move_xy = (new_pos[:2] - self.prev_object_pos[:2]).astype(np.float32)
        self.prev_object_pose = new_pose
        self.prev_object_pos = new_pos
        self.prev_dist = float(reward_info["current_translation_error"])
        self.prev_rot_error_deg = float(reward_info["current_rotation_error"])

    def _object_pose(self, raw_obs):
        obs_obj = self._parse_obs(raw_obs)
        return obs_obj.object_pose.array().astype(np.float32).copy()

    def _trim_raw_obs(self, obs):
        raw = np.asarray(obs, dtype=np.float32).reshape(-1)
        return raw[: self.raw_obs_dim].copy()

    def _parse_obs(self, obs):
        raw = self._trim_raw_obs(obs)
        return PinArrayEnvObservation.from_array(raw, self.config.pins_per_side)

    @staticmethod
    def _xy_distance(object_pose, target_pose):
        a = np.asarray(object_pose, dtype=np.float32)
        b = np.asarray(target_pose, dtype=np.float32)
        return float(np.linalg.norm(a[:2] - b[:2]))

    @staticmethod
    def _rotation_error_vec(object_pose, target_pose):
        object_pose = np.asarray(object_pose, dtype=np.float32)
        target_pose = np.asarray(target_pose, dtype=np.float32)

        r_obj = R.from_euler("xyz", object_pose[3:], degrees=True)
        r_tgt = R.from_euler("xyz", target_pose[3:], degrees=True)
        r_err = r_tgt * r_obj.inv()

        return r_err.as_rotvec().astype(np.float32)

    @classmethod
    def _rotation_error_deg(cls, object_pose, target_pose):
        rotvec = cls._rotation_error_vec(object_pose, target_pose)
        return float(np.degrees(np.linalg.norm(rotvec)))


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

    env = ReliableSphere6DOFEnv(
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
    env = SubprocVecEnv([make_one_env(i) for i in range(N_ENVS)])

    print("run dir:", PATHS.run_dir)
    print("obs space:", env.observation_space)
    print("action space:", env.action_space)

    policy_kwargs = dict(
        net_arch=dict(pi=[256, 256], vf=[256, 256]),
        log_std_init=-0.5,
    )

    model = PPO(
        "MlpPolicy",
        env,
        policy_kwargs=policy_kwargs,
        verbose=1,
        tensorboard_log=PATHS.tensorboard_log,
        learning_rate=1e-4,
        n_steps=4096 // N_ENVS,
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
        model.learn(total_timesteps=TOTAL_TIMESTEPS)
        save_model(model, PATHS, "ppo_6dof_twist_v3")
    finally:
        env.close()


if __name__ == "__main__":
    main()