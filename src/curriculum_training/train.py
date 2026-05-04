# =============================================================================
# SETUP
# =============================================================================

import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# =============================================================================
# IMPORTS
# =============================================================================

import numpy as np
from gymnasium import spaces
from scipy.spatial.transform import Rotation as R

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv

from rl_common import setup, save_model

from constants import *

from pin_array_manipulator_object_control.environment.composite_control_env import (
    CompositeControlEnv,
)
from pin_array_manipulator_object_control.manipulator.observation import (
    PinArrayEnvObservation,
)
from pin_array_manipulator_object_control.manipulator.pin_array_manipulator import (
    PinArrayManipulatorConfig,
)
from pin_array_manipulator_object_control.objects.ball import Ball
from pin_array_manipulator_object_control.rewards.distance_3d import Distance3DRewardModel
from pin_array_manipulator_object_control.routines.multi_target_generator import (
    MultiTargetGenerator,
)


PATHS = setup(__file__, RUN_NAME, RUN_NAME)


# =============================================================================
# ENVIRONMENT
# =============================================================================

class CleanSphereCurriculumEnv(CompositeControlEnv):
    """
    Clean sphere curriculum with a real true_6dof mode.

    Policy action:
        [dx, dy, dz, drotvec_x, drotvec_y, drotvec_z]

    Modes:
        "xy":
            XY transport only.
            Z target synced.
            Z action frozen.
            Rotation frozen.
            Success is XY-only.

        "xy_z":
            XY transport + small Z action.
            Z target synced unless mode is true_6dof.
            Success is XY-only.

        "xy_rot":
            XY transport + auxiliary rotation.
            Z action frozen.
            Success is XY-only.

        "pose_6d":
            Legacy XY + SO(3) orientation mode.
            Z target still synced.
            Success is XY + rotation.

        "true_6dof":
            XYZ + SO(3) target tracking.
            Target Z is fixed.
            Success requires X/Y/Z/XYZ and rotation thresholds.
            Observation is 24D and includes rel_xyz, xyz distance, goal_dir_xyz,
            object_vel_xyz, prev_move_xyz, last_action, time, rotvec, rot_error_rad.

    Important:
        For a sphere, true 6DOF orientation is physically questionable unless the
        simulator/object has meaningful orientation markings. For pencil/rod objects,
        this formulation becomes more meaningful.
    """

    def __init__(
        self,
        *args,
        manipulator_config: PinArrayManipulatorConfig,
        curriculum_mode: str = CURRICULUM_MODE,
        max_episode_steps: int = MAX_EPISODE_STEPS,
        action_repeat: int = ACTION_REPEAT,
        **kwargs,
    ):
        self.wrapper_max_episode_steps = int(max_episode_steps)
        self.curriculum_mode = str(curriculum_mode)
        self.action_repeat = int(action_repeat)

        super().__init__(*args, manipulator_config=manipulator_config, **kwargs)

        self.config = manipulator_config
        self.raw_obs_dim = int(np.prod(self.observation_space.shape))

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=OBSERVATION_SHAPE,
            dtype=np.float32,
        )

        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=ACTION_SHAPE,
            dtype=np.float32,
        )

        self.base_seek_speed = BASE_SEEK_SPEED
        self.min_seek_speed = MIN_SEEK_SPEED

        self.max_xy_step = float(MAX_XY_STEP)
        self.max_z_step = float(MAX_Z_STEP)

        if self.curriculum_mode == "xy":
            self.max_rotvec_step = MAX_ROTVEC_STEP_XY.copy()
        elif self.curriculum_mode in ("xy_z", "xy_rot"):
            self.max_rotvec_step = MAX_ROTVEC_STEP_LIGHT.copy()
        elif self.curriculum_mode == "pose_6d":
            self.max_rotvec_step = MAX_ROTVEC_STEP_FULL.copy()
        elif self.curriculum_mode == "true_6dof":
            self.max_rotvec_step = MAX_ROTVEC_STEP_TRUE_6DOF.copy()
        else:
            raise ValueError(f"Unknown curriculum_mode: {self.curriculum_mode}")

        self.success_xy_radius = float(SUCCESS_XY_RADIUS)
        self.success_rot_deg = float(SUCCESS_ROT_DEG)
        self.failure_radius = float(FAILURE_RADIUS)

        self.step_count = 0
        self.current_raw_obs = None

        self.target_array = None
        self.base_generator_target = None
        self.synthetic_target_used = False

        self.prev_object_pose = None
        self.prev_object_pos = None

        self.prev_xy_dist = None
        self.prev_xyz_dist = None
        self.prev_z_err = None

        self.prev_rot_error_rad = None
        self.prev_rot_error_deg = None

        self.prev_move_xy = np.zeros(2, dtype=np.float32)
        self.prev_move_xyz = np.zeros(3, dtype=np.float32)

        self.last_action = np.zeros(ACTION_DIM, dtype=np.float32)

    # =========================================================================
    # GYM API
    # =========================================================================

    def reset(self, **kwargs):
        raw_obs, info = super().reset(**kwargs)

        self.step_count = 0
        self.current_raw_obs = self._trim_raw_obs(raw_obs)

        object_pose = self._object_pose(self.current_raw_obs)
        object_pos = object_pose[:3].copy()

        self.base_generator_target = np.asarray(
            info.get("target", object_pose),
            dtype=np.float32,
        ).copy()

        self.target_array = self._make_initial_target(object_pose, info)

        if self.curriculum_mode != "true_6dof":
            self.target_array[2] = object_pose[2]

        errors = self._pose_errors(object_pose, self.target_array)

        self.prev_object_pose = object_pose.copy()
        self.prev_object_pos = object_pos.copy()

        self.prev_xy_dist = errors["xy_err"]
        self.prev_xyz_dist = errors["xyz_err"]
        self.prev_z_err = errors["z_err"]

        self.prev_rot_error_rad = errors["rot_err_rad"]
        self.prev_rot_error_deg = errors["rot_err_deg"]

        self.prev_move_xy[:] = 0.0
        self.prev_move_xyz[:] = 0.0
        self.last_action[:] = 0.0

        bounds_info = self._bounds_info(object_pose)

        out_info = dict(info)
        out_info.update(
            {
                "target": self.target_array.copy(),
                "base_generator_target": self.base_generator_target.copy(),
                "synthetic_target_used": bool(self.synthetic_target_used),
                "curriculum_mode": self.curriculum_mode,
                "fixed_episode_target": True,
                "z_target_synced": bool(self.curriculum_mode != "true_6dof"),

                "initial_x_error": errors["x_err"],
                "initial_y_error": errors["y_err"],
                "initial_z_error": errors["z_err"],
                "initial_z_error_signed": errors["z_err_signed"],

                "initial_xy_error": errors["xy_err"],
                "initial_xyz_error": errors["xyz_err"],
                "initial_distance": errors["xy_err"]
                if self.curriculum_mode != "true_6dof"
                else errors["xyz_err"],
                "initial_translation_error": errors["xy_err"]
                if self.curriculum_mode != "true_6dof"
                else errors["xyz_err"],

                "initial_rotation_error": errors["rot_err_deg"],
                "initial_rotation_error_deg": errors["rot_err_deg"],
                "initial_rotation_error_rad": errors["rot_err_rad"],

                "current_x_error": errors["x_err"],
                "current_y_error": errors["y_err"],
                "current_z_error": errors["z_err"],
                "current_z_error_signed": errors["z_err_signed"],
                "current_xy_error": errors["xy_err"],
                "current_xyz_error": errors["xyz_err"],

                "current_translation_error": errors["xy_err"]
                if self.curriculum_mode != "true_6dof"
                else errors["xyz_err"],

                "current_rotation_error": errors["rot_err_deg"],
                "current_rotation_error_deg": errors["rot_err_deg"],
                "current_rotation_error_rad": errors["rot_err_rad"],

                "distance_to_target": errors["xy_err"]
                if self.curriculum_mode != "true_6dof"
                else errors["xyz_err"],

                "current_object_pose": object_pose.copy(),
                "current_object_pos": object_pos.copy(),

                "success": False,
                "is_success": False,
                "failure": False,

                **bounds_info,
            }
        )

        return self._compact_obs(self.current_raw_obs), out_info

    def step(self, action):
        self.step_count += 1

        pre_pose = self._object_pose(self.current_raw_obs)
        prev_action_for_reward = self.last_action.copy()

        policy_action = np.clip(
            np.asarray(action, dtype=np.float32).reshape(-1),
            -1.0,
            1.0,
        )

        if policy_action.shape != ACTION_SHAPE:
            raise RuntimeError(
                f"Bad action shape: got {policy_action.shape}, expected {ACTION_SHAPE}"
            )

        self.last_action = policy_action.copy()

        composite_action, debug_info = self._action_to_composite_action(policy_action)

        if self.render_mode == "human":
            try:
                self.update_debug_visuals(composite_action[2:])
            except Exception:
                pass

        total_base_reward = 0.0
        base_terminated = False
        base_truncated = False
        info = {}

        raw_obs = self.current_raw_obs

        for _ in range(self.action_repeat):
            raw_obs, base_reward, base_terminated, base_truncated, info = super().step(
                composite_action
            )
            total_base_reward += float(base_reward)

            if base_terminated or base_truncated:
                break

        self.current_raw_obs = self._trim_raw_obs(raw_obs)

        post_pose = self._object_pose(self.current_raw_obs)

        if self.curriculum_mode != "true_6dof":
            self.target_array[2] = post_pose[2]

        reward, reward_info = self._compute_reward(
            raw_obs=self.current_raw_obs,
            policy_action=policy_action,
            prev_policy_action=prev_action_for_reward,
        )

        success = bool(reward_info["success"])
        failure = bool(reward_info["failure"])
        timeout = self.step_count >= self.wrapper_max_episode_steps

        terminated = bool(success or failure)
        truncated = bool(timeout and not terminated)

        causality_info = self._action_causality_info(
            pre_pose=pre_pose,
            post_pose=post_pose,
            commanded_waypoint=debug_info["executed_waypoint"],
        )

        self._update_reward_state(reward_info)

        obs = self._compact_obs(self.current_raw_obs)

        out_info = dict(info)
        out_info.update(debug_info)
        out_info.update(reward_info)
        out_info.update(causality_info)
        out_info.update(
            {
                "target": self.target_array.copy(),
                "base_generator_target": self.base_generator_target.copy()
                if self.base_generator_target is not None
                else None,
                "synthetic_target_used": bool(self.synthetic_target_used),
                "curriculum_mode": self.curriculum_mode,
                "fixed_episode_target": True,
                "z_target_synced": bool(self.curriculum_mode != "true_6dof"),

                "success": success,
                "is_success": success,
                "failure": failure,

                "base_reward": float(total_base_reward),
                "base_terminated": bool(base_terminated),
                "base_truncated": bool(base_truncated),
                "TimeLimit.truncated": bool(truncated),
            }
        )

        return obs, float(reward), terminated, truncated, out_info

    # =========================================================================
    # TARGET GENERATION
    # =========================================================================

    def _make_initial_target(self, object_pose, info):
        object_pose = np.asarray(object_pose, dtype=np.float32)

        generator_target = np.asarray(
            info.get("target", object_pose),
            dtype=np.float32,
        ).copy()

        if self.curriculum_mode == "true_6dof":
            generator_target[2] = object_pose[2] + float(SYNTHETIC_TARGET_DELTA_Z)
        else:
            generator_target[2] = object_pose[2]

        if self.curriculum_mode == "true_6dof":
            init_dist = float(np.linalg.norm(object_pose[:3] - generator_target[:3]))
        else:
            init_dist = self._xy_distance(object_pose, generator_target)

        if init_dist >= TRIVIAL_TARGET_XY_RADIUS:
            self.synthetic_target_used = False
            return generator_target.astype(np.float32)

        self.synthetic_target_used = True

        target = object_pose.copy()

        target[:2] = np.clip(
            object_pose[:2] + SYNTHETIC_TARGET_DELTA_XY,
            -SYNTHETIC_TARGET_XY_LIMIT,
            SYNTHETIC_TARGET_XY_LIMIT,
        )

        if self.curriculum_mode == "true_6dof":
            target[2] = object_pose[2] + float(SYNTHETIC_TARGET_DELTA_Z)
        else:
            target[2] = object_pose[2]

        r_obj = R.from_euler("xyz", object_pose[3:], degrees=True)
        r_delta = R.from_euler("xyz", SYNTHETIC_TARGET_ROT_DELTA_DEG, degrees=True)
        target[3:] = (r_delta * r_obj).as_euler("xyz", degrees=True).astype(np.float32)

        return target.astype(np.float32)

    # =========================================================================
    # ACTION -> COMPOSITE ACTION
    # =========================================================================

    def _action_to_composite_action(self, policy_action):
        current_pose = self._object_pose(self.current_raw_obs)
        current_pos = current_pose[:3]

        delta_xy = policy_action[:2] * self.max_xy_step
        raw_delta_z = float(policy_action[2]) * self.max_z_step

        if self.curriculum_mode in ("xy", "xy_rot"):
            delta_z = 0.0
        else:
            delta_z = raw_delta_z

        delta_rotvec = policy_action[3:] * self.max_rotvec_step

        waypoint = current_pose.copy()

        waypoint[0] = current_pos[0] + delta_xy[0]
        waypoint[1] = current_pos[1] + delta_xy[1]

        waypoint[2] = np.clip(
            current_pos[2] + delta_z,
            current_pos[2] - self.max_z_step,
            current_pos[2] + self.max_z_step,
        )

        if np.linalg.norm(delta_rotvec) > 0.0:
            r_current = R.from_euler("xyz", current_pose[3:], degrees=True)
            r_delta = R.from_rotvec(delta_rotvec.astype(np.float64))
            r_new = r_delta * r_current
            waypoint[3:] = r_new.as_euler("xyz", degrees=True).astype(np.float32)
        else:
            waypoint[3:] = current_pose[3:]

        composite_action = np.concatenate(
            [
                np.array(
                    [self.base_seek_speed, self.min_seek_speed],
                    dtype=np.float32,
                ),
                waypoint.astype(np.float32),
            ]
        ).astype(np.float32)

        return composite_action, {
            "executed_waypoint": waypoint.copy(),
            "composite_action": composite_action.copy(),

            "chosen_dx": float(delta_xy[0]),
            "chosen_dy": float(delta_xy[1]),
            "chosen_dz": float(delta_z),
            "raw_chosen_dz": float(raw_delta_z),

            "chosen_drotvec_x": float(delta_rotvec[0]),
            "chosen_drotvec_y": float(delta_rotvec[1]),
            "chosen_drotvec_z": float(delta_rotvec[2]),
            "chosen_drotvec_norm": float(np.linalg.norm(delta_rotvec)),

            "executed_roll": float(waypoint[3]),
            "executed_pitch": float(waypoint[4]),
            "executed_yaw": float(waypoint[5]),

            "raw_action": policy_action.copy(),
        }

    # =========================================================================
    # REWARD HELPERS
    # =========================================================================

    @staticmethod
    def _sigmoid(x):
        x = float(np.clip(x, -60.0, 60.0))
        return float(1.0 / (1.0 + np.exp(-x)))

    @staticmethod
    def _softplus(x):
        x = float(np.clip(x, -60.0, 60.0))
        return float(np.log1p(np.exp(x)))

    @staticmethod
    def _rotation_potential(rot_error_rad):
        return float(1.0 - np.cos(float(rot_error_rad)))

    def _orientation_weight(self, xy_error):
        return self._sigmoid(
            (ROTATION_ON_RADIUS - float(xy_error)) / ROTATION_ON_TEMPERATURE
        )

    def _pose_potential_xy_rot(self, xy_error, rot_error_rad):
        vp = 0.5 * float(xy_error) ** 2
        alpha = self._orientation_weight(xy_error)
        vr = self._rotation_potential(rot_error_rad)
        return float(vp + alpha * ROTATION_POTENTIAL_WEIGHT * vr)

    def _pose_errors(self, object_pose, target_pose):
        object_pose = np.asarray(object_pose, dtype=np.float32)
        target_pose = np.asarray(target_pose, dtype=np.float32)

        pos_err_vec = object_pose[:3] - target_pose[:3]

        x_err = float(pos_err_vec[0])
        y_err = float(pos_err_vec[1])
        z_err_signed = float(pos_err_vec[2])
        z_err = abs(z_err_signed)

        xy_err = float(np.linalg.norm(pos_err_vec[:2]))
        xyz_err = float(np.linalg.norm(pos_err_vec))

        rot_err_vec = self._rotation_error_vec(object_pose, target_pose)
        rot_err_rad = float(np.linalg.norm(rot_err_vec))
        rot_err_deg = float(np.degrees(rot_err_rad))

        return {
            "pos_err_vec": pos_err_vec.astype(np.float32),
            "x_err": x_err,
            "y_err": y_err,
            "z_err": z_err,
            "z_err_signed": z_err_signed,
            "xy_err": xy_err,
            "xyz_err": xyz_err,
            "rot_err_vec": rot_err_vec.astype(np.float32),
            "rot_err_rad": rot_err_rad,
            "rot_err_deg": rot_err_deg,
        }

    def _true_6dof_potential_from_errors(self, errors):
        pos_err_vec = errors["pos_err_vec"]
        rot_err_rad = errors["rot_err_rad"]

        pos_energy = 0.5 * float(
            POSE_W_X * pos_err_vec[0] ** 2
            + POSE_W_Y * pos_err_vec[1] ** 2
            + POSE_W_Z * pos_err_vec[2] ** 2
        )

        rot_energy = self._rotation_potential(rot_err_rad)

        return float(pos_energy + POSE_W_ROT * rot_energy)

    def _safety_barrier(self, object_pos):
        x, y, z = map(float, object_pos[:3])

        clearances = [
            WORKSPACE_XY_LIMIT - abs(x),
            WORKSPACE_XY_LIMIT - abs(y),
            z - MIN_OBJECT_Z,
            MAX_OBJECT_Z - z,
        ]

        barrier = 0.0
        for h in clearances:
            barrier += self._softplus(
                BARRIER_SHARPNESS * (BARRIER_MARGIN - h)
            ) / BARRIER_SHARPNESS

        return float(barrier)

    # =========================================================================
    # REWARD
    # =========================================================================

    def _compute_reward(self, raw_obs, policy_action, prev_policy_action):
        object_pose = self._object_pose(raw_obs)
        object_pos = object_pose[:3].copy()

        bounds_info = self._bounds_info(object_pose)

        curr_errors = self._pose_errors(object_pose, self.target_array)

        curr_xy_dist = curr_errors["xy_err"]
        curr_xyz_dist = curr_errors["xyz_err"]
        curr_z_err = curr_errors["z_err"]
        curr_z_err_signed = curr_errors["z_err_signed"]

        curr_rotvec = curr_errors["rot_err_vec"]
        curr_rot_error_rad = curr_errors["rot_err_rad"]
        curr_rot_error_deg = curr_errors["rot_err_deg"]

        prev_xy_dist = float(self.prev_xy_dist)
        prev_xyz_dist = float(self.prev_xyz_dist)
        prev_z_err = float(self.prev_z_err)
        prev_rot_error_rad = float(self.prev_rot_error_rad)
        prev_rot_error_deg = float(np.degrees(prev_rot_error_rad))

        move_vec = object_pos - self.prev_object_pos
        move_xy = move_vec[:2]
        move_norm = float(np.linalg.norm(move_vec))
        move_xy_norm = float(np.linalg.norm(move_xy))

        goal_vec_xy = self.target_array[:2] - object_pos[:2]
        goal_dist_xy = float(np.linalg.norm(goal_vec_xy))

        if goal_dist_xy > 1e-8:
            goal_dir_xy = goal_vec_xy / goal_dist_xy
        else:
            goal_dir_xy = np.zeros(2, dtype=np.float32)

        movement_toward_goal = float(np.dot(move_xy, goal_dir_xy))

        action_xy_cost = -ACTION_XY_COST_COEFF * float(np.sum(policy_action[:2] ** 2))
        action_z_cost = -ACTION_Z_COST_COEFF * float(policy_action[2] ** 2)
        action_rot_cost = -ACTION_ROT_COST_COEFF * float(np.sum(policy_action[3:] ** 2))

        action_delta = policy_action - prev_policy_action
        action_jerk_cost = -ACTION_JERK_COST_COEFF * float(np.sum(action_delta ** 2))

        barrier = self._safety_barrier(object_pos)
        barrier_penalty = -BARRIER_COST_COEFF * barrier

        step_penalty = -STEP_COST

        # ---------------------------------------------------------------------
        # TRUE 6DOF
        # ---------------------------------------------------------------------

        if self.curriculum_mode == "true_6dof":
            prev_errors = self._pose_errors(self.prev_object_pose, self.target_array)

            v_prev = self._true_6dof_potential_from_errors(prev_errors)
            v_curr = self._true_6dof_potential_from_errors(curr_errors)

            task_potential_delta = float(v_prev - v_curr)
            task_potential_reward = TRUE_6DOF_POTENTIAL_COEFF * task_potential_delta

            xyz_progress = float(prev_xyz_dist - curr_xyz_dist)
            xy_progress = float(prev_xy_dist - curr_xy_dist)
            z_progress = float(prev_z_err - curr_z_err)

            rot_error_delta_rad = float(prev_rot_error_rad - curr_rot_error_rad)
            rot_error_delta_deg = float(np.degrees(rot_error_delta_rad))

            xyz_distance_penalty = -XYZ_DISTANCE_COST_COEFF * curr_xyz_dist
            z_error_penalty = -Z_ERROR_COST_COEFF * curr_z_err

            # This does not truly "balance" axes; it simply penalizes remaining
            # local rotvec error. Useful as a diagnostic and mild stabilizer.
            rot_axis_balance_penalty = -ROT_AXIS_BALANCE_COST_COEFF * float(
                np.sum(curr_rotvec ** 2)
            )

            directional_reward = DIRECTIONAL_REWARD_COEFF * movement_toward_goal

            goal_kernel_reward = GOAL_KERNEL_COEFF * float(
                np.exp(-0.5 * (curr_xyz_dist / GOAL_KERNEL_SIGMA) ** 2)
            )

            x_success = abs(curr_errors["x_err"]) <= SUCCESS_X_RADIUS
            y_success = abs(curr_errors["y_err"]) <= SUCCESS_Y_RADIUS
            z_success = curr_z_err <= SUCCESS_Z_RADIUS
            xyz_success = curr_xyz_dist <= SUCCESS_XYZ_RADIUS
            rot_success = curr_rot_error_rad <= SUCCESS_ROT_RAD

            success = bool(
                x_success
                and y_success
                and z_success
                and xyz_success
                and rot_success
            )

            failure = bool(
                curr_xyz_dist > self.failure_radius
                or bounds_info["object_out_of_bounds"]
            )

            stuck_penalty = 0.0
            if curr_xyz_dist > SUCCESS_XYZ_RADIUS and move_norm < STUCK_MOVE_EPS:
                stuck_penalty = -STUCK_COST

            terminal_reward = SUCCESS_BONUS if success else 0.0
            failure_penalty = -FAILURE_PENALTY if failure else 0.0

            reward_unclipped = (
                task_potential_reward
                + xyz_distance_penalty
                + z_error_penalty
                + rot_axis_balance_penalty
                + directional_reward
                + goal_kernel_reward
                + barrier_penalty
                + action_xy_cost
                + action_z_cost
                + action_rot_cost
                + action_jerk_cost
                + step_penalty
                + stuck_penalty
                + terminal_reward
                + failure_penalty
            )

            reward = float(np.clip(reward_unclipped, REWARD_CLIP_LOW, REWARD_CLIP_HIGH))

            return reward, {
                "success": success,
                "failure": failure,

                "x_success": bool(x_success),
                "y_success": bool(y_success),
                "z_success": bool(z_success),
                "xy_success": bool(curr_xy_dist <= self.success_xy_radius),
                "xyz_success": bool(xyz_success),
                "rot_success": bool(rot_success),

                **bounds_info,

                "current_pose_error": curr_xyz_dist,
                "pose_error": curr_xyz_dist,
                "current_translation_error": curr_xyz_dist,
                "current_xyz_error": curr_xyz_dist,
                "current_xy_error": curr_xy_dist,
                "current_z_error": curr_z_err,
                "current_z_error_signed": curr_z_err_signed,
                "distance_to_target": curr_xyz_dist,
                "current_distance": curr_xyz_dist,

                "current_x_error": float(curr_errors["x_err"]),
                "current_y_error": float(curr_errors["y_err"]),

                "current_rotation_error": curr_rot_error_deg,
                "current_rotation_error_deg": curr_rot_error_deg,
                "current_rotation_error_rad": curr_rot_error_rad,
                "rotation_error": curr_rot_error_deg,
                "rotation_error_rad": curr_rot_error_rad,

                "rotation_error_vec": curr_rotvec.copy(),
                "rotation_error_vec_x": float(curr_rotvec[0]),
                "rotation_error_vec_y": float(curr_rotvec[1]),
                "rotation_error_vec_z": float(curr_rotvec[2]),

                "prev_xy_error": prev_xy_dist,
                "prev_xyz_error": prev_xyz_dist,
                "prev_z_error": prev_z_err,
                "prev_rotation_error": prev_rot_error_deg,
                "prev_rotation_error_rad": prev_rot_error_rad,

                "v_prev": float(v_prev),
                "v_curr": float(v_curr),
                "task_potential_delta": float(task_potential_delta),
                "task_potential_reward": float(task_potential_reward),

                "translation_progress": float(xyz_progress),
                "progress_xyz": float(xyz_progress),
                "progress_xy": float(xy_progress),
                "z_progress": float(z_progress),

                "rotation_progress": float(rot_error_delta_deg),
                "rotation_progress_deg": float(rot_error_delta_deg),
                "rotation_progress_rad": float(rot_error_delta_rad),
                "movement_toward_goal": movement_toward_goal,

                "xyz_distance_penalty": float(xyz_distance_penalty),
                "z_error_penalty": float(z_error_penalty),
                "rot_axis_balance_penalty": float(rot_axis_balance_penalty),
                "directional_reward": float(directional_reward),
                "goal_kernel_reward": float(goal_kernel_reward),

                "safety_barrier": float(barrier),
                "barrier_penalty": float(barrier_penalty),

                "action_xy_cost": float(action_xy_cost),
                "action_z_cost": float(action_z_cost),
                "action_rot_cost": float(action_rot_cost),
                "action_jerk_cost": float(action_jerk_cost),
                "action_penalty": float(
                    action_xy_cost
                    + action_z_cost
                    + action_rot_cost
                    + action_jerk_cost
                ),

                "step_penalty": float(step_penalty),
                "stuck_penalty": float(stuck_penalty),
                "terminal_reward": float(terminal_reward),
                "failure_penalty": float(failure_penalty),

                "reward_unclipped": float(reward_unclipped),
                "reward_clipped": float(reward),

                "move_norm": move_norm,
                "move_xy_norm": move_xy_norm,
                "current_z": float(object_pos[2]),
                "z_movement": abs(float(move_vec[2])),

                "current_object_pose": object_pose.copy(),
                "current_object_pos": object_pos.copy(),
            }

        # ---------------------------------------------------------------------
        # LEGACY / CURRICULUM BRANCH: xy, xy_z, xy_rot, pose_6d
        # ---------------------------------------------------------------------

        vp_prev = 0.5 * prev_xy_dist ** 2
        vp_curr = 0.5 * curr_xy_dist ** 2

        vr_prev = self._rotation_potential(prev_rot_error_rad)
        vr_curr = self._rotation_potential(curr_rot_error_rad)

        v_prev = self._pose_potential_xy_rot(prev_xy_dist, prev_rot_error_rad)
        v_curr = self._pose_potential_xy_rot(curr_xy_dist, curr_rot_error_rad)

        task_potential_delta = float(v_prev - v_curr)
        task_potential_reward = TASK_POTENTIAL_COEFF * task_potential_delta

        xy_progress = float(prev_xy_dist - curr_xy_dist)

        rot_error_delta_rad = float(prev_rot_error_rad - curr_rot_error_rad)
        rot_error_delta_deg = float(np.degrees(rot_error_delta_rad))

        rot_potential_progress = float(vr_prev - vr_curr)
        positive_rot_progress = max(0.0, rot_potential_progress)
        translation_potential_worsening = max(0.0, vp_curr - vp_prev)

        progress_gate = self._sigmoid(
            (xy_progress - XY_PROGRESS_EPS) / XY_PROGRESS_TEMPERATURE
        )

        orientation_weight = self._orientation_weight(curr_xy_dist)

        aux_rotation_reward = (
            AUX_ROT_PROGRESS_COEFF
            * progress_gate
            * orientation_weight
            * positive_rot_progress
        )

        rotation_translation_conflict_penalty = (
            -ROT_TRANSLATION_CONFLICT_COEFF
            * positive_rot_progress
            * translation_potential_worsening
        )

        directional_reward = DIRECTIONAL_REWARD_COEFF * movement_toward_goal

        goal_kernel_reward = GOAL_KERNEL_COEFF * float(
            np.exp(-0.5 * (curr_xy_dist / GOAL_KERNEL_SIGMA) ** 2)
        )

        xy_success = curr_xy_dist <= self.success_xy_radius
        rot_success = curr_rot_error_deg <= self.success_rot_deg

        if self.curriculum_mode == "pose_6d":
            success = bool(xy_success and rot_success)
        else:
            success = bool(xy_success)

        failure = bool(
            curr_xy_dist > self.failure_radius
            or bounds_info["object_out_of_bounds"]
        )

        stuck_penalty = 0.0
        if curr_xy_dist > self.success_xy_radius and move_xy_norm < STUCK_MOVE_EPS:
            stuck_penalty = -STUCK_COST

        terminal_reward = SUCCESS_BONUS if success else 0.0
        failure_penalty = -FAILURE_PENALTY if failure else 0.0

        reward_unclipped = (
            task_potential_reward
            + aux_rotation_reward
            + rotation_translation_conflict_penalty
            + directional_reward
            + goal_kernel_reward
            + barrier_penalty
            + action_xy_cost
            + action_z_cost
            + action_rot_cost
            + action_jerk_cost
            + step_penalty
            + stuck_penalty
            + terminal_reward
            + failure_penalty
        )

        reward = float(np.clip(reward_unclipped, REWARD_CLIP_LOW, REWARD_CLIP_HIGH))

        return reward, {
            "success": success,
            "failure": failure,

            "xy_success": bool(xy_success),
            "rot_success": bool(rot_success),

            **bounds_info,

            "current_pose_error": curr_xy_dist,
            "pose_error": curr_xy_dist,
            "current_translation_error": curr_xy_dist,
            "current_xy_error": curr_xy_dist,
            "current_xyz_error": curr_xyz_dist,
            "current_z_error": curr_z_err,
            "current_z_error_signed": curr_z_err_signed,
            "distance_to_target": curr_xy_dist,
            "current_distance": curr_xy_dist,

            "current_x_error": float(curr_errors["x_err"]),
            "current_y_error": float(curr_errors["y_err"]),

            "current_rotation_error": curr_rot_error_deg,
            "current_rotation_error_deg": curr_rot_error_deg,
            "current_rotation_error_rad": curr_rot_error_rad,
            "rotation_error": curr_rot_error_deg,
            "rotation_error_rad": curr_rot_error_rad,

            "rotation_error_vec": curr_rotvec.copy(),
            "rotation_error_vec_x": float(curr_rotvec[0]),
            "rotation_error_vec_y": float(curr_rotvec[1]),
            "rotation_error_vec_z": float(curr_rotvec[2]),

            "prev_xy_error": prev_xy_dist,
            "prev_xyz_error": prev_xyz_dist,
            "prev_z_error": prev_z_err,
            "prev_rotation_error": prev_rot_error_deg,
            "prev_rotation_error_rad": prev_rot_error_rad,

            "vp_prev": float(vp_prev),
            "vp_curr": float(vp_curr),
            "vr_prev": float(vr_prev),
            "vr_curr": float(vr_curr),
            "v_prev": float(v_prev),
            "v_curr": float(v_curr),
            "task_potential_delta": float(task_potential_delta),

            "translation_progress": xy_progress,
            "progress_xy": xy_progress,
            "progress_xyz": float(prev_xyz_dist - curr_xyz_dist),
            "z_progress": float(prev_z_err - curr_z_err),

            "rotation_progress": rot_error_delta_deg,
            "rotation_progress_deg": rot_error_delta_deg,
            "rotation_progress_rad": rot_error_delta_rad,
            "rotation_potential_progress": rot_potential_progress,
            "positive_rotation_potential_progress": positive_rot_progress,
            "translation_potential_worsening": translation_potential_worsening,
            "movement_toward_goal": movement_toward_goal,

            "progress_gate": float(progress_gate),
            "orientation_weight": float(orientation_weight),

            "task_potential_reward": float(task_potential_reward),
            "aux_rotation_reward": float(aux_rotation_reward),
            "rotation_translation_conflict_penalty": float(
                rotation_translation_conflict_penalty
            ),
            "directional_reward": float(directional_reward),
            "goal_kernel_reward": float(goal_kernel_reward),

            "safety_barrier": float(barrier),
            "barrier_penalty": float(barrier_penalty),

            "action_xy_cost": float(action_xy_cost),
            "action_z_cost": float(action_z_cost),
            "action_rot_cost": float(action_rot_cost),
            "action_jerk_cost": float(action_jerk_cost),
            "action_penalty": float(
                action_xy_cost
                + action_z_cost
                + action_rot_cost
                + action_jerk_cost
            ),

            "step_penalty": float(step_penalty),
            "stuck_penalty": float(stuck_penalty),
            "terminal_reward": float(terminal_reward),
            "failure_penalty": float(failure_penalty),

            "reward_unclipped": float(reward_unclipped),
            "reward_clipped": float(reward),

            "move_norm": move_norm,
            "move_xy_norm": move_xy_norm,
            "current_z": float(object_pos[2]),
            "z_movement": abs(float(move_vec[2])),

            "current_object_pose": object_pose.copy(),
            "current_object_pos": object_pos.copy(),
        }

    # =========================================================================
    # OBSERVATION
    # =========================================================================

    def _compact_obs(self, raw_obs):
        obs_obj = self._parse_obs(raw_obs)

        object_pose = obs_obj.object_pose.array().astype(np.float32)
        object_pos = object_pose[:3]
        object_vel = obs_obj.object_velocity.array().astype(np.float32)

        rel_pos = self.target_array[:3].astype(np.float32) - object_pos[:3]

        if self.curriculum_mode != "true_6dof":
            rel_pos[2] = 0.0

        pos_dist = float(np.linalg.norm(rel_pos))

        if pos_dist > 1e-8:
            goal_dir = rel_pos / pos_dist
        else:
            goal_dir = np.zeros(3, dtype=np.float32)

        rot_error_vec = self._rotation_error_vec(object_pose, self.target_array)
        rot_error_rad = float(np.linalg.norm(rot_error_vec))

        normalized_time = float(self.step_count) / float(
            max(1, self.wrapper_max_episode_steps)
        )

        obs = np.concatenate(
            [
                rel_pos.astype(np.float32),                         # 3
                np.array([pos_dist], dtype=np.float32),              # 1
                goal_dir.astype(np.float32),                         # 3
                object_vel[:3].astype(np.float32),                   # 3
                self.prev_move_xyz.astype(np.float32),               # 3
                self.last_action.astype(np.float32),                 # 6
                np.array([normalized_time], dtype=np.float32),       # 1
                rot_error_vec.astype(np.float32),                    # 3
                np.array([rot_error_rad], dtype=np.float32),         # 1
            ]
        ).astype(np.float32)

        if obs.shape != OBSERVATION_SHAPE:
            raise RuntimeError(
                f"Bad obs shape: got {obs.shape}, expected {OBSERVATION_SHAPE}. "
                f"For this rewrite set OBS_DIM = 24 and OBSERVATION_SHAPE = (24,)."
            )

        return obs

    # =========================================================================
    # DEBUGGING / CAUSALITY
    # =========================================================================

    def _action_causality_info(self, pre_pose, post_pose, commanded_waypoint):
        pre_pos = np.asarray(pre_pose[:3], dtype=np.float32)
        post_pos = np.asarray(post_pose[:3], dtype=np.float32)
        cmd_pos = np.asarray(commanded_waypoint[:3], dtype=np.float32)

        command_vec_xyz = cmd_pos - pre_pos
        actual_vec_xyz = post_pos - pre_pos

        command_norm_xyz = float(np.linalg.norm(command_vec_xyz))
        actual_norm_xyz = float(np.linalg.norm(actual_vec_xyz))

        if command_norm_xyz > 1e-8:
            command_dir_xyz = command_vec_xyz / command_norm_xyz
            actual_motion_along_command_xyz = float(
                np.dot(actual_vec_xyz, command_dir_xyz)
            )
        else:
            actual_motion_along_command_xyz = 0.0

        target_vec_xyz = self.target_array[:3] - pre_pos
        if self.curriculum_mode != "true_6dof":
            target_vec_xyz[2] = 0.0

        target_norm_xyz = float(np.linalg.norm(target_vec_xyz))

        if target_norm_xyz > 1e-8:
            target_dir_xyz = target_vec_xyz / target_norm_xyz
            actual_motion_toward_target_xyz = float(
                np.dot(actual_vec_xyz, target_dir_xyz)
            )
        else:
            actual_motion_toward_target_xyz = 0.0

        pre_xy = pre_pos[:2]
        post_xy = post_pos[:2]
        cmd_xy = cmd_pos[:2]

        command_vec_xy = cmd_xy - pre_xy
        actual_vec_xy = post_xy - pre_xy

        command_norm_xy = float(np.linalg.norm(command_vec_xy))
        actual_norm_xy = float(np.linalg.norm(actual_vec_xy))

        if command_norm_xy > 1e-8:
            command_dir_xy = command_vec_xy / command_norm_xy
            actual_motion_along_command_xy = float(
                np.dot(actual_vec_xy, command_dir_xy)
            )
        else:
            actual_motion_along_command_xy = 0.0

        return {
            "pre_xyz": pre_pos.copy(),
            "post_xyz": post_pos.copy(),
            "commanded_xyz": cmd_pos.copy(),
            "actual_xyz_motion": actual_vec_xyz.copy(),
            "command_xyz_norm": command_norm_xyz,
            "actual_xyz_motion_norm": actual_norm_xyz,
            "actual_motion_along_command_xyz": actual_motion_along_command_xyz,
            "actual_motion_toward_target_xyz": actual_motion_toward_target_xyz,

            "pre_xy": pre_xy.copy(),
            "post_xy": post_xy.copy(),
            "commanded_xy": cmd_xy.copy(),
            "actual_xy_motion": actual_vec_xy.copy(),
            "command_xy_norm": command_norm_xy,
            "actual_xy_motion_norm": actual_norm_xy,
            "actual_motion_along_command": actual_motion_along_command_xy,
            "actual_motion_along_command_xy": actual_motion_along_command_xy,
        }

    # =========================================================================
    # STATE
    # =========================================================================

    def _update_reward_state(self, reward_info):
        new_pose = reward_info["current_object_pose"].copy()
        new_pos = new_pose[:3].copy()

        self.prev_move_xy = (new_pos[:2] - self.prev_object_pos[:2]).astype(np.float32)
        self.prev_move_xyz = (new_pos[:3] - self.prev_object_pos[:3]).astype(np.float32)

        self.prev_object_pose = new_pose
        self.prev_object_pos = new_pos

        self.prev_xy_dist = float(reward_info["current_xy_error"])
        self.prev_xyz_dist = float(reward_info.get("current_xyz_error", self.prev_xy_dist))
        self.prev_z_err = float(reward_info.get("current_z_error", 0.0))

        self.prev_rot_error_rad = float(reward_info["current_rotation_error_rad"])
        self.prev_rot_error_deg = float(reward_info["current_rotation_error"])

    # =========================================================================
    # UTILITY
    # =========================================================================

    def _bounds_info(self, object_pose):
        object_pose = np.asarray(object_pose, dtype=np.float32)
        object_pos = object_pose[:3]

        object_xy_out_of_bounds = bool(
            abs(float(object_pos[0])) > WORKSPACE_XY_LIMIT
            or abs(float(object_pos[1])) > WORKSPACE_XY_LIMIT
        )

        object_fell = bool(float(object_pos[2]) < MIN_OBJECT_Z)
        object_too_high = bool(float(object_pos[2]) > MAX_OBJECT_Z)

        object_out_of_bounds = bool(
            object_xy_out_of_bounds
            or object_fell
            or object_too_high
        )

        return {
            "object_out_of_bounds": object_out_of_bounds,
            "object_xy_out_of_bounds": object_xy_out_of_bounds,
            "object_fell": object_fell,
            "object_too_high": object_too_high,
            "object_z": float(object_pos[2]),
            "object_xy_radius": float(np.linalg.norm(object_pos[:2])),
        }

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


# =============================================================================
# ENV FACTORY
# =============================================================================

def make_env(render_mode=None, curriculum_mode=CURRICULUM_MODE):
    config = PinArrayManipulatorConfig(
        manipulator_size=MANIPULATOR_SIZE,
        pins_per_side=PINS_PER_SIDE,
        pin_height=PIN_HEIGHT,
        actuation_length=ACTUATION_LENGTH,
        pin_spacing=PIN_SPACING,
        has_wall=HAS_WALL,
        rounded_pins=ROUNDED_PINS,
    )

    ball = Ball(diameter=BALL_DIAMETER, starting_z=BALL_STARTING_Z)

    reward_model = Distance3DRewardModel(manipulator_config=config)

    target_generator = MultiTargetGenerator(
        simulation_object=ball,
        manipulator_config=config,
    )

    env = CleanSphereCurriculumEnv(
        simulation_object=ball,
        target_generator=target_generator,
        reward_model=reward_model,
        manipulator_config=config,
        render_mode=render_mode,
        curriculum_mode=curriculum_mode,
        max_episode_steps=MAX_EPISODE_STEPS,
        action_repeat=ACTION_REPEAT,
    )

    return Monitor(env)


def make_one_env(rank, render_mode=None, curriculum_mode=CURRICULUM_MODE):
    def _init():
        env = make_env(render_mode=render_mode, curriculum_mode=curriculum_mode)
        env.action_space.seed(rank)
        env.observation_space.seed(rank)
        return env

    return _init


# =============================================================================
# TRAINING
# =============================================================================

def main():
    if N_ENVS == 1:
        env = DummyVecEnv([make_one_env(0)])
    else:
        env = SubprocVecEnv([make_one_env(i) for i in range(N_ENVS)])

    print("run dir:", PATHS.run_dir)
    print("curriculum mode:", CURRICULUM_MODE)
    print("obs space:", env.observation_space)
    print("action space:", env.action_space)
    print("max rotvec step:", MAX_ROTVEC_STEP_TRUE_6DOF if CURRICULUM_MODE == "true_6dof" else "mode-dependent")
    print("z target synced:", CURRICULUM_MODE != "true_6dof")

    policy_kwargs = dict(
        net_arch=POLICY_NET_ARCH,
        log_std_init=LOG_STD_INIT,
    )

    model = PPO(
        "MlpPolicy",
        env,
        policy_kwargs=policy_kwargs,
        verbose=1,
        tensorboard_log=PATHS.tensorboard_log,
        learning_rate=LEARNING_RATE,
        n_steps=N_STEPS_BASE // N_ENVS,
        batch_size=BATCH_SIZE,
        n_epochs=N_EPOCHS,
        gamma=GAMMA,
        gae_lambda=GAE_LAMBDA,
        clip_range=CLIP_RANGE,
        ent_coef=ENT_COEF,
        vf_coef=VF_COEF,
        max_grad_norm=MAX_GRAD_NORM,
        target_kl=TARGET_KL,
        device=DEVICE,
    )

    try:
        model.learn(total_timesteps=TOTAL_TIMESTEPS)
        save_model(model, PATHS, RUN_NAME)
    finally:
        env.close()


if __name__ == "__main__":
    main()