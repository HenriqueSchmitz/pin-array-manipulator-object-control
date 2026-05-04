# =============================================================================
# IMPORTS
# =============================================================================

# Standard library imports
import sys
from pathlib import Path

# Third-party imports
import numpy as np
from gymnasium import spaces
from scipy.spatial.transform import Rotation as R
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv

# Local imports
from rl_common import setup, save_model
from constants import *
from pin_array_manipulator_object_control.environment.composite_control_env import CompositeControlEnv
from pin_array_manipulator_object_control.manipulator.observation import PinArrayEnvObservation
from pin_array_manipulator_object_control.manipulator.pin_array_manipulator import PinArrayManipulatorConfig
from pin_array_manipulator_object_control.objects.ball import Ball
from pin_array_manipulator_object_control.rewards.distance_3d import Distance3DRewardModel
from pin_array_manipulator_object_control.routines.multi_target_generator import MultiTargetGenerator

# =============================================================================
# SETUP
# =============================================================================

SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

PATHS = setup(__file__, "curriculum_v1", "curriculum_v1")


class ReliableSphere6DOFEnv(CompositeControlEnv):
    """
    Reliable sphere baseline environment for 6DOF control.

    Action space: 6D [dx, dy, dz_unused, drotvec_x, drotvec_y, drotvec_z]
    - XY control is the primary task.
    - Z is constrained to current object height for stability.
    - Rotation is logged but not strictly required for success.
    - Rotation commands are small deltas for reliability.
    """

    # =========================================================================
    # INITIALIZATION
    # =========================================================================

    def __init__(self, *args, manipulator_config: PinArrayManipulatorConfig, **kwargs):
        self.wrapper_max_episode_steps = int(kwargs.pop("max_episode_steps", MAX_EPISODE_STEPS))
        super().__init__(*args, manipulator_config=manipulator_config, **kwargs)

        self.config = manipulator_config
        self.raw_obs_dim = int(np.prod(self.observation_space.shape))
        self.action_repeat = ACTION_REPEAT

        # Observation space: 20D vector with relative pose, velocities, etc.
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=OBSERVATION_SHAPE, dtype=np.float32
        )

        # Action space: 6D normalized actions
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=ACTION_SHAPE, dtype=np.float32
        )

        # Control parameters
        self.max_xy_step = MAX_XY_STEP
        self.base_seek_speed = BASE_SEEK_SPEED
        self.min_seek_speed = MIN_SEEK_SPEED

        # Success/failure thresholds
        self.success_radius = SUCCESS_RADIUS
        self.failure_radius = FAILURE_RADIUS
        self.success_rot_deg = SUCCESS_ROT_DEG
        self.failure_rot_deg = FAILURE_ROT_DEG

        # Episode state
        self.step_count = 0
        self.current_raw_obs = None
        self.target_array = None

        # Previous state for reward computation
        self.prev_object_pose = None
        self.prev_object_pos = None
        self.prev_dist = None
        self.prev_rot_error_deg = None
        self.prev_move_xy = np.zeros(2, dtype=np.float32)
        self.last_action = np.zeros(6, dtype=np.float32)

    # =========================================================================
    # ENVIRONMENT INTERFACE
    # =========================================================================

    def reset(self, **kwargs):
        """Reset the environment and initialize episode state."""
        raw_obs, info = super().reset(**kwargs)

        self.step_count = 0
        self.current_raw_obs = self._trim_raw_obs(raw_obs)
        object_pose = self._object_pose(self.current_raw_obs)

        # Set target: use generator XY but override Z to match object height
        self.target_array = np.asarray(info["target"], dtype=np.float32).copy()
        self.target_array[2] = object_pose[2]

        # Generate synthetic target if initial distance is too small
        init_dist = self._xy_distance(object_pose, self.target_array)
        if init_dist < INIT_DIST_THRESHOLD:
            self.target_array = self._generate_synthetic_target(object_pose)

        # Initialize previous state
        self.prev_object_pose = object_pose.copy()
        self.prev_object_pos = object_pose[:3].copy()
        self.prev_dist = self._xy_distance(object_pose, self.target_array)
        self.prev_rot_error_deg = self._rotation_error_deg(object_pose, self.target_array)
        self.prev_move_xy = np.zeros(2, dtype=np.float32)
        self.last_action = np.zeros(6, dtype=np.float32)

        # Prepare output info
        out_info = dict(info)
        out_info.update({
            "target": self.target_array.copy(),
            "initial_distance": self.prev_dist,
            "initial_translation_error": self.prev_dist,
            "initial_rotation_error": self.prev_rot_error_deg,
            "current_translation_error": self.prev_dist,
            "current_rotation_error": self.prev_rot_error_deg,
            "distance_to_target": self.prev_dist,
            "success": False,
            "is_success": False,
            "failure": False,
        })

        return self._compact_obs(self.current_raw_obs), out_info

    def step(self, action):
        """Execute one step in the environment."""
        self.step_count += 1
        policy_action = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
        self.last_action = policy_action.copy()

        composite_action, debug_info = self._action_to_composite_action(policy_action)

        # Render debug visuals if in human mode
        if self.render_mode == "human":
            try:
                self.update_debug_visuals(composite_action[2:])
            except Exception:
                pass

        # Execute action with repetition
        total_base_reward = 0.0
        for _ in range(self.action_repeat):
            raw_obs, base_reward, base_terminated, base_truncated, info = super().step(composite_action)
            total_base_reward += base_reward
            if base_terminated or base_truncated:
                break

        self.current_raw_obs = self._trim_raw_obs(raw_obs)

        # Sync target Z with current object height
        object_pose = self._object_pose(self.current_raw_obs)
        self.target_array[2] = object_pose[2]

        # Compute reward and check termination
        reward, reward_info = self._compute_reward(self.current_raw_obs, policy_action)
        success = bool(reward_info["success"])
        failure = bool(reward_info["failure"])
        timeout = self.step_count >= self.wrapper_max_episode_steps

        # Termination logic: ignore base termination for sphere wrapper
        terminated = bool(success or failure)
        truncated = bool(timeout and not terminated)

        self._update_reward_state(reward_info)
        obs = self._compact_obs(self.current_raw_obs)

        # Prepare output info
        out_info = dict(info)
        out_info.update(debug_info)
        out_info.update(reward_info)
        out_info.update({
            "target": self.target_array.copy(),
            "success": success,
            "is_success": success,
            "failure": failure,
            "base_reward": float(total_base_reward),
            "base_terminated": bool(base_terminated),
            "base_truncated": bool(base_truncated),
            "TimeLimit.truncated": bool(truncated),
        })

        return obs, float(reward), terminated, truncated, out_info

    # =========================================================================
    # ACTION PROCESSING
    # =========================================================================

    def _action_to_composite_action(self, policy_action):
        """Convert policy action to composite action for the environment."""
        current_pose = self._object_pose(self.current_raw_obs)
        current_pos = current_pose[:3]

        # Compute deltas
        delta_xy = policy_action[:2] * self.max_xy_step
        delta_z = float(policy_action[2]) * DELTA_Z_SCALING

        # Build waypoint
        waypoint = current_pose.copy()
        waypoint[0] = current_pos[0] + delta_xy[0]
        waypoint[1] = current_pos[1] + delta_xy[1]
        waypoint[2] = np.clip(
            current_pose[2] + delta_z,
            current_pose[2] - Z_CLIP_DELTA,
            current_pose[2] + Z_CLIP_DELTA,
        )

        # Apply small rotation deltas
        delta_rotvec = policy_action[3:] * np.array(DELTA_ROTVEC_SCALING, dtype=np.float32)
        r_current = R.from_euler("xyz", current_pose[3:], degrees=True)
        r_delta = R.from_rotvec(delta_rotvec.astype(np.float64))
        r_new = r_delta * r_current
        waypoint[3:] = r_new.as_euler("xyz", degrees=True).astype(np.float32)

        # Create composite action
        composite_action = np.concatenate([
            np.array([self.base_seek_speed, self.min_seek_speed], dtype=np.float32),
            waypoint.astype(np.float32),
        ]).astype(np.float32)

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

    # =========================================================================
    # REWARD COMPUTATION
    # =========================================================================

    def _compute_reward(self, raw_obs, policy_action):
        """Compute reward based on current state and action."""
        object_pose = self._object_pose(raw_obs)
        object_pos = object_pose[:3]

        # Current errors and progress
        curr_dist = self._xy_distance(object_pose, self.target_array)
        progress = float(self.prev_dist - curr_dist)
        rot_error_deg = self._rotation_error_deg(object_pose, self.target_array)
        rot_progress_deg = float(self.prev_rot_error_deg - rot_error_deg)

        # Movement analysis
        move_vec = object_pos - self.prev_object_pos
        move_xy = move_vec[:2]
        move_xy_norm = float(np.linalg.norm(move_xy))

        goal_vec = self.target_array[:2] - object_pos[:2]
        goal_dist = float(np.linalg.norm(goal_vec))
        goal_dir = goal_vec / goal_dist if goal_dist > 1e-8 else np.zeros(2, dtype=np.float32)
        movement_toward_goal = float(np.dot(move_xy, goal_dir))

        # Success/failure checks
        xy_success = curr_dist <= self.success_radius
        rot_success = rot_error_deg <= self.success_rot_deg
        success = bool(xy_success and rot_success)
        failure = bool(curr_dist > self.failure_radius)

        # Reward components
        progress_reward = PROGRESS_REWARD_COEFF * progress
        toward_goal_reward = TOWARD_GOAL_REWARD_COEFF * movement_toward_goal
        distance_penalty = DISTANCE_PENALTY_COEFF * curr_dist
        action_penalty = ACTION_PENALTY_COEFF * float(np.linalg.norm(policy_action[:2]))
        step_penalty = STEP_PENALTY_COEFF

        stuck_penalty = STUCK_PENALTY_COEFF if curr_dist > self.success_radius and move_xy_norm < 1e-5 else 0.0

        # Rotation rewards (gated by XY distance)
        xy_gate = float(np.exp(-curr_dist / XY_GATE_SCALE))
        rotation_progress_reward = ROTATION_PROGRESS_REWARD_COEFF * xy_gate * rot_progress_deg
        rotation_worsening_penalty = ROTATION_WORSENING_PENALTY_COEFF * xy_gate * max(0.0, -rot_progress_deg) * ROTATION_SCALING_FACTOR

        # Terminal rewards
        terminal_reward = TERMINAL_REWARD if success else 0.0
        failure_penalty = FAILURE_PENALTY if failure else 0.0

        # Total reward
        reward_unclipped = (
            progress_reward + toward_goal_reward + distance_penalty + action_penalty +
            step_penalty + stuck_penalty + rotation_progress_reward + rotation_worsening_penalty +
            terminal_reward + failure_penalty
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

    # =========================================================================
    # OBSERVATION PROCESSING
    # =========================================================================

    def _compact_obs(self, raw_obs):
        """Create compact observation from raw observation."""
        obs_obj = self._parse_obs(raw_obs)
        object_pose = obs_obj.object_pose.array().astype(np.float32)
        object_pos = object_pose[:3]
        object_vel = obs_obj.object_velocity.array().astype(np.float32)

        rel_xy = self.target_array[:2].astype(np.float32) - object_pos[:2]
        dist = float(np.linalg.norm(rel_xy))
        goal_dir = rel_xy / dist if dist > 1e-8 else np.zeros(2, dtype=np.float32)

        rot_error_vec = self._rotation_error_vec(object_pose, self.target_array)
        rot_error_rad = float(np.linalg.norm(rot_error_vec))
        normalized_time = float(self.step_count) / float(max(1, self.wrapper_max_episode_steps))

        return np.concatenate([
            rel_xy.astype(np.float32),
            np.array([dist], dtype=np.float32),
            goal_dir.astype(np.float32),
            object_vel[:2].astype(np.float32),
            self.prev_move_xy.astype(np.float32),
            self.last_action.astype(np.float32),
            np.array([normalized_time], dtype=np.float32),
            rot_error_vec.astype(np.float32),
            np.array([rot_error_rad], dtype=np.float32),
        ]).astype(np.float32)

    # =========================================================================
    # STATE MANAGEMENT
    # =========================================================================

    def _update_reward_state(self, reward_info):
        """Update previous state for reward computation."""
        new_pose = reward_info["current_object_pose"].copy()
        new_pos = new_pose[:3].copy()

        self.prev_move_xy = (new_pos[:2] - self.prev_object_pos[:2]).astype(np.float32)
        self.prev_object_pose = new_pose
        self.prev_object_pos = new_pos
        self.prev_dist = float(reward_info["current_translation_error"])
        self.prev_rot_error_deg = float(reward_info["current_rotation_error"])

    # =========================================================================
    # UTILITY METHODS
    # =========================================================================

    def _generate_synthetic_target(self, object_pose):
        """Generate a synthetic target when initial distance is too small."""
        target = object_pose.copy()
        target[:2] = np.clip(
            object_pose[:2] + np.array(SYNTHETIC_TARGET_XY_DELTA, dtype=np.float32),
            -0.35, 0.35
        )
        target[2] = object_pose[2]

        r_obj = R.from_euler("xyz", object_pose[3:], degrees=True)
        r_delta = R.from_euler("xyz", SYNTHETIC_TARGET_ROT_DELTA_DEG, degrees=True)
        target[3:] = (r_delta * r_obj).as_euler("xyz", degrees=True).astype(np.float32)

        return target

    def _object_pose(self, raw_obs):
        """Extract object pose from raw observation."""
        obs_obj = self._parse_obs(raw_obs)
        return obs_obj.object_pose.array().astype(np.float32).copy()

    def _trim_raw_obs(self, obs):
        """Trim raw observation to expected dimensions."""
        raw = np.asarray(obs, dtype=np.float32).reshape(-1)
        return raw[:self.raw_obs_dim].copy()

    def _parse_obs(self, obs):
        """Parse raw observation into structured object."""
        raw = self._trim_raw_obs(obs)
        return PinArrayEnvObservation.from_array(raw, self.config.pins_per_side)

    @staticmethod
    def _xy_distance(object_pose, target_pose):
        """Compute XY distance between poses."""
        a = np.asarray(object_pose, dtype=np.float32)
        b = np.asarray(target_pose, dtype=np.float32)
        return float(np.linalg.norm(a[:2] - b[:2]))

    @staticmethod
    def _rotation_error_vec(object_pose, target_pose):
        """Compute rotation error as rotation vector."""
        object_pose = np.asarray(object_pose, dtype=np.float32)
        target_pose = np.asarray(target_pose, dtype=np.float32)

        r_obj = R.from_euler("xyz", object_pose[3:], degrees=True)
        r_tgt = R.from_euler("xyz", target_pose[3:], degrees=True)
        r_err = r_tgt * r_obj.inv()

        return r_err.as_rotvec().astype(np.float32)

    @classmethod
    def _rotation_error_deg(cls, object_pose, target_pose):
        """Compute rotation error in degrees."""
        rotvec = cls._rotation_error_vec(object_pose, target_pose)
        return float(np.degrees(np.linalg.norm(rotvec)))


# =============================================================================
# ENVIRONMENT FACTORY
# =============================================================================

def make_env(render_mode=None):
    """Create the environment with configured parameters."""
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
        simulation_object=ball, manipulator_config=config
    )

    env = ReliableSphere6DOFEnv(
        simulation_object=ball,
        target_generator=target_generator,
        reward_model=reward_model,
        manipulator_config=config,
        render_mode=render_mode,
        max_episode_steps=MAX_EPISODE_STEPS,
    )

    return Monitor(env)


def make_one_env(rank, render_mode=None):
    """Create a seeded environment for parallel training."""
    def _init():
        env = make_env(render_mode=render_mode)
        env.action_space.seed(rank)
        env.observation_space.seed(rank)
        return env
    return _init


# =============================================================================
# TRAINING SCRIPT
# =============================================================================

def main():
    """Main training function."""
    env = SubprocVecEnv([make_one_env(i) for i in range(N_ENVS)])

    print("run dir:", PATHS.run_dir)
    print("obs space:", env.observation_space)
    print("action space:", env.action_space)

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
        device="cpu",
    )

    try:
        model.learn(total_timesteps=TOTAL_TIMESTEPS)
        save_model(model, PATHS, "curriculum_v1")
    finally:
        env.close()


if __name__ == "__main__":
    main()