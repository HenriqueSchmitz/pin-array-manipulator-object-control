# =============================================================================
# CLEAN SPHERE / TRUE 6DOF CURRICULUM CONSTANTS
# =============================================================================

import numpy as np

# =============================================================================
# RUN CONFIG
# =============================================================================

# Use a new run name because true_6dof uses a 24D observation.
# Old 20D models are not compatible.
RUN_NAME = "sphere_true_6dof_v1"

CURRICULUM_MODE = "true_6dof"
# Options:
# CURRICULUM_MODE = "xy"
# CURRICULUM_MODE = "xy_z"
# CURRICULUM_MODE = "xy_rot"
# CURRICULUM_MODE = "pose_6d"
# CURRICULUM_MODE = "true_6dof"

# Use 1 while debugging SubprocVecEnv crashes.
# Once reset/training starts cleanly, set this back to 4.
N_ENVS = 4
TOTAL_TIMESTEPS = 500_000

# =============================================================================
# MANIPULATOR / OBJECT CONFIG
# =============================================================================

MANIPULATOR_SIZE = 1.0
PINS_PER_SIDE = 15
PIN_HEIGHT = 0.15
ACTUATION_LENGTH = 0.1
PIN_SPACING = 0.001
HAS_WALL = True
ROUNDED_PINS = True

BALL_DIAMETER = 0.1
BALL_STARTING_Z = 0.2

# =============================================================================
# ENV / CONTROL CONFIG
# =============================================================================

MAX_EPISODE_STEPS = 2000
ACTION_REPEAT = 2

BASE_SEEK_SPEED = 5e-4
MIN_SEEK_SPEED = 1e-4

# Unified observation used by the rewritten training file:
#   rel_xyz:         3
#   xyz_dist:        1
#   goal_dir_xyz:    3
#   object_vel_xyz:  3
#   prev_move_xyz:   3
#   last_action:     6
#   normalized_time: 1
#   rot_error_vec:   3
#   rot_error_rad:   1
# total = 24
OBS_DIM = 24
ACTION_DIM = 6

OBSERVATION_SHAPE = (OBS_DIM,)
ACTION_SHAPE = (ACTION_DIM,)

# Policy action:
#   [dx, dy, dz, drotvec_x, drotvec_y, drotvec_z]
MAX_XY_STEP = 0.015
MAX_Z_STEP = 0.002

# For XY-only pretraining.
MAX_ROTVEC_STEP_XY = np.array([0.0, 0.0, 0.0], dtype=np.float32)

# For light auxiliary rotation.
MAX_ROTVEC_STEP_LIGHT = np.array([0.005, 0.005, 0.005], dtype=np.float32)

# Legacy full pose mode.
MAX_ROTVEC_STEP_FULL = np.array([0.02, 0.02, 0.02], dtype=np.float32)

# True 6DOF mode.
# Start conservative; increase only if rotation cannot change.
MAX_ROTVEC_STEP_TRUE_6DOF = np.array([0.02, 0.02, 0.02], dtype=np.float32)

# =============================================================================
# TARGET CONFIG
# =============================================================================

# Use base generator target unless it is too close.
TRIVIAL_TARGET_XY_RADIUS = 0.05

# Fallback deterministic target.
SYNTHETIC_TARGET_DELTA_XY = np.array([0.12, -0.08], dtype=np.float32)
SYNTHETIC_TARGET_XY_LIMIT = 0.35

# For true_6dof synthetic fallback.
# 0.0 means maintain current height while tracking XY + rotation.
# Try small values like -0.003 or +0.003 only after the basic mode works.
SYNTHETIC_TARGET_DELTA_Z = 0.0

# Rotation target.
# For a sphere, this may be physically artificial unless orientation is meaningful.
SYNTHETIC_TARGET_ROT_DELTA_DEG = np.array([20.0, -10.0, 25.0], dtype=np.float32)

# =============================================================================
# SUCCESS / FAILURE
# =============================================================================

SUCCESS_XY_RADIUS = 0.025
SUCCESS_ROT_DEG = 10.0
SUCCESS_ROT_RAD = np.deg2rad(SUCCESS_ROT_DEG)

# True 6DOF success thresholds.
SUCCESS_X_RADIUS = 0.025
SUCCESS_Y_RADIUS = 0.025
SUCCESS_Z_RADIUS = 0.010
SUCCESS_XYZ_RADIUS = 0.035

FAILURE_RADIUS = 0.75

# Conservative bounds. Adjust if your sim workspace differs.
WORKSPACE_XY_LIMIT = 0.5
MIN_OBJECT_Z = 0.02
MAX_OBJECT_Z = 1.0

# =============================================================================
# LEGACY / AUXILIARY REWARD CONSTANTS
# =============================================================================

# Kept for compatibility with older branches/modes.
XY_PROGRESS_REWARD_COEFF = 80.0
TOWARD_GOAL_REWARD_COEFF = 40.0
XY_DISTANCE_PENALTY_COEFF = -2.0

STEP_PENALTY_COEFF = -0.01
STUCK_PENALTY_COEFF = -0.05

TERMINAL_REWARD = 10.0

ROTATION_NEAR_XY_RADIUS = 0.08
ROTATION_PROGRESS_REWARD_COEFF = 0.01
ROTATION_WORSENING_PENALTY_COEFF = -0.01
ROTATION_ERROR_PENALTY_COEFF = -0.001
ROTATION_SCALING_FACTOR = 1.0

# =============================================================================
# ADVANCED REWARD CONSTANTS
# =============================================================================

# XY / xy_rot / pose_6d potential coefficient.
TASK_POTENTIAL_COEFF = 200.0

# Rotation turns on near the XY goal in non-true-6DOF modes.
ROTATION_ON_RADIUS = 0.08
ROTATION_ON_TEMPERATURE = 0.015
ROTATION_POTENTIAL_WEIGHT = 0.05

# Auxiliary rotation progress only gets credit when XY progress is positive.
XY_PROGRESS_EPS = 1e-4
XY_PROGRESS_TEMPERATURE = 5e-4
AUX_ROT_PROGRESS_COEFF = 0.10

# Penalize buying rotation by worsening XY translation in non-true-6DOF modes.
ROT_TRANSLATION_CONFLICT_COEFF = 200.0

# Small directional shaping toward XY target.
DIRECTIONAL_REWARD_COEFF = 20.0

# Smooth dense bonus near target.
GOAL_KERNEL_COEFF = 0.5
GOAL_KERNEL_SIGMA = 0.025

# Smooth safety barrier.
BARRIER_COST_COEFF = 0.05
BARRIER_MARGIN = 0.04
BARRIER_SHARPNESS = 50.0

# Control costs.
ACTION_XY_COST_COEFF = 0.01
ACTION_Z_COST_COEFF = 0.01
ACTION_ROT_COST_COEFF = 0.01
ACTION_JERK_COST_COEFF = 0.005

# Time/stuck costs.
STEP_COST = 0.01
STUCK_COST = 0.05
STUCK_MOVE_EPS = 1e-5

# Terminal terms.
SUCCESS_BONUS = 10.0
FAILURE_PENALTY = 10.0

# Reward clipping.
REWARD_CLIP_LOW = -10.0
REWARD_CLIP_HIGH = 10.0

# =============================================================================
# TRUE 6DOF REWARD CONSTANTS
# =============================================================================

# Per-axis translation weights inside true 6DOF potential.
POSE_W_X = 1.0
POSE_W_Y = 1.0
POSE_W_Z = 1.0

# Rotation energy weight for true 6DOF potential.
POSE_W_ROT = 0.10

# Potential scale.
TRUE_6DOF_POTENTIAL_COEFF = 200.0

# Penalize remaining XYZ/Z error.
XYZ_DISTANCE_COST_COEFF = 2.0
Z_ERROR_COST_COEFF = 5.0

# Mild penalty on remaining local rotvec components.
# This is not a true "axis balance" constraint; it is a small stabilizer.
ROT_AXIS_BALANCE_COST_COEFF = 0.01

# =============================================================================
# PPO CONFIG
# =============================================================================

POLICY_NET_ARCH = dict(pi=[256, 256], vf=[256, 256])
LOG_STD_INIT = -0.5

LEARNING_RATE = 1e-4
N_STEPS_BASE = 4096
BATCH_SIZE = 512
N_EPOCHS = 8
GAMMA = 0.98
GAE_LAMBDA = 0.95
CLIP_RANGE = 0.15
ENT_COEFF = 0.01
ENT_COEF = ENT_COEFF
VF_COEF = 0.5
MAX_GRAD_NORM = 0.5
TARGET_KL = 0.03
DEVICE = "cpu"