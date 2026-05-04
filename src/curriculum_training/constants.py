# =============================================================================
# ENVIRONMENT AND SIMULATION CONSTANTS
# =============================================================================
# These define the physical setup of the pin array manipulator and the object.

# Manipulator configuration parameters
MANIPULATOR_SIZE = 1.0  # Overall size of the manipulator in meters
PINS_PER_SIDE = 15  # Number of pins along each side of the square array
PIN_HEIGHT = 0.15  # Height of each pin in meters
ACTUATION_LENGTH = 0.1  # Maximum actuation length for each pin in meters
PIN_SPACING = 0.001  # Spacing between pins in meters
HAS_WALL = True  # Whether the manipulator has a surrounding wall
ROUNDED_PINS = True  # Whether pins have rounded tips for smoother contact

# Object (ball) parameters
BALL_DIAMETER = 0.1  # Diameter of the ball object in meters
BALL_STARTING_Z = 0.2  # Initial Z position of the ball in meters

# Episode control
MAX_EPISODE_STEPS = 2000  # Maximum steps per episode before timeout

# =============================================================================
# ACTION AND CONTROL CONSTANTS
# =============================================================================
# These control how actions are processed and scaled.

# Seek speeds for contact-seeking behavior
BASE_SEEK_SPEED = 5e-4  # Base speed for seeking contact
MIN_SEEK_SPEED = 1e-4  # Minimum speed for seeking contact

# Action processing
ACTION_REPEAT = 2  # Number of times to repeat each action in the environment
MAX_XY_STEP = 0.012  # Maximum step size in XY plane per action

# Z-axis control (limited for stability)
DELTA_Z_SCALING = 0.15  # Scaling factor for Z delta commands
Z_CLIP_DELTA = 0.002  # Maximum Z deviation from current position

# Rotation control (small deltas for stability)
DELTA_ROTVEC_SCALING = [0.05, 0.05, 0.05]  # Scaling for rotation vector deltas (roll, pitch, yaw)

# =============================================================================
# SUCCESS AND FAILURE THRESHOLDS
# =============================================================================
# Thresholds for determining episode success/failure.

# Distance-based thresholds
SUCCESS_RADIUS = 0.018  # XY distance threshold for success in meters
FAILURE_RADIUS = 0.75  # XY distance threshold for failure in meters

# Rotation-based thresholds (in degrees)
SUCCESS_ROT_DEG = 15.0  # Rotation error threshold for success
FAILURE_ROT_DEG = 170.0  # Rotation error threshold for failure

# =============================================================================
# REWARD COEFFICIENTS
# =============================================================================
# Coefficients for shaping the reward function.

# Progress and movement rewards
PROGRESS_REWARD_COEFF = 80.0  # Reward for reducing distance to target
TOWARD_GOAL_REWARD_COEFF = 20.0  # Reward for movement toward the goal

# Penalties
DISTANCE_PENALTY_COEFF = -0.4  # Penalty for distance to target
ACTION_PENALTY_COEFF = -0.01  # Penalty for action magnitude
STEP_PENALTY_COEFF = -0.002  # Penalty per step
STUCK_PENALTY_COEFF = -0.01  # Penalty for being stuck

# Rotation-related rewards/penalties
ROTATION_PROGRESS_REWARD_COEFF = 0.015  # Reward for rotation progress
ROTATION_WORSENING_PENALTY_COEFF = -0.004  # Penalty for rotation worsening
ROTATION_SCALING_FACTOR = 20  # Scaling factor for rotation penalties

# Terminal rewards
TERMINAL_REWARD = 10.0  # Reward for successful episode
FAILURE_PENALTY = -10.0  # Penalty for failed episode

# Gating and scaling
XY_GATE_SCALE = 0.06  # Scale for XY gating in rotation rewards

# =============================================================================
# TRAINING CONSTANTS
# =============================================================================
# Parameters for the training setup.

N_ENVS = 4  # Number of parallel environments
TOTAL_TIMESTEPS = 300_000  # Total training timesteps

# =============================================================================
# PPO HYPERPARAMETERS
# =============================================================================
# Hyperparameters for the PPO algorithm.

LEARNING_RATE = 1e-4  # Learning rate for the optimizer
N_STEPS_BASE = 4096  # Base number of steps per PPO update (divided by N_ENVS)
BATCH_SIZE = 512  # Batch size for training
N_EPOCHS = 8  # Number of epochs per PPO update
GAMMA = 0.98  # Discount factor
GAE_LAMBDA = 0.95  # GAE lambda parameter
CLIP_RANGE = 0.15  # PPO clip range
ENT_COEF = 0.01  # Entropy coefficient
VF_COEF = 0.5  # Value function coefficient
MAX_GRAD_NORM = 0.5  # Maximum gradient norm
TARGET_KL = 0.03  # Target KL divergence for early stopping
LOG_STD_INIT = -0.5  # Initial log standard deviation for policy

# =============================================================================
# POLICY NETWORK ARCHITECTURE
# =============================================================================
# Architecture for the neural network policy.

POLICY_NET_ARCH = dict(pi=[256, 256], vf=[256, 256])  # Policy and value network layers

# =============================================================================
# OBSERVATION AND ACTION SPACE CONSTANTS
# =============================================================================
# Shapes for observation and action spaces.

OBSERVATION_SHAPE = (20,)  # Shape of the observation space
ACTION_SHAPE = (6,)  # Shape of the action space

# =============================================================================
# SYNTHETIC TARGET CONSTANTS
# =============================================================================
# Constants for generating synthetic targets when needed.

SYNTHETIC_TARGET_XY_DELTA = [0.12, -0.08]  # XY offset for synthetic targets
SYNTHETIC_TARGET_ROT_DELTA_DEG = [20.0, -10.0, 25.0]  # Rotation offset in degrees
INIT_DIST_THRESHOLD = 0.05  # Threshold for triggering synthetic target generation