from typing import Any, Optional

import numpy as np
from gymnasium import spaces

from pin_array_manipulator_object_control.control.composite_control import CompositeControlPolicy
from pin_array_manipulator_object_control.environment.pin_array_env import PinArrayEnv
from pin_array_manipulator_object_control.manipulator.pin_array_manipulator import PinArrayManipulatorConfig
from pin_array_manipulator_object_control.objects.object import Object
from pin_array_manipulator_object_control.rewards.base_model import RewardModel
from pin_array_manipulator_object_control.routines.target_generator import PinArrayTargetGenerator


class CompositeControlEnv(PinArrayEnv):
    def __init__(self,
                 simulation_object: Object,
                 target_generator: PinArrayTargetGenerator,
                 reward_model: RewardModel,
                 manipulator_config: Optional[PinArrayManipulatorConfig] = None,
                 render_mode: Optional[str] = None,
                 max_episode_steps: int = -1):
        super().__init__(simulation_object,
                         target_generator,
                         reward_model,
                         manipulator_config,
                         render_mode,
                         max_episode_steps)
        if not manipulator_config:
            manipulator_config = PinArrayManipulatorConfig()
        self.composite_control_policy = CompositeControlPolicy(manipulator_config)
        incremental_pose_action_length = 6
        contact_seeking_speeds_action_length = 2
        action_space_length = incremental_pose_action_length + contact_seeking_speeds_action_length
        self.action_space = spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(action_space_length,), 
            dtype=np.float32
        )
        self.last_obs = None

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        obs, info = super().reset(seed=seed, options=options)
        self.last_obs = obs
        return obs, info

    def step(self, action):
        base_seek_speed = action[0]
        min_seek_speed = action[1]
        self.composite_control_policy.update_contact_seeking_speeds(base_seek_speed, min_seek_speed)
        incremental_target = action[2:8]
        if self.last_obs is None:
            raise Exception("Environment has not been reset.")
        action = self.composite_control_policy.sample(target=incremental_target, observation=self.last_obs)
        obs, reward, terminated, truncated, info = super().step(action)
        self.last_obs = obs
        return obs, reward, terminated, truncated, info