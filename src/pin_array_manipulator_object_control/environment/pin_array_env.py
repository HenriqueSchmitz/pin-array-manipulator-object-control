from typing import Any, Optional

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import mujoco
import mujoco.viewer
from mujoco import MjModel, MjData # type: ignore


from pin_array_manipulator_object_control.environment.observation import PinArrayEnvObservation
from pin_array_manipulator_object_control.environment.rewards import RewardModel
from pin_array_manipulator_object_control.environment.target_generator import PinArrayTargetGenerator
from pin_array_manipulator_object_control.manipulator.pin_array_manipulator import PinArrayManipulator, PinArrayManipulatorConfig
from pin_array_manipulator_object_control.objects.object import Object



class PinArrayEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}

    def __init__(self,
                 simulation_object: Object,
                 target_generator: PinArrayTargetGenerator,
                 reward_model: RewardModel,
                 manipulator_config: Optional[PinArrayManipulatorConfig] = None,
                 render_mode: Optional[str] = None):
        super().__init__()
        if not manipulator_config:
            manipulator_config = PinArrayManipulatorConfig()
        self.target_pose = np.array([0.1, -0.2, 0.05]) # Target for the ball
        self.action_space = spaces.Box(
            low=-1.0, 
            high=1.0, 
            shape=(manipulator_config.pins_per_side, manipulator_config.pins_per_side), 
            dtype=np.float32
        )
        self.observation_space = spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(9,), 
            dtype=np.float32
        )
        self.simulation_object = simulation_object
        self.target_generator = target_generator
        self.reward_model = reward_model
        self.manipulator = PinArrayManipulator(config=manipulator_config)
        self.render_mode = render_mode
        self.model = None
        self.data = None
        self.viewer = None
        self.current_target = None

    def _get_obs(self):
        observation = PinArrayEnvObservation(
            object_pose = self.simulation_object.get_pose(),
            object_velocity = self.simulation_object.get_velocity(),
            pin_positions=np.array([]),
            pin_forces=np.array([])
        )
        return observation
    
    def _build_info(self, observation: PinArrayEnvObservation) -> dict[str, Any]:
        return {
            "target": self.current_target
        }

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed)
        xml = self._generate_xml()
        self.model = MjModel.from_xml_string(xml)
        self.data = MjData(self.model)
        self.manipulator.set_data(self.data)
        self.simulation_object.set_data(self.data)
        if self.render_mode == "human" and self.viewer is None:
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
        observation = self._get_obs()
        self.target_generator.reset()
        self.current_target = self.target_generator.get_current_target(observation)
        info = self._build_info(observation)
        return observation.array(), info

    def step(self, action):
        self.manipulator.actuate_from_matrix(action)
        mujoco.mj_step(self.model, self.data) # type: ignore
        observation = self._get_obs()
        if self.current_target is None:
            reward = 0
            terminated = True
        else:
            reward = self.reward_model.get_reward(target_pose=self.current_target, object_pose=observation.object_pose)
            terminated = False
        truncated = False
        self.current_target = self.target_generator.get_current_target(observation)
        info = self._build_info(observation)
        if self.render_mode == "human":
            self.viewer.sync() # type: ignore
        return self._get_obs(), reward, terminated, truncated, info

    def _generate_xml(self):
        manip_body = self.manipulator.generate_bodies()
        manip_act = self.manipulator.generate_actuators()
        object_body = self.simulation_object.generate_bodies()
        return f"""
        <mujoco>
            <option timestep="0.002" integrator="RK4" gravity="0 0 -9.81"/>
            <worldbody>
                <light diffuse=".5 .5 .5" pos="0 0 3" dir="0 0 -1"/>
                {manip_body}
                {object_body}
            </worldbody>
            <actuator>
                {manip_act}
            </actuator>
        </mujoco>
        """

    def render(self):
        if self.render_mode == "rgb_array":
            # Implementation for offscreen rendering if needed
            pass

    def close(self):
        if self.viewer is not None:
            self.viewer.close()