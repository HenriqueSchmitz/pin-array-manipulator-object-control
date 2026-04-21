import time
from typing import Any, Optional

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import mujoco
import mujoco.viewer
from mujoco import MjModel, MjData # type: ignore
from scipy.spatial.transform import Rotation as R


from pin_array_manipulator_object_control.manipulator.observation import PinArrayEnvObservation
from pin_array_manipulator_object_control.rewards.base_model import RewardModel
from pin_array_manipulator_object_control.routines.target_generator import PinArrayTargetGenerator
from pin_array_manipulator_object_control.manipulator.pin_array_manipulator import PinArrayManipulator, PinArrayManipulatorConfig
from pin_array_manipulator_object_control.objects.object import Object, Pose



class PinArrayEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}

    def __init__(self,
                 simulation_object: Object,
                 target_generator: PinArrayTargetGenerator,
                 reward_model: RewardModel,
                 manipulator_config: Optional[PinArrayManipulatorConfig] = None,
                 render_mode: Optional[str] = None,
                 max_episode_steps: int = -1):
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
        num_pins = manipulator_config.pins_per_side ** 2
        obs_shape = 18 + (2 * num_pins)
        self.observation_space = spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(obs_shape,), 
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
        self.max_episode_steps = max_episode_steps
        self._elapsed_steps = 0

    def _get_obs(self):
        target_pose = self.current_target if self.current_target is not None else Pose(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        observation = PinArrayEnvObservation(
            target_pose=target_pose,
            object_pose = self.simulation_object.get_pose(),
            object_velocity = self.simulation_object.get_velocity(),
            pin_positions=self.manipulator.get_pin_heights(),
            pin_forces=self.manipulator.get_pin_forces()
        )
        return observation
    
    def _build_info(self, observation: PinArrayEnvObservation) -> dict[str, Any]:
        target_array = self.current_target.array() if self.current_target is not None else np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        return {
            "target": target_array
        }

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed)
        self._elapsed_steps = 0
        xml = self._generate_xml()
        self.model = MjModel.from_xml_string(xml)
        self.data = MjData(self.model)
        self.manipulator.set_data(self.data)
        self.simulation_object.set_data(self.data)
        if self.render_mode == "human" and self.viewer is None:
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
        mujoco.mj_forward(self.model, self.data) # type: ignore
        observation = self._get_obs()
        self.target_generator.reset()
        self.current_target = self.target_generator.get_current_target(observation)
        self._update_target_visual()
        info = self._build_info(observation)
        return observation.array(), info

    def step(self, action):
        self._elapsed_steps += 1
        step_start = 0
        if self.render_mode == "human":
            step_start = time.perf_counter()
        self.manipulator.actuate_from_matrix(action)
        mujoco.mj_step(self.model, self.data) # type: ignore
        observation = self._get_obs()
        if self.current_target is None:
            reward = 0
            terminated = True
        else:
            reward = self.reward_model.get_reward(target_pose=self.current_target, object_pose=observation.object_pose)
            terminated = False
        truncated = self.max_episode_steps > 0 and self._elapsed_steps >= self.max_episode_steps
        self.current_target = self.target_generator.get_current_target(observation)
        info = self._build_info(observation)
        if self.render_mode == "human":
            self._update_target_visual()
            self.viewer.sync() # type: ignore
            elapsed = time.perf_counter() - step_start
            time_to_sleep = self.model.opt.timestep - elapsed # type: ignore
            if time_to_sleep > 0:
                time.sleep(time_to_sleep)
        return self._get_obs().array(), reward, terminated, truncated, info

    def _generate_xml(self):
        manip_body = self.manipulator.generate_bodies()
        manip_act = self.manipulator.generate_actuators()
        object_body = self.simulation_object.generate_bodies()
        target_visual = self.simulation_object.generate_visual_body(name="target_visualizer")
        return f"""
        <mujoco>
            <option timestep="0.002" integrator="RK4" gravity="0 0 -9.81"/>
            <worldbody>
                <light diffuse=".5 .5 .5" pos="0 0 3" dir="0 0 -1"/>
                {manip_body}
                {object_body}
                {target_visual}
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

    def _update_target_visual(self):
        """Updates the mocap body to match the current target pose."""
        if self.current_target is None or self.data is None:
            return
        target_pos = [self.current_target.x, self.current_target.y, self.current_target.z]
        self.data.mocap_pos[0] = target_pos
        r = R.from_euler('xyz', [
            self.current_target.roll, 
            self.current_target.pitch, 
            self.current_target.yaw
        ], degrees=True)
        quat = r.as_quat()
        self.data.mocap_quat[0] = [quat[3], quat[0], quat[1], quat[2]]