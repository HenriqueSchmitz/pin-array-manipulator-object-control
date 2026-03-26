import time
from collections.abc import Callable

import mujoco
import mujoco.viewer
from mujoco import MjModel, MjData, mj_step # type: ignore

from manipulator.manipulator import Manipulator
from objects.object import Object



class SimulationEnvironment():
    def __init__(self,
                 manipulator: Manipulator,
                 objects: list[Object],
                 headless: bool = False,
                 time_synchronized: bool = True):
        self.manipulator = manipulator
        self.objects = objects
        self.headless = headless
        self.time_synchronized = time_synchronized
        self.viewer: mujoco.viewer.Handle = None # type: ignore
        self.reset()

    def __del__(self):
        if self.viewer is not None:
            self.viewer.close()

    def reset(self):
        if self.viewer is not None:
            self.viewer.close()
        environment_xml = self.generate_xml()
        self.model = MjModel.from_xml_string(environment_xml)
        self.data = MjData(self.model)
        self.manipulator.set_data(self.data)
        for object in self.objects:
            object.set_data(self.data)
        self.viewer: mujoco.viewer.Handle = None # type: ignore
        if not self.headless:
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)

    def generate_xml(self):
        manipulator_body_xml = self.manipulator.generate_bodies()
        manipulator_actuator_xml = self.manipulator.generate_actuators()
        object_bodies = ""
        for object in self.objects:
            object_bodies += f"\n                {object.generate_bodies()}"

        xml = f"""
        <mujoco>
            <option timestep="0.002" integrator="RK4" gravity="0 0 -9.81"/>
            
            <worldbody>
                <light diffuse=".5 .5 .5" pos="0 0 3" dir="0 0 -1"/>

                {manipulator_body_xml}
                {object_bodies}
            </worldbody>

            <actuator>
                {manipulator_actuator_xml}
            </actuator>
        </mujoco>
        """
        return xml
    
    def run(self, control_logic: Callable):
        if self.headless:
            while True:
                self.step(control_logic)
        else:
            while self.viewer.is_running():
                self.step(control_logic)
        
    def step(self, control_logic: Callable):
        step_start = time.time()
        control_logic()
        mj_step(self.model, self.data)
        if not self.headless:
            self.viewer.sync()
        self.time_synchronization(step_start)

    def time_synchronization(self, step_start: float):
        elapsed = time.time() - step_start
        if self.time_synchronized and elapsed < self.model.opt.timestep:
            time.sleep(self.model.opt.timestep - elapsed)