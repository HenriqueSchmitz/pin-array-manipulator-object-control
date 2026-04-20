from control.ball_cup import SphericalCradlePolicy
from control.control_policy import EnvironmentState
from control.simple_translation import SimpleTranslationControlPolicy
from environment import SimulationEnvironment
from objects.ball import Ball
from objects.object import Pose
from manipulator.pin_array_manipulator import PinArrayManipulator
from control.wave import SineWaveControlPolicy

MANIPULATOR_SIZE = 1
PINS_PER_SIDE = 15
PIN_HEIGHT = 0.15
ACTUATION_LENGTH = 0.1
PIN_SPACING = 0.001

BALL_DIAMETER = 0.1

target_pose = Pose(0.1, -0.2, 0)

manipulator = PinArrayManipulator(manipulator_size=MANIPULATOR_SIZE,
                                  pins_per_side=PINS_PER_SIDE,
                                  pin_height=PIN_HEIGHT,
                                  actuation_length=ACTUATION_LENGTH,
                                  pin_spacing=PIN_SPACING,
                                  has_wall=True)
object = Ball(diameter=BALL_DIAMETER, starting_z=0.2)
control_policy = SineWaveControlPolicy(pins_per_side=PINS_PER_SIDE)
# control_policy = SimpleTranslationControlPolicy(manipulator)
# control_policy = SphericalCradlePolicy(manipulator, ball_diameter=BALL_DIAMETER)
environment = SimulationEnvironment(manipulator, [object])

def control_logic():
    state = EnvironmentState(object.get_pose(), target_pose)
    control_policy.update(state)
    control_tensor = control_policy.get_control_tensor()
    manipulator.actuate_from_tensor_percentage(control_tensor)

environment.run(control_logic)