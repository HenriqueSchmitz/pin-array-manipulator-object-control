from pin_array_manipulator_object_control.objects.object import Object, Size3D

class Cylinder(Object):
    def __init__(self,
                 name: str = "cylinder",
                 radius: float = 0.01,
                 length: float = 0.14,
                 starting_x: float = 0.0,
                 starting_y: float = 0.0,
                 starting_z: float = 1.0):
        super().__init__(name)
        self.radius = radius
        self.length = length
        self.starting_x = starting_x
        self.starting_y = starting_y
        self.starting_z = starting_z

    def generate_bodies(self) -> str:
        half_length = self.length / 2
        return f"""
            <body name="{self.name}" pos="{self.starting_x} {self.starting_y} {self.starting_z}">
                <joint type="free"/>
                <geom type="cylinder" size="{self.radius} {half_length}" 
                      euler="0 90 0" rgba="0.3 0.3 0.3 1" mass="0.05"/>
            </body>"""

    def generate_visual_body(self, name: str) -> str:
        half_length = self.length / 2
        return f"""
            <body name="{name}" mocap="true">
                <geom type="cylinder" size="{self.radius} {half_length}" 
                      euler="0 90 0" rgba="0.3 0.3 0.3 0.3" 
                      contype="0" conaffinity="0" group="1"/>
            </body>"""

    def get_size(self) -> Size3D:
        return Size3D(self.length, self.radius * 2, self.radius * 2)

    def generate_assets(self) -> str:
        return ""