from pin_array_manipulator_object_control.objects.object import Object, Size3D



class Ball(Object):
    def __init__(self,
                 name: str = "ball",
                 diameter: float = 1.0,
                 starting_x: float = 0.0,
                 starting_y: float = 0.0,
                 starting_z: float = 2.0):
        super().__init__(name)
        self.diameter = diameter
        self.starting_x = starting_x
        self.starting_y = starting_y
        self.starting_z = starting_z
        self.data = None
    
    def generate_bodies(self):
        object_xml = f"""
            <body name="{self.name}" pos="{self.starting_x} {self.starting_y} {self.starting_z}">
                <joint type="free"/>
                <geom type="sphere" size="{self.diameter}" rgba="1 0 0 1" mass="0.5"/>
            </body>"""
        return object_xml
    
    def generate_visual_body(self, name: str) -> str:
        rgba: str = "1 0 0 0.3"
        return f"""
            <body name="{name}" mocap="true">
                <geom type="sphere" size="{self.diameter}" rgba="{rgba}" 
                      contype="0" conaffinity="0" group="1"/>
            </body>"""

    def get_size(self) -> Size3D:
        return Size3D(self.diameter, self.diameter, self.diameter)