from pin_array_manipulator_object_control.objects.object import Object, Size3D

class Slab(Object):
    def __init__(self,
                 name: str = "slab",
                 width: float = 1.0,
                 length: float = 1.0,
                 thickness: float = 0.2,
                 starting_x: float = 0.0,
                 starting_y: float = 0.0,
                 starting_z: float = 1.0):
        super().__init__(name)
        self.width = width
        self.length = length
        self.thickness = thickness
        self.starting_x = starting_x
        self.starting_y = starting_y
        self.starting_z = starting_z
    
    def generate_bodies(self) -> str:
        half_x = self.width / 2
        half_y = self.length / 2
        half_z = self.thickness / 2
        object_xml = f"""
            <body name="{self.name}" pos="{self.starting_x} {self.starting_y} {self.starting_z}">
                <joint type="free"/>
                <geom type="box" size="{half_x} {half_y} {half_z}" rgba="0.2 0.5 0.8 1" mass="1.0"/>
            </body>"""
        return object_xml
    
    def generate_visual_body(self, name: str) -> str:
        half_x = self.width / 2
        half_y = self.length / 2
        half_z = self.thickness / 2
        rgba: str = "0.2 0.5 0.8 0.3"
        return f"""
            <body name="{name}" mocap="true">
                <geom type="box" size="{half_x} {half_y} {half_z}" rgba="{rgba}" 
                      contype="0" conaffinity="0" group="1"/>
            </body>"""

    def get_size(self) -> Size3D:
        return Size3D(self.width, self.length, self.thickness)
    
    def generate_assets(self) -> str:
        return ""