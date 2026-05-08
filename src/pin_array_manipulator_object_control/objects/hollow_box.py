from pin_array_manipulator_object_control.objects.object import Object, Size3D

class HollowBox(Object):
    def __init__(self,
                 name: str = "hollow_box",
                 width: float = 0.2,   # x-axis
                 length: float = 0.2,  # y-axis
                 height: float = 0.1,  # z-axis
                 thickness: float = 0.01,
                 starting_x: float = 0.0,
                 starting_y: float = 0.0,
                 starting_z: float = 1.0):
        super().__init__(name)
        self.width = width
        self.length = length
        self.height = height
        self.thickness = thickness
        self.starting_x = starting_x
        self.starting_y = starting_y
        self.starting_z = starting_z
    
    def _generate_geoms(self, rgba: str) -> str:
        hw, hl, hh = self.width / 2, self.length / 2, self.height / 2
        ht = self.thickness / 2
        
        # Positions relative to the body origin (center of the box volume)
        # Base
        base = f'<geom type="box" pos="0 0 {-hh + ht}" size="{hw} {hl} {ht}" rgba="{rgba}"/>'
        # Walls (standing on the base)
        wall_h = (self.height - self.thickness) / 2
        wall_z = ht # Centered above the base
        
        left  = f'<geom type="box" pos="{-hw + ht} 0 {wall_z}" size="{ht} {hl} {wall_h}" rgba="{rgba}"/>'
        right = f'<geom type="box" pos="{hw - ht} 0 {wall_z}" size="{ht} {hl} {wall_h}" rgba="{rgba}"/>'
        front = f'<geom type="box" pos="0 {hl - ht} {wall_z}" size="{hw - self.thickness} {ht} {wall_h}" rgba="{rgba}"/>'
        back  = f'<geom type="box" pos="0 {-hl + ht} {wall_z}" size="{hw - self.thickness} {ht} {wall_h}" rgba="{rgba}"/>'
        
        return f"{base}\n{left}\n{right}\n{front}\n{back}"

    def generate_bodies(self) -> str:
        geoms = self._generate_geoms("0.8 0.4 0.1 1")
        return f"""
            <body name="{self.name}" pos="{self.starting_x} {self.starting_y} {self.starting_z}">
                <joint type="free"/>
                {geoms}
            </body>"""
    
    def generate_visual_body(self, name: str) -> str:
        geoms = self._generate_geoms("0.8 0.4 0.1 0.3")
        return f"""
            <body name="{name}" mocap="true">
                {geoms}
            </body>"""

    def get_size(self) -> Size3D:
        return Size3D(self.width, self.length, self.height)
    
    def generate_assets(self) -> str:
        return ""