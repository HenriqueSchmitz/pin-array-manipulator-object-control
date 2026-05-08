from pin_array_manipulator_object_control.objects.object import Object, Size3D

class Cross3D(Object):
    def __init__(self,
                 name: str = "cross_3d",
                 arm_length: float = 0.2,    # Total span of the arms
                 thickness: float = 0.04,     # Thickness of the bars
                 starting_x: float = 0.0,
                 starting_y: float = 0.0,
                 starting_z: float = 1.0):
        super().__init__(name)
        self.arm_length = arm_length
        self.thickness = thickness
        self.starting_x = starting_x
        self.starting_y = starting_y
        self.starting_z = starting_z
    
    def _generate_geoms(self, rgba: str, is_visual: bool = False) -> str:
        coll_attr = 'contype="0" conaffinity="0" group="1"' if is_visual else ''
        
        h_len = self.arm_length / 2
        h_thick = self.thickness / 2
        mass_attr = 'mass="0.5"' if not is_visual else ''

        bar_x = f'<geom type="box" size="{h_len} {h_thick} {h_thick}" rgba="{rgba}" {coll_attr} {mass_attr}/>'
        bar_y = f'<geom type="box" size="{h_thick} {h_len} {h_thick}" rgba="{rgba}" {coll_attr} {mass_attr}/>'
        bar_z = f'<geom type="box" size="{h_thick} {h_thick} {h_len}" rgba="{rgba}" {coll_attr} {mass_attr}/>'
        
        return f"{bar_x}\n{bar_y}\n{bar_z}"

    def generate_bodies(self) -> str:
        geoms = self._generate_geoms("0.9 0.2 0.2 1", is_visual=False)
        return f"""
            <body name="{self.name}" pos="{self.starting_x} {self.starting_y} {self.starting_z}">
                <joint type="free"/>
                {geoms}
            </body>"""
    
    def generate_visual_body(self, name: str) -> str:
        geoms = self._generate_geoms("0.9 0.2 0.2 0.3", is_visual=True)
        return f"""
            <body name="{name}" mocap="true">
                {geoms}
            </body>"""

    def get_size(self) -> Size3D:
        # The bounding box is equal to the arm length in all directions
        return Size3D(self.arm_length, self.arm_length, self.arm_length)
    
    def generate_assets(self) -> str:
        return ""