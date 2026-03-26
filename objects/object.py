from abc import ABC, abstractmethod

from mujoco import MjData # type: ignore



class Translation():
    def __init__(self, x: float, y: float, z: float):
        self.x = x
        self.y = y
        self.z = z

    def __str__(self):
        return f"({self.x}, {self.y}, {self.z})"
    
    def length(self) -> float:
        return (self.x**2 + self.y**2 + self.z**2)**0.5


class Pose():
    def __init__(self, x: float, y: float, z: float):
        self.x = x
        self.y = y
        self.z = z

    def __str__(self):
        return f"({self.x}, {self.y}, {self.z})"
    
    def translation_to(self, other) -> Translation:
        return Translation(other.x - self.x, other.y - self.y, other.z - self.z)
    

class Object(ABC):
    def __init__(self, name: str = "object"):
        self.name = name
        self.data = None
    
    @abstractmethod
    def generate_bodies(self):
        raise NotImplementedError
    
    def set_data(self, data: MjData):
        self.data = data
    
    def get_pose(self) -> Pose:
        if not self.data:
            raise Exception("Data not set")
        object_info = self.data.body(self.name)
        return Pose(object_info.xpos[0], object_info.xpos[1], object_info.xpos[2])