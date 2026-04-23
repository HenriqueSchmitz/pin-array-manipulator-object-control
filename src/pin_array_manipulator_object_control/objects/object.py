from abc import ABC, abstractmethod

import numpy as np
from mujoco import MjData # type: ignore
from scipy.spatial.transform import Rotation as R



class Translation():
    def __init__(self, x: float, y: float, z: float):
        self.x = x
        self.y = y
        self.z = z

    def array(self) -> np.ndarray:
        return np.array(self.list())
    
    @staticmethod
    def from_array(array: np.ndarray) -> 'Translation':
        return Translation(array[0], array[1], array[2])
    
    def list(self) -> list:
        return [self.x, self.y, self.z]

    def __str__(self):
        return f"({self.x}, {self.y}, {self.z})"
    
    def length(self) -> float:
        return (self.x**2 + self.y**2 + self.z**2)**0.5
    

class Size3D():
    def __init__(self, x: float, y: float, z: float):
        self.x = x
        self.y = y
        self.z = z

    def array(self) -> np.ndarray:
        return np.array(self.list())
    
    @staticmethod
    def from_array(array: np.ndarray) -> 'Size3D':
        return Size3D(array[0], array[1], array[2])
    
    def list(self) -> list:
        return [self.x, self.y, self.z]

    def __str__(self):
        return f"({self.x}, {self.y}, {self.z})"


class Pose():
    
    def __init__(self, x: float, y: float, z: float, roll: float, pitch: float, yaw: float):
        self.x, self.y, self.z = x, y, z
        self.roll, self.pitch, self.yaw = roll, pitch, yaw

    def array(self) -> np.ndarray:
        return np.array(self.list())
    
    @staticmethod
    def from_array(array: np.ndarray) -> 'Pose':
        return Pose(array[0], array[1], array[2], array[3], array[4], array[5])
    
    def list(self) -> list:
        return [self.x, self.y, self.z, self.roll, self.pitch, self.yaw]

    def __str__(self):
        return (f"Pos: (x:{self.x:.2f}, y:{self.y:.2f}, z:{self.z:.2f}) "
                f"Rot: (r:{self.roll:.1f}, p:{self.pitch:.1f}, y:{self.yaw:.1f})")
    
    def translation_to(self, other) -> Translation:
        return Translation(other.x - self.x, other.y - self.y, other.z - self.z)
    

class Velocity():
    def __init__(self, lin_x: float, lin_y: float, lin_z: float, ang_x: float, ang_y: float, ang_z: float):
        self.x = lin_x
        self.y = lin_y
        self.z = lin_z
        self.roll = ang_x
        self.pitch = ang_y
        self.yaw = ang_z

    def array(self) -> np.ndarray:
        return np.array(self.list())
    
    @staticmethod
    def from_array(array: np.ndarray) -> 'Velocity':
        return Velocity(array[0], array[1], array[2], array[3], array[4], array[5])
    
    def list(self) -> list:
        return [self.x, self.y, self.z, self.roll, self.pitch, self.yaw]

    def __str__(self):
        return (f"Linear: (x:{self.x:.2f}, y:{self.y:.2f}, z:{self.z:.2f}) "
                f"Angular: (r:{self.roll:.2f}, p:{self.pitch:.2f}, y:{self.yaw:.2f})")
    

class Object(ABC):
    def __init__(self, name: str = "object"):
        self.name = name
        self.data = None
    
    @abstractmethod
    def generate_bodies(self):
        raise NotImplementedError
    
    @abstractmethod
    def generate_visual_body(self, name: str) -> str:
        raise NotImplementedError
    
    def set_data(self, data: MjData):
        self.data = data

    @abstractmethod
    def get_size(self) -> Size3D:
        raise NotImplementedError
    
    def get_pose(self) -> Pose:
        if not self.data:
            raise Exception("Data not set")
        obj = self.data.body(self.name)
        pos = obj.xpos
        quat = obj.xquat
        r = R.from_quat([quat[1], quat[2], quat[3], quat[0]])
        roll, pitch, yaw = r.as_euler('xyz', degrees=True)
        return Pose(pos[0], pos[1], pos[2], roll, pitch, yaw)
    
    def get_velocity(self) -> Velocity:
        if not self.data:
            raise Exception("Data not set")
        object_info = self.data.body(self.name)
        v = object_info.cvel
        return Velocity(
            lin_x=v[3], lin_y=v[4], lin_z=v[5],
            ang_x=v[0], ang_y=v[1], ang_z=v[2]
        )
    
    @abstractmethod
    def generate_assets(self) -> str:
        raise NotImplementedError