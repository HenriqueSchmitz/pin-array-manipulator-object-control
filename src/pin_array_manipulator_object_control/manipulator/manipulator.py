from abc import ABC, abstractmethod

from pin_array_manipulator_object_control.objects.object import Object



class Manipulator(Object, ABC):
    def __init__(self, name="manipulator"):
        super().__init__(name)
        self.data = None
    
    @abstractmethod
    def generate_bodies(self):
        raise NotImplementedError
    
    @abstractmethod
    def generate_actuators(self):
        raise NotImplementedError