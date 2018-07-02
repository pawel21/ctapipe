import numpy as np

from ctapipe.core import Component, Factory, traits

__all__ = ["Dragon", "DragonPedestal"]


class Dragon():
    """
    Base class fot calibration dragon
    """

    def __init__(self):
        pass


class DragonPedestal:
    n_channels = 8
    RoI = 40
    size4drs = 4*1024
    high_gain = 0
    low_gain = 1

    def __init__(self):
        self.pedestal_value = np.zeros((2, 14, self.size4drs))
        self.numbers_of_events = np.zeros((2, 14, self.size4drs))
        self.first_capacitor = np.zeros((2, 8))