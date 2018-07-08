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
    n_channels = 7
    roisize = 40
    size4drs = 4*1024
    high_gain = 0
    low_gain = 1

    def __init__(self):
        self.first_capacitor = np.zeros((2, 8))
        self.meanped = np.zeros((2, self.n_channels, self.size4drs))
        self.numped = np.zeros((2, self.n_channels, self.size4drs))

    def fill_pedestal_event(self, event, nr):
        first_cap = event.lst.tel[0].evt.first_capacitor_id[nr * 8:(nr + 1) * 8]
        for channel_hg in [0, 2, 4, 6]:
            self.first_capacitor[self.high_gain, channel_hg] = first_cap[channel_hg]
            self.first_capacitor[self.high_gain, channel_hg + 1] = first_cap[channel_hg]
        for channel_lg in [1, 3, 5, 7]:
            self.first_capacitor[self.low_gain, channel_lg - 1] = first_cap[channel_lg]
            self.first_capacitor[self.low_gain, channel_lg] = first_cap[channel_lg]
        waveform = event.r0.tel[0].waveform[:, nr * 7:(nr + 1) * 7, :]
        for i in range(0, 2):
            for j in range(0, self.n_channels):
                fc = int(self.first_capacitor[i, j])
                for k in range(0, self.roisize):
                    posads = int((k+fc)%self.size4drs)
                    val = waveform[i, j, k]
                    self.meanped[i, j, posads] += val
                    self.numped[i, j, posads] += 1

    def finalize_pedestal(self):
        try:
            self.meanped = self.meanped/self.numped
        except Exception as err:
            print(err)


def get_first_capacitor(event, nr):
    hg = 0
    lg = 1
    fc = np.zeros((2, 8))
    first_cap = event.lst.tel[0].evt.first_capacitor_id[nr * 8:(nr + 1) * 8]
    for channel_hg in [0, 2, 4, 6]:
        fc[hg, channel_hg] = first_cap[channel_hg]
        fc[hg, channel_hg + 1] = first_cap[channel_hg]
    for channel_lg in [1, 3, 5, 7]:
        fc[lg, channel_lg - 1] = first_cap[channel_lg]
        fc[lg, channel_lg] = first_cap[channel_lg]
    return fc


def remove_pedestal(event, pedestal, nr):
    first_cap = get_first_capacitor(event, nr)
    n_pixels = 7
    size4drs = 4 * 1024
    roisize = 40
    waveform = event.r0.tel[0].waveform[:, nr * 7:(nr + 1) * 7, :]
    for i in range(0, 2):
        for j in range(0, n_pixels):
            for k in range(0, roisize):
                position = int(k + first_cap[i, j])
                waveform[i, j, k] -= 1#int(pedestal.meanped[i, j, position])
