import matplotlib.pyplot as plt
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
    n_pixels = 7
    roisize = 40
    size4drs = 4*1024
    high_gain = 0
    low_gain = 1

    def __init__(self):
        self.first_capacitor = np.zeros((2, 8))
        self.meanped = np.zeros((2, self.n_pixels, self.size4drs))
        self.numped = np.zeros((2, self.n_pixels, self.size4drs))
        self.rms = np.zeros((2, self.n_pixels, self.size4drs))

    def fill_pedestal_event(self, event, nr):
        first_cap = event.lst.tel[0].evt.first_capacitor_id[nr * 8:(nr + 1) * 8]
        self.first_capacitor[self.high_gain, 0] = first_cap[0]
        self.first_capacitor[self.high_gain, 1] = first_cap[0]
        self.first_capacitor[self.high_gain, 2] = first_cap[1]
        self.first_capacitor[self.high_gain, 3] = first_cap[1]
        self.first_capacitor[self.high_gain, 4] = first_cap[2]
        self.first_capacitor[self.high_gain, 5] = first_cap[2]
        self.first_capacitor[self.high_gain, 6] = first_cap[3]

        self.first_capacitor[self.low_gain, 0] = first_cap[4]
        self.first_capacitor[self.low_gain, 1] = first_cap[4]
        self.first_capacitor[self.low_gain, 2] = first_cap[5]
        self.first_capacitor[self.low_gain, 3] = first_cap[5]
        self.first_capacitor[self.low_gain, 4] = first_cap[6]
        self.first_capacitor[self.low_gain, 5] = first_cap[6]
        self.first_capacitor[self.low_gain, 6] = first_cap[7]

        waveform = event.r0.tel[0].waveform[:, nr * 7:(nr + 1) * 7, :]
        for i in range(0, 2):
            for j in range(0, self.n_pixels):
                fc = int(self.first_capacitor[i, j])
                for k in range(2, self.roisize-2):
                    posads = int((k+fc)%self.size4drs)
                    val = waveform[i, j, k]
                    self.meanped[i, j, posads] += val
                    self.numped[i, j, posads] += 1
                    self.rms[i, j, posads] += val**2

    def finalize_pedestal(self):
        try:
            self.meanped = self.meanped/self.numped
            self.rms = self.rms/self.numped
            self.rms = np.sqrt(self.rms - self.meanped**2)
        except Exception as err:
            print(err)


def get_first_capacitor(event, nr):
    hg = 0
    lg = 1
    fc = np.zeros((2, 8))
    first_cap = event.lst.tel[0].evt.first_capacitor_id[nr * 8:(nr + 1) * 8]

    fc[hg, 0] = first_cap[0]
    fc[hg, 1] = first_cap[0]
    fc[hg, 2] = first_cap[1]
    fc[hg, 3] = first_cap[1]
    fc[hg, 4] = first_cap[2]
    fc[hg, 5] = first_cap[2]
    fc[hg, 6] = first_cap[3]

    fc[lg, 0] = first_cap[4]
    fc[lg, 1] = first_cap[4]
    fc[lg, 2] = first_cap[5]
    fc[lg, 3] = first_cap[5]
    fc[lg, 4] = first_cap[6]
    fc[lg, 5] = first_cap[6]
    fc[lg, 6] = first_cap[7]
    return fc


# nr - number module
def remove_pedestal(event, pedestal, nr):
    """

    Parameters
    ----------
    event: container
        A `ctapipe` event container
    """
    first_cap = get_first_capacitor(event, nr)
    hg = 0
    lg = 1
    n_pixels = 7
    size4drs = 4 * 1024
    roisize = 40
    offset = 300

    for i in range(0, 2):
        for j in range(0, n_pixels):
            for k in range(0, roisize):
                position = int((k + first_cap[i, j]) % size4drs)
                val = event.r0.tel[0].waveform[i, nr * 7:(nr + 1) * 7, k][j] - int(
                pedestal.meanped[i, j, position]) + offset
                event.r0.tel[0].waveform[i, nr * 7:(nr + 1) * 7, k][j] = val


def remove_pedestal_and_plot(ev, ped, nr, gain, pixel):
    t = np.linspace(2, 38, 36)

    fig, ax = plt.subplots()
    ax.step(t, ev.r0.tel[0].waveform[gain, nr * 7:(nr + 1) * 7, 2:38][pixel], color="blue",
            label="before remove pedestal", lw=3)
    remove_pedestal(ev, ped, nr)
    ax.step(t, ev.r0.tel[0].waveform[gain, nr * 7:(nr + 1) * 7, 2:38][pixel], color="red",
            label="after remove pedestal", lw=3, alpha=0.75)

    ax.plot([2, 38], [300, 300], 'k--', label="offset", lw=2)
    ax.set_xlabel("time sample [ns]")
    ax.set_ylabel("signal [counts]")
    ax.set_ylim([150, 400])
    ax.grid()
    ax.legend()
    plt.show()