import matplotlib.pyplot as plt
import numpy as np

from ctapipe.core import Component, Factory, traits

__all__ = ["DragonPedestal", "DRS4Pedestal", "remove_pedestal_from_file"]


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

        for i, j in zip([0, 1, 2, 3, 4, 5, 6], [0, 0, 1, 1, 2, 2, 3]):
            self.first_capacitor[self.high_gain, i] = first_cap[j]

        for i, j in zip([0, 1, 2, 3, 4, 5, 6], [4, 4, 5, 5, 6, 6, 7]):
            self.first_capacitor[self.low_gain, i] = first_cap[j]

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
    for i, j in zip([0, 1, 2, 3, 4, 5, 6], [0, 0, 1, 1, 2, 2, 3]):
        fc[hg, i] = first_cap[j]
    for i, j in zip([0, 1, 2, 3, 4, 5, 6], [4, 4, 5, 5, 6, 6, 7]):
        fc[lg, i] = first_cap[j]
    return fc


def remove_pedestal(event, pedestal, nr):
    """

    Parameters
    ----------
    event: container
        A `ctapipe` event container
    pedestal:
        pedestal object
    nr: int
        number of module
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


class DRS4Pedestal():
    def __init__(self, path_to_file, nr):
        self.path = path_to_file
        self.nr = nr
        self.pedestal_value = np.zeros((self.nr, 2, 7, 4096))

    def read_binary_file(self):
        with open(self.path, "rb") as binary_file:
            # Read the whole file at once
            data = binary_file.read()

            pos = 7
            for i in range(0, self.nr):
                for gain in range(0, 2):
                    for pixel in range(0, 7):
                        for cap in range(0, 4096):
                            value = int.from_bytes(data[pos:pos + 2], byteorder='big')
                            self.pedestal_value[i, gain, pixel, cap] = value
                            pos += 2


def remove_pedestal_from_file(event, pedestal):
    waveform_after_remove_pedestal = np.zeros((event.r0.tel[0].waveform.shape))
    n_pixels = 7
    size4drs = 4 * 1024
    roisize = 40
    offset = 300
    number_of_modules = event.lst.tel[0].svc.num_modules
    for nr in range(0, number_of_modules):
        first_cap = get_first_capacitor(event, nr)
        for i in range(0, 2):
            for j in range(0, n_pixels):
                for k in range(0, roisize):
                    position = int((k + first_cap[i, j]) % size4drs)
                    val = (event.r0.tel[0].waveform[i, nr * 7:(nr + 1) * 7, k][j] - int(
                        pedestal.pedestal_value[nr, i, j, position])) + offset
                    waveform_after_remove_pedestal[i, nr * 7:(nr + 1) * 7, k][j] = val

    return waveform_after_remove_pedestal
