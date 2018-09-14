import matplotlib.pyplot as plt
import numpy as np
import os
from ctapipe.io.lsteventsource import LSTEventSource
from ctapipe.io import EventSeeker
from ctapipe.calib.camera import DragonPedestal
from ctapipe.calib.camera.dragon import remove_pedestal


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
    for i in range(0, 2):
        for j in range(0, n_pixels):
            for k in range(0, roisize):
                position = int((k + first_cap[i, j])%size4drs)
                val = event.r0.tel[0].waveform[i, nr * 7:(nr + 1) * 7, k] - int(pedestal.meanped[i, j, position])
                event.r0.tel[0].waveform[i, nr * 7:(nr + 1) * 7, k] = val

path_to_data = os.path.join("/home", "pawel1", "Pulpit", "Astrophysics", "CTA", "dragon",
                            "data", "dane_lst", "20180705", "Date20180705.0000.fits.fz")
reader = LSTEventSource(input_url=path_to_data)
seeker = EventSeeker(reader)

ped = DragonPedestal()
for i in range(0, 3500):
    ev = seeker[i]
    ped.fill_pedestal_event(ev, 1)

ped.finalize_pedestal()

for i in range(3550, 3555):
    nr = 1
    t = np.linspace(0, 39, 40)
    ev = seeker[i]
    pixel = 0
    fig, ax = plt.subplots(2, 1)
    ax[0].step(t, ev.r0.tel[0].waveform[0, nr * 7:(nr + 1) * 7, :][pixel], color="blue")
    remove_pedestal(ev, ped, 1)
    ax[1].step(t, ev.r0.tel[0].waveform[0, nr * 7:(nr + 1) * 7, :][pixel], color="red")
    plt.show()