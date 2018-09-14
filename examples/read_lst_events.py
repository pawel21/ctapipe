import matplotlib.pyplot as plt
import numpy as np
from ctapipe.io.lsteventsource import LSTEventSource
from ctapipe.io import EventSeeker
from ctapipe.calib.camera import DragonPedestal


def get_data_from_module(nr, ev):
    first_cap = ev.lst.tel[0].evt.first_capacitor_id[i * 8:(i + 1) * 8]

reader = LSTEventSource(input_url="/home/pawel1/Pulpit/Astrophysics/CTA/dragon/data/dane_lst/20180629/Date20180629.0000.fits.fz")

seeker = EventSeeker(reader)
ev = seeker[10]

hg = 0
lg = 1
fc = np.zeros((2, 8))

num_modules = ev.lst.tel[0].svc.num_modules

for i in range(0, num_modules):
    print("module id = ", i, end="\t")
    print("fc = ", ev.lst.tel[0].evt.first_capacitor_id[i*8:(i+1)*8], end="\n")
    first_cap = ev.lst.tel[0].evt.first_capacitor_id[i*8:(i+1)*8]

    for channel_hg in [0, 2, 4, 6]:
        fc[hg, channel_hg] = first_cap[channel_hg]
        fc[hg, channel_hg+1] = first_cap[channel_hg]
    for channel_lg in [1, 3, 5, 7]:
        fc[lg, channel_lg-1] = first_cap[channel_lg]
        fc[lg, channel_lg] = first_cap[channel_lg]
    print("fc hg:", fc[hg, :])
    print("fc lg: ", fc[lg, :])

    print("waveform = ", ev.r0.tel[0].waveform[hg, i, :])


print(ev.lst.tel[0].evt.first_capacitor_id)
print(ev.lst.tel[0].evt.first_capacitor_id)
fig, ax0 = plt.subplots()
ax0.plot(ev.r0.tel[0].waveform[1, 35, :])

plt.show()