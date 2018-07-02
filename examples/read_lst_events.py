import matplotlib.pyplot as plt
from ctapipe.io.lsteventsource import LSTEventSource
from ctapipe.io import EventSeeker


reader = LSTEventSource(input_url="/home/pawel1/Pulpit/Astrophysics/CTA/dragon/data/dane_lst/20180610/Run0038.0000.fits.fz")


seeker = EventSeeker(reader)
ev = seeker[10]

num_modules = ev.lst.tel[0].svc.num_modules

for i in range(0, num_modules):
    print("i = ", i, end="\t")
    print("fc = ", ev.lst.tel[0].evt.first_capacitor_id[i*8:(i+1)*8], end="\t")
    print("waveform = ", ev.r0.tel[0].waveform[0, i, :])

print(ev.lst.tel[0].evt.first_capacitor_id)
print(ev.lst.tel[0].evt.first_capacitor_id)
fig, ax0 = plt.subplots()
ax0.plot(ev.r0.tel[0].waveform[1, 35, :])

plt.show()