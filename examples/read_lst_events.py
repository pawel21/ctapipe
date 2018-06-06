from ctapipe.io.lsteventsource import LSTEventSource
from ctapipe.io import EventSeeker


reader = LSTEventSource(input_url="/home/pawel1/Pulpit/Astrophysics/CTA/dragon/data/Run009_000.fits.fz")


seeker = EventSeeker(reader)
ev = seeker[5]

print(ev.lst.tel[0].evt.first_capacitor_id)