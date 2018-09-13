import numpy as np
import matplotlib.pyplot as plt
from traitlets import Dict, List
from ctapipe.calib import CameraR1CalibratorFactory
from ctapipe.core import Tool
from ctapipe.io.lsteventsource import LSTEventSource
from ctapipe.io import EventSeeker

from ctapipe.core import Provenance
from ctapipe.utils import json2fits
from pprint import pprint

class R1Tester(Tool):
    name = "R1Tester"
    description = "Test R1 Calibrator"

    aliases = Dict(
        dict(
            r1='CameraR1CalibratorFactory.product',
            pedestal_path='CameraR1CalibratorFactory.pedestal_path',
        )
    )
    classes = List([
        CameraR1CalibratorFactory,
    ])

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.eventsource = None
        self.r1_calibrator = None
        self.plotter = None

    def setup(self):
        kwargs = dict(config=self.config, tool=self)

        self.r1_calibrator = CameraR1CalibratorFactory.produce(
            eventsource=None, **kwargs
        )

        self.r1_calibrator.roisize = 40


    def start(self):
        list_file = ['/home/pawel1/Pulpit/Astrophysics/CTA/dragon/data/dane_lst/20180705/Date20180705.0010.fits.fz',
                     '/home/pawel1/Pulpit/Astrophysics/CTA/dragon/data/dane_lst/20180629/Date20180629.0001.fits.fz']
        max_events = 5
        inputfile_reader_event0 = LSTEventSource(input_url = list_file[0], max_events = max_events)

        event_array_r0 = np.zeros((max_events, 2, 1834, 40)) #1834 for 20180705; 1813 for 20180629
        event_array_r1 = np.zeros((max_events, 2, 1834, 40))

        nr = 0  # number of module
        pixel = 5

        for i, ev in enumerate(inputfile_reader_event0):
            event_array_r0[i, :, :, :] = ev.r0.tel[0].waveform[:, :, :]
            self.r1_calibrator.calibrate(ev)
            event_array_r1[i, :, :, :] = ev.r1.tel[0].waveform[:, :, :]

            #self.plot_waveform(ev.r0.tel[0].waveform, ev.r1.tel[0].waveform, 0, nr, pixel)

        plt.figure()
        plt.hist(event_array_r0[0:max_events, 0, pixel+nr*7, 2:38].ravel(), bins=50, color='blue', label="r0")
        print("r0 std: ", np.std(event_array_r0[:, 0, pixel+nr*7, 2:38].ravel()))
        plt.hist(event_array_r1[0:max_events, 0, pixel+nr*7, 2:38].ravel(), bins=50, color='green', alpha=0.7, label="r1")
        print("r1 std: ", np.std(event_array_r1[:, 0, pixel+nr*7, 2:38].ravel()))
        plt.grid(True)
        plt.legend()

        plt.show()


    def plot_waveform(self, event_r0, event_r1, gain, nr, pixel):
        time = np.linspace(2, 37, 36)
        plt.step(time, event_r0[gain, nr * 7:(nr + 1) * 7, 2:38][pixel], label="before remove pedestal")
        plt.step(time, event_r1[gain, nr * 7:(nr + 1) * 7, 2:38][pixel], 'r-', label="after remove pedestal")
        plt.plot([0, 40], [300, 300], 'g--', label="offset")
        plt.ylim([200, 400])
        plt.legend()
        plt.grid(True)
        plt.show()

    def finish(self):
        pass
#        p = Provenance()  # note this is a singleton, so only ever one global provenence object
#        p.clear()
#        p.start_activity()
#        p.add_input_file(self.r1_calibrator.pedestal_path)
#        p.finish_activity()
#        print(p.as_json(indent=1))

if __name__ == '__main__':
    exe = R1Tester()
    exe.run()