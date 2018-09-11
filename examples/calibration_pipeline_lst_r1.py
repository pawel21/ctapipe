import numpy as np
import matplotlib.pyplot as plt
from traitlets import Dict, List
from ctapipe.calib import CameraR1CalibratorFactory
from ctapipe.core import Tool
from ctapipe.io.lsteventsource import LSTEventSource
from ctapipe.io import EventSeeker

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

    def start(self):
        list_file = ['/home/pawel1/Pulpit/Astrophysics/CTA/dragon/data/dane_lst/20180705/Date20180705.0010.fits.fz',
                     '/home/pawel1/Pulpit/Astrophysics/CTA/dragon/data/dane_lst/20180629/Date20180629.0001.fits.fz']
        inputfile_reader_event0 = LSTEventSource(input_url = list_file[1], max_events = 15)

        event_array_r0 = np.zeros((15, 2, 1813, 40)) #1834 for 20180705
        event_array_r1 = np.zeros((15, 2, 1813, 40))

        for i, ev in enumerate(inputfile_reader_event0):
            nr = 25  # number of module
            pixel = 3
            #time = np.linspace(2, 37, 36)

            event_array_r0[i, :, :, :] = ev.r0.tel[0].waveform[:, :, :]

            #plt.step(time, ev.r0.tel[0].waveform[1, nr * 7:(nr + 1) * 7, 2:38][pixel], label="before remove pedestal")

            self.r1_calibrator.calibrate(ev)

            event_array_r1[i, :, :, :] = ev.r1.tel[0].waveform[:, :, :]

            #plt.step(time, ev.r1.tel[0].waveform[1, nr * 7:(nr + 1) * 7, 2:38][pixel], 'r-', label="after remove pedestal")

            #plt.plot([0, 40], [300, 300], 'g--', label="offset")
            #plt.ylim([200, 400])
            #plt.legend()
            #plt.grid(True)
            #plt.show()

        plt.figure()
        plt.hist(event_array_r0[:, 0, nr * 7:(nr + 1) * 7, :][pixel].ravel(), bins=50, color='blue', label="r0")
        print("r0 std: ", np.std(event_array_r0[:, 0, nr * 7:(nr + 1) * 7, :][pixel].ravel()))
        plt.hist(event_array_r1[:, 0, nr * 7:(nr + 1) * 7, :][pixel].ravel(), bins=50, color='green', alpha=0.7, label="r1")
        print("r1 std: ", np.std(event_array_r1[:, 0, nr * 7:(nr + 1) * 7, :][pixel].ravel()))
        plt.grid(True)
        plt.legend()

        plt.show()

    def finish(self):
        pass

if __name__ == '__main__':
    exe = R1Tester()
    exe.run()