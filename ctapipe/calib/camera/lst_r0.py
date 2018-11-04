import numpy as np
from ...core import Component, Factory
from ...core.traits import Unicode
from ...io import EventSource

from numba import jit, njit, prange

from ctapipe.core import Component
from ctapipe.core.traits import Unicode

import matplotlib.pyplot as plt

class CameraR0Calibrator(Component):
    """
    The base R0-level calibrator. Fills the r0 container.
    The R0 calibrator performs the camera-specific R0 calibration that is
    usually performed on the raw data by the camera server. This calibrator
    exists in ctapipe for testing and prototyping purposes.
    Parameters
    ----------
    config : traitlets.loader.Config
        Configuration specified by config file or cmdline arguments.
        Used to set traitlet values.
        Set to None if no configuration to pass.
    tool : ctapipe.core.Tool or None
        Tool executable that is calling this component.
        Passes the correct logger to the component.
        Set to None if no Tool to pass.
    kwargs
    """

    def __init__(self, config=None, tool=None, **kwargs):
        """
        Parent class for the r0 calibrators. Fills the r0 container.
        Parameters
        ----------
        config : traitlets.loader.Config
            Configuration specified by config file or cmdline arguments.
            Used to set traitlet values.
            Set to None if no configuration to pass.
        tool : ctapipe.core.Tool or None
            Tool executable that is calling this component.
            Passes the correct logger to the component.
            Set to None if no Tool to pass.
        kwargs
        """
        super().__init__(config=config, parent=tool, **kwargs)
        self._r0_empty_warn = False


    def check_r0_exists(self, event, telid):
        """
        Check that r0 data exists. If it does not, then do not change r0.
        This ensures that if the containers were filled from a file containing
        r0 data, it is not overwritten by non-existant data.
        Parameters
        ----------
        event : container
            A `ctapipe` event container
        telid : int
            The telescope id.
        Returns
        -------
        bool
            True if r0.tel[telid].waveform is not None, else false.
        """
        r0 = event.r0.tel[telid].waveform
        if r0 is not None:
            return True
        else:
            if not self._r0_empty_warn:
                self.log.warning("Encountered an event with no R0 data. ")
                self._r0_empty_warn = True
        return False


class LSTR0Corrections(CameraR0Calibrator):

    pedestal_path = Unicode(
        '',
        allow_none=True,
        help='Path to the LST pedestal binary file'
    ).tag(config=True)

    def __init__(self, config=None, tool=None, **kwargs):
        """
        The R0 calibrator for LST data.
        Fills the r0 container.
        Parameters
        ----------
        config : traitlets.loader.Config
            Configuration specified by config file or cmdline arguments.
            Used to set traitlet values.
            Set to None if no configuration to pass.
        tool : ctapipe.core.Tool
            Tool executable that is calling this component.
            Passes the correct logger to the component.
            Set to None if no Tool to pass.
        kwargs
        """
        super().__init__(config=config, tool=tool, **kwargs)
        self.telid = 0
        self.pedestal_value_array = None
        self.n_pixels = 7
        self.size4drs = 4 * 1024
        self.roisize = 40
        self.offset = 300
        self.high_gain = 0
        self.low_gain = 1

        self._load_calib()

        self.offset = 300

        self.first_cap_array = np.zeros((265, 2, 7))
        self.first_cap_array_spike_A = np.zeros((265, 2, 7))
        self.first_cap_old_array = np.zeros((265, 2, 7))

        self.last_time_array = np.zeros((265, 2, 7, 4096))

    def subtract_pedestal(self, event):
        number_of_modules = event.lst.tel[0].svc.num_modules
        for nr_module in range(0, number_of_modules):
            self.first_cap_array[nr_module, :, :] = self._get_first_capacitor(event, nr_module)

        event.r0.tel[self.telid].waveform[:, :, :] = self.calibrate_jit(event.r0.tel[self.telid].waveform,
                                                                        self.first_cap_array,
                                                                        self.pedestal_value_array,
                                                                        number_of_modules)


    def interpolate_spike_A(self, event):
        self.first_cap_old_array[:, :, :] = self.first_cap_array_spike_A[:, :, :]

        number_of_modules = event.lst.tel[0].svc.num_modules

        waveform = event.r0.tel[0].waveform[:, : , :]

        for nr_clus in range(0, number_of_modules):
            self.first_cap_array_spike_A[nr_clus, :, :] = self._get_first_capacitor(event, nr_clus)

        wf = waveform.copy()
        wf = wf.astype('int16')
        event.r0.tel[0].waveform = self.interpolate_pseudo_pulses(wf, self.first_cap_array_spike_A,
                                                                  self.first_cap_old_array, number_of_modules)


        #    for gain in range(0, 2):
        #        for pixel in range(0, 7):
        #            for k in range(0, 4):
        #                # looking for spike A first case
        #                abspos = int(1024 - self.roisize - 2 - self.first_cap_old_array[nr_clus, gain, pixel] + k*1024 + self.size4drs)
        #                pos = int((abspos - self.first_cap_array_spike_A[nr_clus, gain, pixel] + self.size4drs) % self.size4drs)
        #                if (pos > 2 and pos < 38):
        #                    self.inter_spike_A(event, gain, pos, pixel, nr_clus)

        #                # looking for spike A second case
        #                abspos = int(self.roisize - 2 + self.first_cap_old_array[nr_clus, gain, pixel] + k * 1024 + self.size4drs)
        #                pos = int((abspos - self.first_cap_array_spike_A[nr_clus, gain, pixel] + self.size4drs) % self.size4drs)
        #                if (pos > 2 and pos < 38):
        #                    self.inter_spike_A(event, gain, pos, pixel, nr_clus)

    @staticmethod
    @njit(parallel=True)
    def interpolate_pseudo_pulses(waveform, fc, fc_old, number_of_modules):
        roisize = 40
        size4drs = 4096
        #waveform = waveform.astype('int16')
        for nr_clus in prange(0, number_of_modules):
            for gain in prange(0, 2):
                 for pixel in prange(0, 7):
                    for k in prange(0, 4):
                        # looking for spike A first case
                        abspos = int(1024 - roisize - 2 - fc_old[nr_clus, gain, pixel] + k*1024 + size4drs)
                        pos = int((abspos - fc[nr_clus, gain, pixel] + size4drs) % size4drs)
                        if (pos > 2 and pos < 38):
                            fun_to_interpolate_spike_A(waveform, gain, pos, pixel, nr_clus)
                        abspos = int(roisize - 2 + fc_old[nr_clus, gain, pixel] + k * 1024 + size4drs)
                        pos = int((abspos - fc[nr_clus, gain, pixel] + size4drs) % size4drs)
                        if (pos > 2 and pos < 38):
                            fun_to_interpolate_spike_A(waveform, gain, pos, pixel, nr_clus)


        return waveform

    def inter_spike_A(self, event, gain, pos, pixel, nr_clus):
        samples = event.r0.tel[0].waveform[gain, pixel + nr_clus * 7, :]
        a = int(samples[pos - 1])
        b = int(samples[pos + 2])
        value1 = samples[pos - 1] + (0.33 * (b - a))
        value2 = samples[pos - 1] + (0.66 * (b - a))
        event.r0.tel[0].waveform[gain, pixel + nr_clus * 7, pos] = value1
        event.r0.tel[0].waveform[gain, pixel + nr_clus * 7, pos + 1] = value2

    @staticmethod
    @jit(parallel=True)
    def calibrate_jit(event_waveform, fc_cap, pedestal_value_array, nr_clus):
        ev_waveform = np.zeros(event_waveform.shape)
        size4drs = 4096
        for nr in prange(0, nr_clus):
            for gain in prange(0, 2):
                for pixel in prange(0, 7):
                    position = int((fc_cap[nr, gain, pixel]) % size4drs)
                    ev_waveform[gain, pixel + nr * 7, :] = \
                        (event_waveform[gain, pixel + nr * 7, :] -
                         pedestal_value_array[nr, gain, pixel, position:position + 40])
        return ev_waveform

    def _load_calib(self):
        """
        If a pedestal file has been supplied, create a array with
        pedestal value . If it hasn't then point calibrate to
        fake_calibrate, where nothing is done to the waveform.
        """

        if self.pedestal_path:
            with open(self.pedestal_path, "rb") as binary_file:
                data = binary_file.read()
                file_version = int.from_bytes(data[0:1], byteorder='big')
                self.number_of_clusters_from_file = int.from_bytes(data[7:9],
                                                                   byteorder='big')
                self.pedestal_value_array = np.zeros((self.number_of_clusters_from_file, 2,
                                                      self.n_pixels, self.size4drs + 40))
                self.log.info("Load binary file with pedestal version {}: {} ".format(
                    file_version, self.pedestal_path))
                self.log.info("Number of modules in file: {}".format(
                    self.number_of_clusters_from_file))

                start_byte = 9
                for i in range(0, self.number_of_clusters_from_file):
                    for gain in range(0, 2):
                        for pixel in range(0, self.n_pixels):
                            for cap in range(0, self.size4drs):
                                value = int.from_bytes(data[start_byte:start_byte + 2],
                                                       byteorder='big') - self.offset
                                self.pedestal_value_array[i, gain, pixel, cap] = value
                                start_byte += 2
                            self.pedestal_value_array[i, gain, pixel, self.size4drs:self.size4drs+40] = self.pedestal_value_array[i, gain, pixel, 0:40]
        else:
            self.log.warning("No pedestal path supplied, "
                             "r0 samples will equal r0 samples.")
            self.calibrate = self.fake_calibrate

    def _get_first_capacitor(self, event, nr_module):
        """
        Get first capacitor values from event for nr module.
        Parameters
        ----------
        event : `ctapipe` event-container
        nr_module : number of module
        """
        fc = np.zeros((2, 7))
        first_cap = event.lst.tel[0].evt.first_capacitor_id[nr_module * 8:
                                                            (nr_module + 1) * 8]
        for i, j in zip([0, 1, 2, 3, 4, 5, 6], [0, 0, 1, 1, 2, 2, 3]):
            fc[self.high_gain, i] = first_cap[j]
        for i, j in zip([0, 1, 2, 3, 4, 5, 6], [4, 4, 5, 5, 6, 6, 7]):
            fc[self.low_gain, i] = first_cap[j]
        return fc

    def fake_calibrate(self, event):
        """
        Don't perform any calibration on the waveforms, just fill the
        R1 container.
        Parameters
        ----------
        event : `ctapipe` event-container
        """

        for telid in event.r0.tels_with_data:
            if self.check_r0_exists(event, telid):
                samples = event.r0.tel[telid].waveform
                event.r0.tel[telid].waveform = samples.astype('uint16')

@jit
def fun_to_interpolate_spike_A(waveform, gain, pos, pixel, nr_clus):
    samples = waveform[gain, pixel + nr_clus * 7, :]
    a = int(samples[pos - 1])
    b = int(samples[pos + 2])
    value1 = (samples[pos - 1]) + (0.33 * (b - a))
    value2 = (samples[pos - 1]) + (0.66 * (b - a))
    waveform[gain, pixel + nr_clus * 7, pos] = value1
    waveform[gain, pixel + nr_clus * 7, pos + 1] = value2
