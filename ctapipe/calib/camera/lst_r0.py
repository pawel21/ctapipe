import numpy as np
from ...core import Component, Factory
from ...core.traits import Unicode
from ...io import EventSource

from numba import jit, njit, prange


class CameraR0Calibrator(Component):
    """
    The base R0-level calibrator. Change the r0 container.
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
        self.first_cap_array_spike = np.zeros((265, 2, 7))
        self.first_cap_old_array = np.zeros((265, 2, 7))
        self.first_cap_array_time_lapse = np.zeros((265, 2, 7))

        self.last_time_array = np.zeros((265, 2, 7, 4096))

    def subtract_pedestal(self, event):
        number_of_modules = event.lst.tel[0].svc.num_modules
        for nr_module in range(0, number_of_modules):
            self.first_cap_array[nr_module, :, :] = self._get_first_capacitor(event, nr_module)

        event.r0.tel[self.telid].waveform[:, :, :] = calibrate_jit(
            event.r0.tel[self.telid].waveform,
            self.first_cap_array,
            self.pedestal_value_array,
            number_of_modules)


    def interpolate_spike(self, event):
        self.first_cap_old_array[:, :, :] = self.first_cap_array_spike[:, :, :]
        number_of_modules = event.lst.tel[0].svc.num_modules
        for nr_clus in range(0, number_of_modules):
            self.first_cap_array_spike[nr_clus, :, :] = self._get_first_capacitor(event, nr_clus)

        waveform = event.r0.tel[0].waveform[:, :, :]
        wf = waveform.copy()
        wf = wf.astype('int16')
        event.r0.tel[0].waveform = self.interpolate_pseudo_pulses(wf, self.first_cap_array_spike,
                                                                  self.first_cap_old_array, number_of_modules)

    @staticmethod
    @njit(parallel=True)
    def interpolate_pseudo_pulses(waveform, fc, fc_old, number_of_modules):
        roisize = 40
        size4drs = 4096

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

                    spike_b_pos = int((fc_old[nr_clus, gain, pixel] - 1 - fc[nr_clus, gain, pixel] + 2*size4drs)%size4drs)
                    if spike_b_pos < roisize - 1:
                        fun_to_interpolate_spike_B(waveform, gain, spike_b_pos, pixel, nr_clus)

        return waveform

    def time_lapse_corr(self, event):
        EVB = event.lst.tel[0].evt.counters
        number_of_modules = event.lst.tel[0].svc.num_modules
        for nr_clus in range(0, number_of_modules):
            self.first_cap_array_time_lapse[nr_clus, :, :] = self._get_first_capacitor(event, nr_clus)

        do_time_lapse_corr(event.r0.tel[0].waveform, EVB,
                           self.first_cap_array_time_lapse, self.last_time_array, number_of_modules)

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


@jit(parallel=True)
def calibrate_jit(event_waveform, fc_cap, pedestal_value_array, nr_clus):
    waveform = np.zeros(event_waveform.shape)
    size4drs = 4096
    N_gain = 2
    N_pixel = 7
    for nr in prange(0, nr_clus):
        for gain in prange(0, N_gain):
            for pixel in prange(0, N_pixel):
                position = int((fc_cap[nr, gain, pixel]) % size4drs)
                waveform[gain, pixel + nr * 7, :] = \
                    (event_waveform[gain, pixel + nr * 7, :] -
                    pedestal_value_array[nr, gain, pixel, position:position + 40])
    return waveform

@jit
def fun_to_interpolate_spike_A(waveform, gain, pos, pixel, nr_clus):
    samples = waveform[gain, pixel + nr_clus * 7, :]
    a = int(samples[pos - 1])
    b = int(samples[pos + 2])
    value1 = (samples[pos - 1]) + (0.33 * (b - a))
    value2 = (samples[pos - 1]) + (0.66 * (b - a))
    waveform[gain, pixel + nr_clus * 7, pos] = value1
    waveform[gain, pixel + nr_clus * 7, pos + 1] = value2

@jit
def fun_to_interpolate_spike_B(waveform, gain, spike_b_pos, pixel, nr_clus):
    samples = waveform[gain, pixel + nr_clus * 7, :]
    value = 0.5 * (samples[spike_b_pos - 1] + samples[spike_b_pos + 1])
    waveform[gain, pixel + nr_clus * 7, spike_b_pos] = value

@jit(parallel=True)
def do_time_lapse_corr(waveform, EVB, fc, last_time_array, number_of_modules):
    size4drs = 4096
    for nr_clus in prange(0, number_of_modules):
        time_now = int64(EVB[14 + (nr_clus * 22): 22 + (nr_clus * 22)])
        for gain in prange(0, 2):
            for pixel in prange(0, 7):
                for k in prange(0, 40):
                    posads = int((k + fc[nr_clus, gain, pixel]) % size4drs)
                    if last_time_array[nr_clus, gain, pixel, posads] > 0:
                        time_diff = time_now - last_time_array[nr_clus, gain, pixel, posads]
                        val = waveform[gain, pixel + nr_clus * 7, k] - ped_time(time_diff / (133.e3))
                        waveform[gain, pixel + nr_clus * 7, k] = val
                    if (k < 39):
                        last_time_array[nr_clus, gain, pixel, posads] = time_now

@jit
def int64(x):
    return x[0] + x[1] * 256 + x[2] * 256 ** 2 + x[3] * 256 ** 3 + x[4] * 256 ** 4 + x[5] * 256 ** 5 + x[
            6] * 256 ** 6 + x[7] * 256 ** 7

@jit
def ped_time(timediff):
    return 29.3 * np.power(timediff, -0.2262) - 12.4