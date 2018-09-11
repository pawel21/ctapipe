import numpy as np
import os
import struct
from ctapipe.io.lsteventsource import LSTEventSource
from ctapipe.io import EventSeeker
from ctapipe.calib.camera import DragonPedestal


def write_pedestal_to_file(PedList, number_modules):
    f_out = open('pedestal_value20180629.dat', 'wb')
    # header
    f_out.write(struct.pack('>B', 1))  # version 1
    f_out.write(struct.pack('>H', 7))  # number of pixels
    f_out.write(struct.pack('>H', 4096))  # number of samples
    f_out.write(struct.pack('>H', 40))  # RoI
    f_out.write(struct.pack('>H', number_modules))  # number of modules

    for nr in range(0, number_modules):
        # high gain
        for pixel in range(0, 7):
            pedestal_value = (PedList[nr].meanped[0, pixel, :])
            for value in (pedestal_value):
                if np.isnan(value):
                    value = 0
                    f_out.write(struct.pack('>H', int(value)))
                else:
                    f_out.write(struct.pack('>H', int(value)))
        # low gain
        for pixel in range(0, 7):
            pedestal_value = (PedList[nr].meanped[1, pixel, :])
            for value in (pedestal_value):
                if np.isnan(value):
                    value = 0
                    f_out.write(struct.pack('>H', int(value)))
                else:
                    f_out.write(struct.pack('>H', int(value)))

    f_out.close()


#path_to_data = os.path.join("/home", "pawel1", "Pulpit", "Astrophysics", "CTA", "dragon",
#                            "data", "dane_lst", "20180705", "Date20180705.0000.fits.fz")
path_to_data =  '/home/pawel1/Pulpit/Astrophysics/CTA/dragon/data/dane_lst/20180629/Date20180629.0000.fits.fz'
reader = LSTEventSource(input_url=path_to_data)
seeker = EventSeeker(reader)
ev = seeker[0]

number_modules = ev.lst.tel[0].svc.num_modules
ped = DragonPedestal()
PedList = []


for i in range(0, number_modules):
    print("nr: ", i)
    reader = LSTEventSource(input_url=path_to_data)
    seeker = EventSeeker(reader)
    PedList.append(DragonPedestal())
    for j in range(0, 5000):
        ev = seeker[j]
        PedList[i].fill_pedestal_event(ev, nr=i)
    PedList[i].finalize_pedestal()

write_pedestal_to_file(PedList, number_modules)
