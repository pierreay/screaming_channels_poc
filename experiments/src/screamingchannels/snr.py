#!/usr/bin/python3


import click
import collections
import enum
import json
import os
from os import path, system
import serial
import sys
import time
import logging
from Crypto.Cipher import AES
import zmq
import subprocess

from gnuradio import blocks, gr, uhd, iio
import osmosdr

import numpy as np


logging.basicConfig()
l = logging.getLogger('reproduce')

Radio = enum.Enum("Radio", "USRP USRP_mini USRP_B210 USRP_B210_MIMO HackRF bladeRF PlutoSDR")


# Global settings, for simplicity
DEVICE = None
BAUD = None
OUTFILE = None
RADIO = None
RADIO_ADDRESS = None
COMMUNICATE_SLOW = None
YKUSH_PORT = None


class EnumType(click.Choice):
    """Teach click how to handle enums."""
    def __init__(self, enumcls):
        self._enumcls = enumcls
        click.Choice.__init__(self, enumcls.__members__)

    def convert(self, value, param, ctx):
        value = click.Choice.convert(self, value, param, ctx)
        return self._enumcls[value]


@click.group()
@click.option("-d", "--device", default="/dev/ttyACM0", show_default=True,
              help="The serial dev path of device tested for screaming channels")
@click.option("-b", "--baudrate", default=115200, show_default=True,
              help="The baudrate of the serial device")
@click.option("-y", "--ykush-port", default=0, show_default=True,
              help="If set, use given ykush-port to power-cycle the device")
@click.option("-s", "--slowmode", is_flag=True, show_default=True,
              help=("Enables slow communication mode for targets with a small"
                    "serial rx-buffer"))
@click.option("-r", "--radio", default="USRP", type=EnumType(Radio), show_default=True,
              help="The type of SDR to use.")
@click.option("--radio-address", default="10.0.3.40",
              help="Address of the radio (X.X.X.X for USRP, ip:X.X.X.X or usb:X.X.X for PlutoSDR).")
@click.option("-l", "--loglevel", default="INFO", show_default=True,
              help="The loglevel to be used ([DEBUG|INFO|WARNING|ERROR|CRITICAL])")
@click.option("-o", "--outfile", default="/tmp/time", type=click.Path(), show_default=True,
              help="The file to write the GNUradio trace to.")
def cli(device, baudrate, ykush_port, slowmode, radio, radio_address,
        outfile, loglevel, **kwargs):
    """
    Reproduce screaming channel experiments with vulnerable devices.

    This script assumes that the device has just been plugged in (or is in an
    equivalent state), that it is running our modified firmware, and that an SDR
    is available. It will carry out the chosen experiment, producing a trace and
    possibly other artifacts. Make sure that the "--outfile" option points to a
    file that doesn't exist or can be safely overwritten.

    Call any experiment with "--help" for details. You most likely want to use
    "collect".
    """
    global DEVICE, OUTFILE, RADIO, RADIO_ADDRESS, BAUD, COMMUNICATE_SLOW, YKUSH_PORT
    DEVICE = device
    BAUD = baudrate
    OUTFILE = outfile
    RADIO = radio
    RADIO_ADDRESS = radio_address
    COMMUNICATE_SLOW = slowmode
    YKUSH_PORT = ykush_port

    l.setLevel(loglevel)


def _plot_outfile():
    """
    Plot the recorded data.
    """
    from matplotlib import pyplot as plt
    import scipy

    with open(OUTFILE) as f:
        data = scipy.fromfile(f, dtype=scipy.complex64)

    plt.plot(np.absolute(data))
    plt.show()

def _encode_for_device(data):
    """
    Encode the given bytes in our special format.
    """
    return " ".join(str(data_byte) for data_byte in data)


def _send_parameter(ser, command, param):
    """
    Send a parameter (key or plaintext) to the target device.

    The function assumes that we've already entered tiny_aes mode.
    """
    command_line = '%s%s\r\n' % (command, _encode_for_device(param))
    l.debug('Sending command:  %s\n' % command_line)
    if not COMMUNICATE_SLOW:
        ser.write(command_line.encode())
    else:
        for p in command_line.split(' '):
            ser.write((p+' ').encode())
            time.sleep(.05)

    l.debug('Waiting check\n')
    x = ser.readline()
    print ("received: "+x.decode())
    if len(x) == 0:
        print("nothing received on timeout, ignoring error")
        return 
    #check = ''.join(chr(int(word)) for word in x.split(' '))
    # -- create check like this instead for ESP32:
    #response = ser.readline()
    #response = [ a for a in x.decode().split(' ') if a.isdigit() ]
    #check = ''.join(chr(int(word)) for word in response)
    param2 = '%s' %  _encode_for_device(param)
    
    print ("param: "+param2)
    print ("check: "+x.decode())
    if x.decode().strip() != param2.strip():
        print(("ERROR\n%s\n%s" % (_encode_for_device(param),
                                 _encode_for_device(x))))
        ser.write(b'q')
        sys.exit(1)
    l.debug('Check done\n')

def _send_key(ser, key):
    _send_parameter(ser, 'k', key)

def _send_plaintext(ser, plaintext):
    _send_parameter(ser, 'p', plaintext)

def _send_init(ser, init):
    _send_parameter(ser, 'i', init)

#Code from https://stackoverflow.com/questions/16716302/how-do-i-fit-a-sine-curve-to-my-data-with-pylab-and-numpy
def fit_sin(tt, yy):
    import scipy.optimize
    '''Fit sin to the input time sequence, and return fitting parameters "amp", "omega", "phase", "offset", "freq", "period" and "fitfunc"'''
    tt = numpy.array(tt)
    yy = numpy.array(yy)
    ff = numpy.fft.fftfreq(len(tt), (tt[1]-tt[0]))   # assume uniform spacing
    Fyy = abs(numpy.fft.fft(yy))
    guess_freq = abs(ff[numpy.argmax(Fyy[1:])+1])   # excluding the zero frequency "peak", which is related to offset
    guess_amp = numpy.std(yy) * 2.**0.5
    guess_offset = numpy.mean(yy)
    guess = numpy.array([guess_amp, 2.*numpy.pi*guess_freq, 0., guess_offset])

    def sinfunc(t, A, w, p, c):  return A * numpy.sin(w*t + p) + c
    popt, pcov = scipy.optimize.curve_fit(sinfunc, tt, yy, p0=guess)
    A, w, p, c = popt
    f = w/(2.*numpy.pi)
    fitfunc = lambda t: A * numpy.sin(w*t + p) + c
    return {"amp": A, "omega": w, "phase": p, "offset": c, "freq": f, "period": 1./f, "fitfunc": fitfunc, "maxcov": numpy.max(pcov), "rawres": (guess,popt,pcov)}
    
    
    
def analyze_snr(capture_file, plot=False, target_path=None, config_variables=""):
    from matplotlib import pyplot as plt
    from scipy.optimize import curve_fit
    import scipy
    
    """
    Post-process a GNUradio capture to get a clean and well-aligned trace.

    The configuration is a reproduce.AnalysisConfig tuple. The optional
    average_file_name specifies the file to write the average to (i.e. the
    template candidate).
    """
    try:
        with open(capture_file) as f:
            data = np.fromfile(f, dtype=scipy.complex64)

        if len(data) == 0:
            print("Warning! empty data, replacing with zeros")
    
        N = len(data) # number of data points
        tt = numpy.linspace(0, 10, N)
        tt2 = numpy.linspace(0, 10, 10*N)
        res = fit_sin(tt,data)
        print( "Amplitude=%(amp)s, Angular freq.=%(omega)s, phase=%(phase)s, offset=%(offset)s, Max. Cov.=%(maxcov)s" % res )

        plt.plot(tt, yy, "-k", label="y", linewidth=2)
        plt.plot(tt, yynoise, "ok", label="y with noise")
        plt.plot(tt2, res["fitfunc"](tt2), "r-", label="y fit curve", linewidth=2)
        plt.legend(loc="best")

        plt.savefig(target_path+"/plotted_fit.png")
        if plot:
            plt.show()
        with open(target_path+'/config.txt', 'w') as o:
            o.write(str(config_variables))
        
        np.save(target_path+"/raw.npy",data)
    except Exception as inst:
        print(inst)
        

@cli.command()
@click.argument("output-path", type=click.Path(exists=True, file_okay=False))
@click.option("-k","--broadcast-on-freq", default=0, show_default=True,
              help="Sets the broadcast frequency (see device firmware for supported frequency)")
@click.option("-c","--channel", default=-1, show_default=True,
              help="Sets the Bluetooth channel (if set, option k is ignored)")
@click.option("--plot/--no-plot", default=False, show_default=True,
              help="Plot the results of trace collection.")
@click.option("-p", "--set-power", default=0, show_default=True,
              help="Sets the device to a specific power level")
@click.option("--sampling-rate", default=5000000, show_default=True,
              help="Sets the rate to sample the received signal")
@click.option("--broadcast-time", default=0.5, show_default=True, type=float,
              help="Number of seconds to broadcast")
@click.option("--usrp-gain", default=20, show_default=True,
              help="Sets the receiver's Universal Software Radio Peripheral gain")
@click.option("--rf-gain", default=0, show_default=True,
              help="Sets the receiver's RF gain")
@click.option("--if-gain", default=35, show_default=True,
              help="Sets the receiver's baseband gain")
@click.option("--bb-gain", default=40, show_default=True,
              help="Sets the receiver's intermediate gain")
@click.option("--plutosdr-gain", default=65, show_default=True,
              help="Sets the receiver's Pluto SDR gain")
@click.option("--gnuradio-startup-wait", default=0.03, show_default=True, type=float,
              help="Number of seconds to wait for gnuradio to begin the trace")
@click.option("--gnuradio-closing-wait", default=0.05, show_default=True, type=float,
              help="Number of seconds to wait for gnuradio to begin the trace")
@click.option("-w","--radio-wave", default='c', show_default=True,
              type=click.Choice(['c','o'], case_sensitive=True),
              help="Sets the transmission of the target device, 'c' for continuous (unmodulated) wave or 'o' for a modulated mave")
def calc_snr(output_path, broadcast_on_freq, channel, plot, set_power, sampling_rate, broadcast_time,
              usrp_gain, rf_gain, if_gain, bb_gain, plutosdr_gain, gnuradio_startup_wait, gnuradio_closing_wait, radio_wave):
    """
    Collect Spectrograms to analyze digital noise from crypto.

    The config is a JSON file containing parameters for trace analysis; see the
    definitions of FirmwareConfig and CollectionConfig for descriptions of each
    parameter.
    """
    print(str(locals()))

    with _open_serial_port() as ser:
        print((ser.readline()))

        if set_power != 0:
            l.debug('Setting power level to '+str(set_power))
            ser.write(('p'+str(set_power)).encode('UTF-8'))
            ser.readline()
            ser.readline()

        #This section diverges from the Screaming Channels code 
        if channel == -1:
            l.debug('Selecting frequency')
            ser.write(b'k')
            print((ser.readline()))
            ser.write(b'%02d\n'%broadcast_on_freq)
            print((ser.readline()))
        else:
            l.debug('Selecting channel')
            ser.write(b'a')
            print((ser.readline()))
            ser.write(b'%02d\n'%channel)
            print((ser.readline()))
	
        l.debug('Starting continuous (unmodulated) or modulated RF wave')
        ser.write(radio_wave.encode())     # start continuous (c) or modulated (o) wave
        print((ser.readline()))
    
        l.debug('Starting GNUradio')
        gnuradio = GNUradio(listen_on_freq, sampling_rate, False, usrp_gain, rf_gain, if_gain, bb_gain, plutosdr_gain)

        gnuradio.start()
        time.sleep(gnuradio_startup_wait + broadcast_time + gnuradio_closing_wait)

        gnuradio.stop()
        gnuradio.wait()
        

        analyze_snr(OUTFILE, plot, output_path, locals())

        gnuradio.reset_trace()

        ser.write(b'e')     # turn off continuous wave
        
        time.sleep(1)
        ser.close()

def _open_serial_port():
    l.debug("Opening serial port")
    return serial.Serial(DEVICE, BAUD, timeout=5)


class GNUradio(gr.top_block):
    """GNUradio capture from SDR to file."""
    def __init__(self, frequency=2.464e9, sampling_rate=5e6, conventional=False,
                 usrp_gain=40, hackrf_gain=0, hackrf_gain_if=40, hackrf_gain_bb=44, plutosdr_gain=64):
        gr.top_block.__init__(self, "Top Block")

        if RADIO in (Radio.USRP, Radio.USRP_mini, Radio.USRP_B210):
            radio_block = uhd.usrp_source(
                ("addr=" + RADIO_ADDRESS.encode("ascii"))
                if RADIO == Radio.USRP else "",
                uhd.stream_args(cpu_format="fc32", channels=[0]))
            radio_block.set_center_freq(frequency)
            radio_block.set_samp_rate(sampling_rate)
            radio_block.set_gain(usrp_gain)
            radio_block.set_antenna("TX/RX")
        elif RADIO == Radio.USRP_B210_MIMO:
            radio_block = uhd.usrp_source(
        	",".join(('', "")),
        	uhd.stream_args(
        		cpu_format="fc32",
        		channels=list(range(2)),
        	),
            )
            radio_block.set_samp_rate(sampling_rate)
            radio_block.set_center_freq(frequency, 0)
            radio_block.set_gain(usrp_gain, 0)
            radio_block.set_antenna('RX2', 0)
            radio_block.set_bandwidth(sampling_rate/2, 0)
            radio_block.set_center_freq(frequency, 1)
            radio_block.set_gain(usrp_gain, 1)
            radio_block.set_antenna('RX2', 1)
            radio_block.set_bandwidth(sampling_rate/2, 1)
 
        elif RADIO == Radio.HackRF or RADIO == Radio.bladeRF:
            mysdr = str(RADIO).split(".")[1].lower() #get "bladerf" or "hackrf"
            radio_block = osmosdr.source(args="numchan=1 "+mysdr+"=0")
            radio_block.set_center_freq(frequency, 0)
            radio_block.set_sample_rate(sampling_rate)
            # TODO tune parameters
            radio_block.set_freq_corr(0, 0)
            radio_block.set_dc_offset_mode(True, 0)
            radio_block.set_iq_balance_mode(True, 0)
            radio_block.set_gain_mode(True, 0)
            radio_block.set_gain(hackrf_gain, 0)
            if conventional:
                # radio_block.set_if_gain(27, 0)
                # radio_block.set_bb_gain(30, 0)
                radio_block.set_if_gain(25, 0)
                radio_block.set_bb_gain(27, 0)
            else:
                radio_block.set_if_gain(hackrf_gain_if, 0)
                radio_block.set_bb_gain(hackrf_gain_bb, 0)
            radio_block.set_antenna('', 0)
            radio_block.set_bandwidth(3e6, 0)
            
        elif RADIO == Radio.PlutoSDR:
            bandwidth = 3e6
            radio_block = iio.pluto_source(RADIO_ADDRESS.encode("ascii"),
                                           int(frequency), int(sampling_rate),
                                           1 - 1, int(bandwidth), 0x8000, True,
                                           True, True, "manual", plutosdr_gain,
                                           '', True)
        else:
            raise Exception("Radio type %s is not supported" % RADIO)


        self._file_sink = blocks.file_sink(gr.sizeof_gr_complex, OUTFILE)
        print(radio_block)
        print(self._file_sink)
        self.connect((radio_block, 0), (self._file_sink, 0))

        if RADIO == Radio.USRP_B210_MIMO:
            self._file_sink_2 = blocks.file_sink(gr.sizeof_gr_complex,
            OUTFILE+"_2")
            self.connect((radio_block, 1), (self._file_sink_2, 0))


    def reset_trace(self):
        """
        Remove the current trace file and get ready for a new trace.
        """
        self._file_sink.open(OUTFILE)
        
        if RADIO == Radio.USRP_B210_MIMO:
            self._file_sink_2.open(OUTFILE+"_2")

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *args):
        self.stop()


if __name__ == "__main__":
    cli()
