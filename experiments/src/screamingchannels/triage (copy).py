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

from matplotlib import pyplot as plt
from matplotlib import mlab

from scipy import signal
from scipy.signal import butter, lfilter
import peakutils



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
#

def butter_bandstop(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='bandstop')
    return b, a

def butter_bandstop_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandstop(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y

def butter_highpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    return b, a

def butter_highpass_filter(data, cutoff, fs, order=5):
    b, a = butter_highpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y

def find_starts(config, data):
    """
    Find the starts of interesting activity in the signal.

    The result is a list of indices where interesting activity begins, as well
    as the trigger signal and its average.
    """
    
    trigger = butter_bandpass_filter(
        data, config.bandpass_lower, config.bandpass_upper,
        config.sampling_rate, 6)
    trigger = np.absolute(trigger)
    trigger = butter_lowpass_filter(
        trigger, config.lowpass_freq,config.sampling_rate, 6)

    # transient = 0.0005
    # start_idx = int(transient * config.sampling_rate)
    start_idx = 0
    average = np.average(trigger[start_idx:])
    maximum = np.max(trigger[start_idx:])
    minimum = np.min(trigger[start_idx:])
    middle = (np.max(trigger[start_idx:]) - min(trigger[start_idx:])) / 2
    if average < 1.1*middle:
        print("")
        print("Adjusting average to avg + (max - avg) / 2")
        average = average + (maximum - average) / 2
    offset = -int(config.trigger_offset * config.sampling_rate)

    if config.trigger_rising:
        trigger_fn = lambda x, y: x > y
    else:
        trigger_fn = lambda x, y: x < y

    # The cryptic numpy code below is equivalent to looping over the signal and
    # recording the indices where the trigger crosses the average value in the
    # direction specified by config.trigger_rising. It is faster than a Python
    # loop by a factor of ~1000, so we trade readability for speed.
    trigger_signal = trigger_fn(trigger, average)[start_idx:]
    starts = np.where((trigger_signal[1:] != trigger_signal[:-1])
                      * trigger_signal[1:])[0] + start_idx + offset + 1
    if trigger_signal[0]:
        starts = np.insert(starts, 0, start_idx + offset)

    # plt.plot(data)
    # plt.plot(trigger*100)
    # plt.axhline(y=average*100)
    # plt.show()

    return starts, trigger, average


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
    
    
def makePlots(capture_file, plot=False, target_path=None, config_variables=""):
    from matplotlib import pyplot as plt
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
    
        plt.plot(data)
        # Annotation
        annotation_text = ""
        for cvar in ['listen_on_freq', 'set_power', 'encrypt_mode', 'radio_wave', 'rf_gain', 'if_gain', 'bb_gain']:
        	annotation_text += cvar +": "+str(config_variables[cvar])+"\r\n"        	
        plt.text(40, 0, annotation_text, fontsize=8)
        plt.savefig(target_path+"/plotted_raw_data.png")
        if plot:
            plt.show()
        with open(target_path+'/config.txt', 'w') as o:
            o.write(str(config_variables))
        
        
        ## cut usless transient
        #data = data[int(config.drop_start * config.sampling_rate):]
        
        
        
        np.save(target_path+"/raw.npy",data)
    except Exception as inst:
        print(inst)
        

@cli.command()
@click.argument("output-path", type=click.Path(exists=True, file_okay=False))
@click.option("-k","--broadcast-on-freq", default=0, show_default=True,
              help="Sets the broadcast frequency (see device firmware for supported frequency)")
@click.option("-c","--channel", default=-1, show_default=True,
              help="Sets the Bluetooth channel (if set, option k is ignored)")
@click.option("-f","--listen-on-freq", default=0, show_default=True,
              help="Sets the frequency for the device to broadcast")
@click.option("--plot/--no-plot", default=False, show_default=True,
              help="Plot the results of trace collection.")
@click.option("-p", "--set-power", default=0, show_default=True,
              help="Sets the device to a specific power level")
@click.option("-e","--encrypt-mode", default="tinyaes", show_default=True,
              type=click.Choice(["tinyaes","hwcrypto","hwcrypto_ecb", "maskaes"], case_sensitive=True),
              help="Sets the encryption mode")
@click.option("-m","--mask-mode", default=0, show_default=True,
              type=click.Choice([0,1,2,3,4,5,6,7,8,9], case_sensitive=True),
              help="If --ecrypt-mode is set to aesmask, then aesmask-mode will determine the maksing level")
@click.option("--sampling-rate", default=5000000, show_default=True,
              help="Sets the rate to sample the received signal")
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
def triage(output_path, broadcast_on_freq, channel, listen_on_freq, plot, set_power, encrypt_mode, mask_mode, sampling_rate,
              usrp_gain, rf_gain, if_gain, bb_gain, plutosdr_gain, gnuradio_startup_wait, gnuradio_closing_wait, radio_wave):
    """
    Collect Spectrograms to analyze digital noise from crypto.

    The config is a JSON file containing parameters for trace analysis; see the
    definitions of FirmwareConfig and CollectionConfig for descriptions of each
    parameter.
    """
    if encrypt_mode == "maskaes":
        mode_command='w'
    elif encrypt_mode == "hwcrypto":
        mode_command='u'
    elif encrypt_mode == "hwcrypto_ecb":
        mode_command='U'
    else:
        mode_command='n' #tinyaes mode

    # Generate the key/plaintext
    _key = ('\x13'*16).encode() 
    _plaintext = ('\x37'*16).encode() 


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
        
        l.debug('Setting spike frequency')
        ser.write(b'K')
        print((ser.readline()))
        ser.write(b'%d\n'%(listen_on_freq/(1000*1000))) #converting Hz to MHz
        print((ser.readline()))
	
        l.debug('Starting continuous (unmodulated) or modulated RF wave')
        ser.write(radio_wave.encode())     # start continuous (c) or modulated (o) wave
        print((ser.readline()))

        l.debug('Entering test mode')
        ser.write(mode_command.encode()) # enter test mode
        print((ser.readline()))

        l.debug('Setting trace repetions at 1')
        ser.write(('n1\r\n').encode())
        print((ser.readline()))

        # The key never changes, so we can just set it once and for all.
        _send_key(ser, _key)

        # The plaintext never changes, so we can just set it once and for all.
        _send_plaintext(ser, _plaintext)

        if encrypt_mode == 'maskaes':
            l.debug('Setting masking mode to %d', mask_mode)
            ser.write(('%d\r\n' % mask_mode).encode())
            print((ser.readline()))


        l.debug('Starting GNUradio')
        gnuradio = GNUradio(listen_on_freq, sampling_rate, False, usrp_gain, rf_gain, if_gain, bb_gain, plutosdr_gain)

        gnuradio.start()
        time.sleep(gnuradio_startup_wait)

        # The test mode supports repeated actions.
        l.debug('Start repetitions')
        ser.write('r'.encode())
        ser.readline() # wait until done
                
        time.sleep(gnuradio_closing_wait)
        gnuradio.stop()
        gnuradio.wait()
        

        makePlots(OUTFILE, plot, output_path, locals())

        gnuradio.reset_trace()

        ser.write(b'q')     # quit tiny_aes mode
        print((ser.readline()))
        ser.write(b'e')     # turn off continuous wave
        
        time.sleep(1)
        ser.close()



@cli.command()
@click.argument("output-path", type=click.Path(exists=True, file_okay=False))
@click.option("-k","--broadcast-on-freq", default=0, show_default=True,
              help="Sets the broadcast frequency (see device firmware for supported frequency)")
@click.option("-c","--channel", default=-1, show_default=True,
              help="Sets the Bluetooth channel (if set, option k is ignored)")
@click.option("-p", "--set-power", default=0, show_default=True,
              help="Sets the device to a specific power level")
@click.option("-e","--encrypt-mode", default="tinyaes", show_default=True,
              type=click.Choice(["tinyaes","hwcrypto","hwcrypto_ecb", "maskaes"], case_sensitive=True),
              help="Sets the encryption mode")
@click.option("-m","--mask-mode", default=0, show_default=True,
              type=click.Choice([0,1,2,3,4,5,6,7,8,9], case_sensitive=True),
              help="If --ecrypt-mode is set to aesmask, then aesmask-mode will determine the maksing level")
@click.option("--sampling-rate", default=5000000, show_default=True,
              help="Sets the rate to sample the received signal")
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
@click.option("-w","--radio-wave", default='c', show_default=True,
              type=click.Choice(['c','o'], case_sensitive=True),
              help="Sets the transmission of the target device, 'c' for continuous (unmodulated) wave or 'o' for a modulated mave")
def find_digital_noise(broadcast_on_freq, channel, set_power, encrypt_mode, mask_mode, sampling_rate,
              usrp_gain, rf_gain, if_gain, bb_gain, plutosdr_gain, radio_wave):
    
    if encrypt_mode == "maskaes":
        mode_command='w'
    elif encrypt_mode == "hwcrypto":
        mode_command='u'
    elif encrypt_mode == "hwcrypto_ecb":
        mode_command='U'
    else:
        mode_command='n' #tinyaes mode

    # Generate the key/plaintext
    _key = ('\x13'*16).encode() 
    _plaintext = ('\x37'*16).encode() 


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

        l.debug('Entering test mode')
        ser.write(mode_command.encode()) # enter test mode
        print((ser.readline()))

        l.debug('Setting trace repetions at 1000')
        ser.write(('n1\r\n').encode())
        print((ser.readline()))

        # The key never changes, so we can just set it once and for all.
        _send_key(ser, _key)

        # The plaintext never changes, so we can just set it once and for all.
        _send_plaintext(ser, _plaintext)

        if encrypt_mode == 'maskaes':
            l.debug('Setting masking mode to %d', mask_mode)
            ser.write(('%d\r\n' % mask_mode).encode())
            print((ser.readline()))


        l.debug('Starting GNUradio')
        gnuradio = GNUradio(listen_on_freq, sampling_rate, False, usrp_gain, rf_gain, if_gain, bb_gain, plutosdr_gain)

        gnuradio.start()
        time.sleep(gnuradio_startup_wait)

        # The test mode supports repeated actions.
        l.debug('Start repetitions')
        ser.write('r'.encode())
        ser.readline() # wait until done
                
        time.sleep(gnuradio_closing_wait)
        gnuradio.stop()
        gnuradio.wait()
        

        makePlots(OUTFILE, plot, output_path, locals())

        gnuradio.reset_trace()

        ser.write(b'q')     # quit tiny_aes mode
        print((ser.readline()))
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
