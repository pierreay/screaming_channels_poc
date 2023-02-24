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

#from gnuradio import blocks, gr, uhd, iio
#import osmosdr

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
    
    


@cli.command()
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
def survey(broadcast_on_freq, channel, set_power, encrypt_mode, mask_mode, sampling_rate,
              usrp_gain, rf_gain, if_gain, bb_gain, plutosdr_gain, radio_wave):
    
    os.system("sc-waterfall "+str(broadcast_on_freq)+" &")
    
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
            #Fixing frequency for compatibility
            broadcast_on_freq = 2.4e9
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

        l.debug('Setting trace repetions at 10000')
        ser.write(('n10000\r\n').encode())
        print((ser.readline()))

        # The key never changes, so we can just set it once and for all.
        _send_key(ser, _key)

        # The plaintext never changes, so we can just set it once and for all.
        _send_plaintext(ser, _plaintext)

        if encrypt_mode == 'maskaes':
            l.debug('Setting masking mode to %d', mask_mode)
            ser.write(('%d\r\n' % mask_mode).encode())
            print((ser.readline()))

        for x in range(5):
            # The test mode supports repeated actions.
            print('Start repetitions')
            ser.write('r'.encode())
            ser.readline() # wait until done
            time.sleep(2)

        ser.write(b'q')     # quit tiny_aes mode
        print((ser.readline()))
        ser.write(b'e')     # turn off continuous wave
        
        time.sleep(1)
        ser.close()

def _open_serial_port():
    l.debug("Opening serial port")
    return serial.Serial(DEVICE, BAUD, timeout=5)



if __name__ == "__main__":
    cli()
