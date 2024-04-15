#!/usr/bin/python3
import numpy as np

import matplotlib

from matplotlib import pyplot as plt
from matplotlib import mlab

from scipy import signal
from scipy.signal import butter, lfilter

import lib.load as load
import lib.soapysdr as soapysdr
import lib.complex as complex

#
# Filter creation functions taken from https://stackoverflow.com/a/12233959
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

def find_starts(config, data, target_path, index):
    """
    Find the starts of interesting activity in the signal.

    The result is a list of indices where interesting activity begins, as well
    as the trigger signal and its average.
    """
    
    trigger = butter_bandpass_filter(
        data, config.bandpass_lower, config.bandpass_upper,
        config.sampling_rate, 6)
    
    #TOM ADDITION START
    #plt.clf()
    #plt.plot(trigger)
    #plt.savefig(target_path+"/"+str(index)+"_4-trigger-bandpass.png")
    #TOM ADDITION END
        
    trigger = np.absolute(trigger)
    # Use an SOS filter because the old one raised exception when using small
    # lowpass values:
    lpf = signal.butter(5, config.lowpass_freq, 'low', fs=config.sampling_rate, output='sos')
    trigger = np.array(signal.sosfilt(lpf, trigger), dtype=trigger.dtype)
    # trigger = butter_lowpass_filter(trigger, config.lowpass_freq,config.sampling_rate, 6)

    #TOM ADDITION START
    #plt.clf()
    #plt.plot(trigger)
    #plt.savefig(target_path+"/"+str(index)+"_5-trigger-lowpass.png")
    #TOM ADDITION END

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

    if config.trigger_threshold is not None and config.trigger_threshold > 0:
        print("Use config trigger treshold instead of average")
        average = config.trigger_threshold / 100 # NOTE: / 100 because of *100 in plot_results().

    # The cryptic numpy code below is equivalent to looping over the signal and
    # recording the indices where the trigger crosses the average value in the
    # direction specified by config.trigger_rising. It is faster than a Python
    # loop by a factor of ~1000, so we trade readability for speed.
    trigger_signal = trigger_fn(trigger, average)[start_idx:]
    starts = np.where((trigger_signal[1:] != trigger_signal[:-1])
                      * trigger_signal[1:])[0] + start_idx + offset + 1
    # if trigger_signal[0]:
    #     starts = np.insert(starts, 0, start_idx + offset)

    #TOM ADDITION START
    #plt.clf()
    #plt.plot(trigger_signal)
    #plt.savefig(target_path+"/"+str(index)+"_6-triggerstart.png")
    #TOM ADDITION END


    # plt.plot(data)
    # plt.plot(trigger*100)
    # plt.axhline(y=average*100)
    # plt.show()

    return starts, trigger, average

# The part that uses a frequency component as trigger was initially
# inspired by https://github.com/bolek42/rsa-sdr
# The code below contains a few hacks to deal with all possible errors we
# encountered with different radios and setups. It is not very clean but it is
# quite stable.
def extract(capture_file, config, average_file_name=None, plot=False, target_path=None, savePlot=False, index=0):
    """Post-process a GNUradio capture to get a clean and well-aligned trace.

    The configuration is a reproduce.AnalysisConfig tuple. The optional
    average_file_name specifies the file to write the average to (i.e. the
    template candidate).

    Return a tuple composed of the extracted amplitude and phase rotation
    traces.

    """
    try:
        # Load data from custom dtype.
        data = soapysdr.MySoapySDR.numpy_load(capture_file)

        # assert len(data) != 0, "ERROR, empty data just after measuring"
        if len(data) == 0:
            print("Warning! empty data, replacing with zeros")
            template = np.load(config.template_name)
            return np.zeros(len(template)), np.zeros(len(template))
    
        #TOM ADDITION START
        #plt.clf()
        #plt.plot(data)
        #plt.savefig(target_path+"/"+str(index)+"_1-data.png")
        #TOM ADDITION END
        # plt.plot(data)
        # plt.show()

        template = np.load(config.template_name) if config.template_name else None
        
        if template is not None and len(template) != int(
                config.signal_length * config.sampling_rate):
            print("WARNING: Template length doesn't match collection parameters. "
                  "Is this the right template?")

        # cut usless transient
        data = data[int(config.drop_start * config.sampling_rate):]


        #TOM ADDITION START
        #plt.clf()
        #plt.plot(data)
        #plt.savefig(target_path+"/"+str(index)+"_2-data-trimmed.png")
        #TOM ADDITION END

        # assert len(data) != 0, "ERROR, empty data after drop_start"
        if len(data) == 0:
           print("Warning! empty data after drop start, replacing with zeros")
           template = np.load(config.template_name)
           return np.zeros(len(template)), np.zeros(len(template))

   
        # polar discriminator
        # fdemod = data[1:] * np.conj(data[:-1])
        # fdemod = np.angle(fdemod)
        # plt.plot(fdemod)
        # plt.show()
        # return fdemod
        # data = fdemod

        # AMPlitude
        data_amp = np.absolute(data)
        # PHase Rotation
        data_phr = complex.get_phase_rot(data)
        # I
        data_i = np.real(data)
        # Q
        data_q = np.imag(data)

        # Compute augmented I/Q.
        data_augmented = data_amp * np.exp(1j * data_phr)
        # I from augmented IQ.
        data_i_augmented = np.real(data_augmented)
        # Q from augmented IQ.
        data_q_augmented = np.imag(data_augmented)

        #TOM ADDITION START
        #plt.clf()
        #plt.plot(data)
        #plt.savefig(target_path+"/"+str(index)+"_3-data-absolute.png")
        #TOM ADDITION END
        #
        # extract/aling trace with trigger frequency + autocorrelation
        #
        # NOTE: find_starts() will work with the amplitude, but we will use the
        # starts indexes against the raw I/Q.
        trace_starts, trigger, trigger_avg = find_starts(config, data_amp, target_path, index)
        
        # extract at trigger + autocorrelate with the first to align
        traces_amp = []
        traces_phr = []
        traces_i = []
        traces_q = []
        traces_i_augmented = []
        traces_q_augmented = []
        trace_length = int(config.signal_length * config.sampling_rate)
        for start in trace_starts:
            if len(traces_amp) >= min(config.num_traces_per_point, config.num_traces_per_point_keep):
                break

            stop = start + trace_length

            if stop > len(data_amp):
                break

            trace_amp = data_amp[start:stop]
            if template is None or len(template) == 0:
                template = trace_amp
                continue

            trace_lpf = butter_lowpass_filter(trace_amp, config.sampling_rate / 4,
                    config.sampling_rate)
            template_lpf = butter_lowpass_filter(template, config.sampling_rate / 4,
                    config.sampling_rate)
            correlation = signal.correlate(trace_lpf**2, template_lpf**2)
            # print("corr={}".format(max(correlation)))
            if max(correlation) <= config.min_correlation:
                print("Skip trace start because corr={} <= corr_min={}".format(max(correlation), config.min_correlation))
                continue

            shift = np.argmax(correlation) - (len(template)-1)
            traces_amp.append(data_amp[start+shift:stop+shift])
            traces_phr.append(data_phr[start+shift:stop+shift])
            traces_i.append(data_i[start+shift:stop+shift])
            traces_q.append(data_q[start+shift:stop+shift])
            traces_i_augmented.append(data_i_augmented[start+shift:stop+shift])
            traces_q_augmented.append(data_q_augmented[start+shift:stop+shift])

        avg_amp = np.average(traces_amp, axis=0)
        avg_phr = np.average(traces_phr, axis=0)
        avg_i = np.average(traces_i, axis=0)
        avg_q = np.average(traces_q, axis=0)
        avg_i_augmented = np.average(traces_i_augmented, axis=0)
        avg_q_augmented = np.average(traces_q_augmented, axis=0)

        if (np.shape(avg_amp) == () or np.shape(avg_phr) == ()
            or np.shape(avg_i) == () or np.shape(avg_q) == ()
            or np.shape(avg_i_augmented) == () or np.shape(avg_q_augmented) == ()):
            return np.zeros(len(template)), np.zeros(len(template)), np.zeros(len(template)), np.zeros(len(template)), np.zeros(len(template)), np.zeros(len(template))

        if average_file_name:
            np.save(average_file_name, avg_amp)

        if plot or savePlot:
            plot_results(config, data_amp, trigger, trigger_avg, trace_starts, traces_amp, target_path, plot, savePlot, "amp")
            plot_results(config, data_phr, trigger, trigger_avg, trace_starts, traces_phr, target_path, plot, savePlot, "phr")
            # plot_results(config, data_i, trigger, trigger_avg, trace_starts, traces_i, target_path, plot, savePlot, "i")
            # plot_results(config, data_q, trigger, trigger_avg, trace_starts, traces_q, target_path, plot, savePlot, "q")
            # plot_results(config, data_i_augmented, trigger, trigger_avg, trace_starts, traces_i_augmented, target_path, plot, savePlot, "i_augmented")
            # plot_results(config, data_q_augmented, trigger, trigger_avg, trace_starts, traces_q_augmented, target_path, plot, savePlot, "q_augmented")

        std = np.std(traces_amp,axis=0)

        print("Extracted ")
        print("Number = ",len(traces_amp))
        print("avg[Max(std)] = %.2E"%avg_amp[std.argmax()])
        print("Max(u) = Max(std) = %.2E"%(max(std)))
        print("Max(u_rel) = %.2E"%(100*max(std)/avg_amp[std.argmax()]),"%")

        # plt.plot(avg, 'r')
        # plt.plot(template, 'b')
        # plt.show()

        if config.keep_all:
            return traces_amp
        else:
            return avg_amp, avg_phr, avg_i, avg_q, avg_i_augmented, avg_q_augmented

    except Exception as inst:
        print(inst)
        print("Error, returning zeros")
        template = np.load(config.template_name)
        return np.zeros(len(template)), np.zeros(len(template)), np.zeros(len(template)), np.zeros(len(template)), np.zeros(len(template)), np.zeros(len(template))

def plot_results(config, data, trigger, trigger_average, starts, traces, target_path=None, plot=True, savePlot=False, title=""):
    plt.subplot(4, 1, 1)

    t = np.linspace(0,len(data) / config.sampling_rate, len(data))
    plt.plot(t, data)
    plt.title(title)
    plt.xlabel("time [s]")
    plt.ylabel("normalized amplitude")
   
    plt.plot(t, trigger*100)
    plt.axhline(y=trigger_average*100, color='y')
    trace_length = int(config.signal_length * config.sampling_rate)
    for start in starts:
        stop = start + trace_length
        plt.axvline(x=start / config.sampling_rate, color='r', linestyle='--')
        plt.axvline(x=stop / config.sampling_rate, color='g', linestyle='--')

    plt.subplot(4, 1, 2)
    #np.set_printoptions(threshold=np.inf)
    #print(data)
    
    plt.specgram(
        data, NFFT=256, Fs=config.sampling_rate, Fc=0, detrend=mlab.detrend_none,
        window=mlab.window_hanning, noverlap=127, cmap=None, xextent=None,
        pad_to=None, sides='default', scale_by_freq=None, mode='default',
        scale='default')
    plt.axhline(y=config.bandpass_lower, color='b', lw=0.2)
    plt.axhline(y=config.bandpass_upper, color='b', lw=0.2)
    plt.title("Spectrogram")
    plt.xlabel("time [s]")
    plt.ylabel("frequency [Hz]")

    # plt.subplot(4, 1, 3)
    # plt.psd(
        # data, NFFT=1024, Fs=config.sampling_rate, Fc=0, detrend=mlab.detrend_none,
        # window=mlab.window_hanning, noverlap=0, pad_to=None,
        # sides='default', scale_by_freq=None, return_line=None)

    if(len(traces) == 0):
        print("WARNING: no encryption was extracted")
    else:
        t = np.linspace(0,len(traces[0]) / config.sampling_rate, len(traces[0]))
        plt.subplot(4, 1, 3)
        for trace in traces:
            plt.plot(t, trace / max(trace))
        plt.title("%d aligned traces" % min(config.num_traces_per_point, config.num_traces_per_point_keep))
        plt.xlabel("time [s]")
        plt.ylabel("normalized amplitude")

        plt.subplot(4,1,4)
        avg = np.average(traces, axis=0)
        plt.plot(t, avg / max(avg))
        plt.title("Average of %d traces" % min(config.num_traces_per_point, config.num_traces_per_point_keep))
        plt.xlabel("time [s]")
        plt.ylabel("normalized amplitude")

    plt.tight_layout()
    plt.subplots_adjust(hspace = 0.5, left=0.1)

    # NOTE: Fix savefig() layout.
    figure = plt.gcf() # Get current figure
    figure.set_size_inches(32, 18) # Set figure's size manually to your full screen (32x18).

    if savePlot and target_path != None:
        plt.savefig(target_path + "/plot_{}.png".format(title), dpi=100, bbox_inches='tight')
    if plot:
        plt.show()

    plt.clf()

if __name__ == "__main__":
    extract(True)
