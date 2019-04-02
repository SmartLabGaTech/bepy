from scipy.signal import butter, lfilter
import numpy as np


def stretch_exp(x, R, A, b, t):  #
    exponent = (1 / t) * x  # define the function to fit too
    stretch = exponent ** b  #
    return R * np.exp(-stretch) + A  #

def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a


def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y


# A mean absolute deviation
def madfilter(data, sensitivity=3):

    outflags = np.full(np.shape(data), False)

    data[np.where(np.isnan(data))] = 0
    data[np.where(np.isinf(data))] = 0

    mean = np.mean(data)

    absdevi = lambda x: np.abs(x - mean)

    vfunc = np.vectorize(absdevi)

    deviations = vfunc(data)

    mad = np.mean(deviations)

    outflags[np.where(deviations > mad * sensitivity)] = True

    return outflags


def cleanbychirp(data, sensitivity=3):

    outflags = np.full(np.shape(data), False)

    i = 0

    for column in data.T:
        outflags[:, i] = madfilter(column, sensitivity)
        i = i + 1

    return outflags


def collapseflags(flags, sensitivity=3, axis=1):
    badperpoint = np.sum(flags.values, axis=axis)
    return madfilter(badperpoint, sensitivity)

