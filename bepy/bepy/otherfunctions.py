from scipy.signal import butter, lfilter
from scipy.ndimage import filters
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


def normalize_image(img):
    return (img - np.nanmin(img)) / (np.nanmax(img) - np.nanmin(img))


def generate_distance_kernel(shape):
    num_rows = shape[0]
    num_cols = shape[1]
    master_num_rows = 2*num_rows-1
    master_num_cols = 2*num_cols-1
    dist_master = np.empty((master_num_rows, master_num_cols))
    for row in range(master_num_rows):
        for col in range(master_num_cols):
            dist = pow(pow(row-num_rows+1, 2) + pow(col-num_cols+1, 2), 0.5)
            dist_master[row, col] = dist
    return dist_master


def generate_template_grid(img_side_length, grid_size):
    img = np.zeros((img_side_length, img_side_length))
    buffer_pixels = (img_side_length - grid_size) / (grid_size - 1)
    for row in range(grid_size):
        for col in range(grid_size):
            img[int(round(row + buffer_pixels * row)), int(round(col + buffer_pixels * col))] = 1
    return img == 1


def intersect_lines(m1, b1, m2, b2):
    x = (b2 - b1)/(m1 - m2)
    y = m1*x + b1
    return x, y


def center_phase(single_phase_loop):
    loop1 = single_phase_loop - min(single_phase_loop)
    loop2 = loop1 - max(loop1)/2
    return loop2


def intersect_lines(m1, b1, m2, b2):
    x = (b2 - b1) / (m1 - m2)
    y = m1 * x + b1
    return x, y