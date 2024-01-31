import io

#import matplotlib
#matplotlib.use('agg')
#import matplotlib.pyplot as plt

import scipy.signal


def frame_key_to_num(key):
    return int(key[-4:])


def convert_dist_pixel_to_cm(dist, sour):
    out = dist * sour['PhysicalDeltaY']
    return out


def make_dist_fig(dist, format='png'):
    plt.figure()
    plt.plot(dist)
    buf = io.BytesIO()
    plt.savefig(buf, format=format)
    buf.seek(0)
    return buf.getvalue()

def find_peaks_lvid_cm(dist):
    max_peaks = scipy.signal.find_peaks(dist, threshold=0.5, distance=4)
    min_peaks = scipy.signal.find_peaks(dist, threshold=0.5, distance=4)
