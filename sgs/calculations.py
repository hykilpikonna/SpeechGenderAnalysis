from __future__ import annotations

import math
from dataclasses import dataclass

import numpy
import numpy as np
import parselmouth


def calculate_tilt(sound: parselmouth.Sound) -> float | None:
    """
    Compute spectral tilt

    Based on statistics, spectral tilt's range is around [-0.5, -0.08]. Higher spectral tilt is
    correlated with a creaky voice, and lower spectral tilt is correlated with a breathy voice.

    Implementation modified from https://github.com/Voice-Lab/VoiceLab/blob/main/Voicelab/toolkits/Voicelab/MeasureSpectralTiltNode.py

    Credit to VoiceLab (https://github.com/Voice-Lab/VoiceLab)

    :param sound: Decoded sound
    :return: Spectral tilt value or None if no value is found
    """
    spectrum = sound.to_spectrum()
    total_bins = spectrum.get_number_of_bins()
    dBValue = []
    bins = []

    # convert spectral values to dB
    for bin in range(total_bins):
        bin_number = bin + 1
        realValue = spectrum.get_real_value_in_bin(bin_number)
        imagValue = spectrum.get_imaginary_value_in_bin(bin_number)
        rmsPower = math.sqrt((realValue ** 2) + (imagValue ** 2))
        if rmsPower <= 0:
            print(f'Error: rmsPower={rmsPower}, needs to be positive!')
            return None
        db = 20 * (math.log10(rmsPower / 0.0002))
        dBValue.append(db)
        bin_number += 1
        bins.append(bin)

    # find maximum dB value, for rescaling purposes
    maxdB = max(dBValue)
    mindB = min(dBValue)  # this is wrong in Owren's script, where mindB = 0
    rangedB = maxdB - mindB

    # stretch the spectrum to a normalized range that matches the number of frequency values
    scalingConstant = (total_bins - 1) / rangedB
    scaled_dB_values = []
    for value in dBValue:
        scaled_dBvalue = value + abs(mindB)
        scaled_dBvalue *= scalingConstant
        scaled_dB_values.append(scaled_dBvalue)

    # find slope
    sumXX = 0
    sumXY = 0
    sumX = sum(bins)
    sumY = sum(scaled_dB_values)

    for bin in bins:
        currentX = bin
        sumXX += currentX ** 2
        sumXY += currentX * scaled_dB_values[bin]

    sXX = sumXX - ((sumX * sumX) / len(bins))
    sXY = sumXY - ((sumX * sumY) / len(bins))
    spectral_tilt = sXY / sXX
    return spectral_tilt


def calculate_freq_info(audio: parselmouth.Sound, show_plot=False) -> numpy.ndarray:
    """
    Calculate pitch and frequency

    :param show_plot: Show pyplot plot or not
    :param audio: Sound input
    :return: 2D Array (Each row is 1/100 of a second, row[0] is pitch (fundamental frequency), row[1:4] is formant)
    """
    pitch_values = audio.to_pitch(0.01).selected_array['frequency']
    formant_values = audio.to_formant_burg(0.01)
    result = numpy.ndarray([len(pitch_values), 4], 'float32')

    for i in range(len(pitch_values)):
        pitch = pitch_values[i]
        result[i][0] = pitch if pitch else None
        for f in range(1, 4):
            result[i][f] = formant_values.get_value_at_time(f, i / 100) if pitch else None

    if show_plot:
        import matplotlib.pyplot as plt
        plt.plot(result)
        plt.show()

    return result


@dataclass
class FrequencyStats:
    pitch: Statistics
    f1: Statistics
    f2: Statistics
    f3: Statistics


@dataclass
class Statistics:
    mean: float
    median: float
    q1: float
    q3: float
    iqr: float
    min: float
    max: float
    n: int


def calc_col_stats(col: np.ndarray) -> Statistics:
    """
    Compute statistics for a data column

    :param col: Input column (tested on 1D array)
    :return: Statistics
    """
    col = col[~numpy.isnan(col)]
    q1 = np.quantile(col, 0.25)
    q3 = np.quantile(col, 0.75)
    return Statistics(
        float(np.mean(col)),
        float(np.median(col)),
        float(q1),
        float(q3),
        float(q3 - q1),
        float(np.min(col)),
        float(np.max(col)),
        len(col)
    )


def calculate_freq_statistics(arr: np.ndarray) -> FrequencyStats:
    """
    Calculate frequency data array statistics

    :param arr: n-by-4 Array from calculate_freq_info
    :return: Statistics
    """
    result = [calc_col_stats(arr[:, i]) for i in range(0, 4)]

    return FrequencyStats(*result)
