import math
import parselmouth

from parselmouth.praat import call


# https://github.com/Voice-Lab/VoiceLab/blob/2edf9678866eb5f5f230bf1578e1aa418f7f4917/Voicelab/toolkits/Voicelab/MeasureSpectralTiltNode.py
# The closer to positive the creakier it is#
def tilt(sound: parselmouth.Sound):
    # read the sound

    window_length = sound.get_total_duration()


    # Compute begin and end times, set window
    end = call(sound, "Get end time")
    midpoint = end / 2

    begintime = midpoint - (window_length / 2)
    endtime = midpoint + (window_length / 2)
    part_to_measure = sound.extract_part(begintime, endtime)
    spectrum = part_to_measure.to_spectrum()
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
    # print(spectral_tilt)
    return spectral_tilt
