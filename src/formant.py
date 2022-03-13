from __future__ import annotations

import matplotlib.pyplot as plt
import numpy
import parselmouth


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
        plt.plot(result)
        plt.show()

    return result


if __name__ == '__main__':
    print(calculate_freq_info(parselmouth.Sound('D:/Downloads/Vowels-Extract-Z-44kHz.flac')))
    print(calculate_freq_info(parselmouth.Sound('D:/Downloads/Vowels-Azalea.flac')))
