from __future__ import annotations

import matplotlib.pyplot as plt
import numpy
import parselmouth


def calculate_freq_info(audio: parselmouth.Sound) -> numpy.ndarray:
    """
    Calculate pitch and frequency

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

    plt.plot(result)
    plt.show()

    return result


if __name__ == '__main__':
    # sound = parselmouth.Sound.read('../test.wav')
    print(calculate_freq_info(parselmouth.Sound('D:/Downloads/Vowels-Extract-Z-44kHz.flac')))
    print(calculate_freq_info(parselmouth.Sound('D:/Downloads/Vowels-Azalea.flac')))
    # spf = wave.open('../test.wav', 'r')
    # x = spf.readframes(-1)
    # # Get file as numpy array.
    # x = numpy.fromstring(x, dtype=numpy.int16)
    # # Get Hamming window.
    # N = len(x)
    # w = numpy.hamming(N)
    # print(w)
    # # Apply window and high pass filter.
    # x1 = x * w
    # x1 = lfilter([1., -0.63], 1, x1)
    # Fs = spf.getframerate()
    # ncoeff = 2 + Fs // 1000
    # # Get LPC.
    # A = lpc(x1, ncoeff, -1)
    #
    # # Get roots.
    # rts = numpy.roots(A)
    # rts = [r for r in rts if numpy.imag(r) >= 0]
    #
    # # Get angles.
    # angz = numpy.arctan2(numpy.imag(rts), numpy.real(rts))
    #
    # # Get frequencies.
    # Fs = spf.getframerate()
    # frqs = sorted(angz * (Fs / (2 * math.pi)))
    # pitch_values = parselmouth.Sound('../test.wav').to_pitch(0.01).selected_array['frequency']
    # print(len(pitch_values))
    # print(N)
    # print(frqs)
    # formant_values = parselmouth.Sound('../test.wav').to_formant_burg(0.01)
    # #formant_values = parselmouth.Formant.get_value_at_time(formant_values,2,1.0)
    # formant=[]
    # for i in range (len(pitch_values)):
    #         formant.append([parselmouth.Formant.get_value_at_time(formant_values,k, i/100) for k in range(1, 5)])
    # print(formant)
    # print(parselmouth.Formant.get_value_at_time(formant_values,1,4.1))
    # print(parselmouth.Formant.get_value_at_time(formant_values,2,4.1))
    # print(parselmouth.Formant.get_value_at_time(formant_values,3,4.1))
    # print(parselmouth.Formant.get_value_at_time(formant_values,4,4.1))
    # print(parselmouth.Formant.get_value_at_time(formant_values,5,4.1))
    # print(len(formant))
