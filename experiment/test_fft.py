import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import parselmouth

import sgs

if __name__ == '__main__':
    librosa.filters.mel(sr=16000, n_fft=1024, htk=True)

    f = 'Z:/EECS 6414/voice_cnn/test.wav'
    y, sr = librosa.load(f, sr=16000)

    # Plot waveform
    # plt.plot(y)
    # plt.title('Signal')
    # plt.xlabel('Time (samples)')
    # plt.ylabel('Amplitude')
    # plt.show()
    # plt.clf()

    # Plot frequency domain graph at a single time
    n_fft = 2048
    ft = np.abs(librosa.stft(y[:n_fft], hop_length=n_fft + 1))

    # plt.plot(ft)
    # plt.title('Spectrum')
    # plt.xlabel('Frequency Bin')
    # plt.ylabel('Amplitude')
    # plt.show()
    # plt.clf()

    # Plot spectrogram
    spec = np.abs(librosa.stft(y, n_fft=1024, hop_length=512))
    # spec = librosa.amplitude_to_db(spec, ref=np.max)

    # librosa.display.specshow(spec, sr=sr, x_axis='time', y_axis='log')
    # plt.colorbar(format='%+2.0f dB')
    # plt.title('Spectrogram')
    # plt.show()
    # plt.clf()

    # Mel transform
    mel_spect = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=2048, hop_length=512, htk=True)
    mel_spect = librosa.power_to_db(mel_spect, ref=np.max)

    print(len(mel_spect))
    librosa.display.specshow(mel_spect, y_axis='mel', fmax=8000, x_axis='time', n_fft=1024, hop_length=512)
    result, freq_array = sgs.api.calculate_feature_classification(parselmouth.Sound(f))

    pitch_array = freq_array[:, 0]
    # x_len = len(pitch_array) / len(mel_spect)
    # x = np.arange(len(mel_spect))
    # y = []
    # for x in range(len(mel_spect) // 2):
    #     y.append(float(np.mean(pitch_array[int(x_len * x):int(x_len * (x + 1))])))
    # print(len(y))
    x = np.linspace(0, 4.1)
    print(x)
    x_len = len(pitch_array) / len(x)
    y = []
    for a in range(len(x)):
        y.append(np.mean(pitch_array[int(x_len * a):int(x_len * (a + 1))]))

    plt.plot(x, y, color='#7bff4f')
    plt.plot(x, [100] * len(x), color='#7bff4f')
    plt.yticks([0,100,200,300,400,500,600,700,800,900,1000,1200,1400,1600])
    plt.title('Mel Spectrogram')
    plt.colorbar(format='%+2.0f dB')
    plt.show()
    plt.clf()