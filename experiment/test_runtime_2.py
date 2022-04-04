import base64
import zlib
from multiprocessing import cpu_count
from subprocess import Popen, PIPE

import librosa
import numpy as np
import pandas as pd
import parselmouth
import tensorflow
import torch
import torchaudio
from inaSpeechSegmenter.features import to_wav
from inaSpeechSegmenter.sidekit_mfcc import read_wav, mfcc
import tensorflow as tf
import tensorflow_io as tfio
from tqdm.contrib.concurrent import process_map

from server.utils import Timer


def test_readfile(file: str, iterations: int):
    results = []
    timer = Timer()
    for _ in range(iterations):
        result = []
        results.append(result)

        parselmouth.Sound(file)
        result.append(timer.elapsed())

        librosa.load(file)
        result.append(timer.elapsed())

        read_wav(file)
        result.append(timer.elapsed())

        torchaudio.load(file)
        result.append(timer.elapsed())
    return pd.DataFrame(results, columns=['Parselmouth', 'librosa', 'read_wav', 'torchaudio'])


def test_resampling(file: str, iterations: int, resample: bool):
    results = []
    timer = Timer()
    sr = 16000 if resample else None
    for _ in range(iterations):
        result = []
        results.append(result)

        # FFMPEG
        to_wav(file, sr=sr)
        result.append(timer.elapsed())

        # SOX
        args = ['sox', file, '-c', '1', '-e', 'floating-point']
        if sr:
            args += ['-r', str(sr)]
        args += ['output-sox.wav']
        p = Popen(args, stdout=PIPE, stderr=PIPE)
        output, error = p.communicate()
        assert p.returncode == 0, error
        result.append(timer.elapsed())

        # MPlayer
        args = ['mplayer', '-ao', 'pcm:fast:waveheader:file=output-mplayer.wav', '-vo', 'null', '-vc', 'null']
        if sr:
            args += ['-af', f'resample={sr},pan=1:0.5:0.5']
        else:
            args += ['-af', 'pan=1:0.5:0.5']
        args += [file]
        p = Popen(args, stdout=PIPE, stderr=PIPE)
        output, error = p.communicate()
        assert p.returncode == 0, error
        result.append(timer.elapsed())

    return pd.DataFrame(results, columns=['ffmpeg', 'sox', 'mplayer'])


def test_spectrogram(y: np.ndarray, sr: int, iterations: int, n_fft=2048, hop_length=512):
    results = []
    timer = Timer()
    nfft_s = n_fft / sr
    step_s = hop_length / sr
    for _ in range(iterations):
        result = []
        results.append(result)

        sound = parselmouth.Sound(y, float(sr))
        sound.to_spectrogram(window_length=nfft_s, time_step=step_s)
        result.append(timer.elapsed())

        librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, htk=True)
        result.append(timer.elapsed())

        mfcc(y.astype(np.float32), get_mspec=True, nwin=nfft_s, shift=step_s, fs=sr)
        result.append(timer.elapsed())

        t = torchaudio.transforms.MelSpectrogram(sample_rate=sr, n_fft=n_fft, hop_length=hop_length)
        tensor = torch.from_numpy(y)
        t(tensor)
        result.append(timer.elapsed())

        t = tfio.audio.spectrogram(y, n_fft, n_fft, hop_length)
        mel_spectrogram = tfio.audio.melscale(t, rate=sr, mels=128, fmin=0, fmax=8000)
        result.append(timer.elapsed())

    return pd.DataFrame(results, columns=['Parselmouth', 'librosa', 'sidekit', 'torchaudio', 'tensorflow-io'])


def test_pitch(y: np.ndarray, sr: int, iterations: int, n_fft=2048, hop_length=512):
    results = []
    timer = Timer()
    nfft_s = n_fft / sr
    step_s = hop_length / sr
    for _ in range(iterations):
        result = []
        results.append(result)

        sound = parselmouth.Sound(y, float(sr))
        sound.to_pitch(time_step=step_s)
        result.append(timer.elapsed())

        librosa.yin(y=y, sr=sr, frame_length=n_fft, hop_length=hop_length, fmin=75, fmax=600)
        result.append(timer.elapsed())

        librosa.pyin(y=y, sr=sr, frame_length=n_fft, hop_length=hop_length, fmin=75, fmax=600)
        result.append(timer.elapsed())

        # TODO: essentia (yin, pyin), in-formant (yin, mpm, rapt, irapt)

    return pd.DataFrame(results, columns=['Parselmouth (Boersma 1993)', 'librosa.yin (Kawahara 2002)',
                                          'librosa.pyin (Mauch 2014)'])


def test_formant(y: np.ndarray, sr: int, iterations: int, n_fft=2048, hop_length=512):
    results = []
    timer = Timer()
    nfft_s = n_fft / sr
    step_s = hop_length / sr
    for _ in range(iterations):
        result = []
        results.append(result)

        sound = parselmouth.Sound(y, float(sr))
        sound.to_formant_burg(time_step=step_s)
        result.append(timer.elapsed())

        # TODO: in-formant (deepformants, filteredlp, simplelp, karma)

    return pd.DataFrame(results, columns=['Parselmouth (Marple 1980)'])


def _formant(args: tuple[np.ndarray, float]):
    y, sr = args
    sound = parselmouth.Sound(y, sr)
    step = 512 / sr
    formant = sound.to_formant_burg(time_step=512 / sr)
    result = np.ndarray([len(formant), 3], 'float32')
    for i in range(len(formant)):
        for f in range(1, 4):
            result[i][f - 1] = formant.get_value_at_time(f, i * step)
    return result


if __name__ == '__main__':
    f = '/workspace/EECS 6414/voice_cnn/VT 150hz baseline example.mp3'
    fp = str(to_wav(f, sr=16000).absolute())

    # print(read_wav(f))

    # Test readfile
    # df = test_readfile(fp, 10)
    # print(df)
    # print(df.mean())

    # Test resampling
    # df = test_resampling(f, 10, True)
    # print(df)
    # print(df.mean())

    y, sr, _ = read_wav(fp)
    #
    # # Tensorflow warm-up
    # t = tfio.audio.spectrogram(y, 1, 1, 2048)
    # tfio.audio.melscale(t, rate=sr, mels=128, fmin=0, fmax=8000)
    # print('Warmup done')
    #
    # # Test mel spect
    # df = test_spectrogram(y, sr, 10)
    # print(df)
    # print(df.mean())

    # Test pitch
    # df = test_pitch(y, sr, 10)
    # print(df)
    # print(df.mean())

    # Test formant
    # df = test_formant(y, sr, 10)
    # print(df)
    # print(df.mean())
    # timer = Timer()
    # split = [(y, float(sr)) for y in np.array_split(y, 512 * 30)]
    # print(split)
    # print(len(split))
    # formants = process_map(_formant, split, max_workers=cpu_count(), chunksize=1)
    # timer.log('Done')
    # print(formants)
    # sound = parselmouth.Sound(y, float(sr))
    # formant = sound.to_formant_burg(time_step=512 / sr)

    # sound.to_formant_burg()

    n_fft = 2048
    hop_length = 512
    t = tfio.audio.spectrogram(y, n_fft, n_fft, hop_length)
    mel_spectrogram: tf.Tensor = tfio.audio.melscale(t, rate=sr, mels=128, fmin=0, fmax=8000)
    nd: np.ndarray = mel_spectrogram.numpy()
    print(nd)
    print(nd.shape)
    print(nd.dtype)
    by = nd.tobytes()
    print('Raw Numpy bytes:', type(by), f'{len(by) / 1024 / 1024:.2f}mb')
    zl = zlib.compress(by, 9)
    print('zlib compressed (level 9):', type(zl), f'{len(zl) / 1024 / 1024:.2f}mb')
    b6 = base64.b64encode(by)
    print('base64 encoded utf-8:', type(b6), f'{len(b6) / 1024 / 1024:.2f}mb')

