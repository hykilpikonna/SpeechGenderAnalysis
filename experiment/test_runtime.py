import librosa as librosa
import numpy as np
import parselmouth
import torchaudio
from inaSpeechSegmenter import Segmenter
from inaSpeechSegmenter.features import to_wav, _wav2feats
from inaSpeechSegmenter.sidekit_mfcc import read_wav

from server.utils import Timer

np.seterr(invalid='ignore')

if __name__ == '__main__':
    f = '/workspace/EECS 6414/voice_cnn/VT 150hz baseline example.mp3'
    timer = Timer()

    seg = Segmenter()
    seg('/workspace/EECS 6414/voice_cnn/test.wav')
    timer.log('ML engine loaded (One time expense, not counted in running time).')

    print()

    fp = f
    fp = str(to_wav(f).absolute())
    timer.log('FFMPEG Convert file to WAV 16000Hz')
    fp = str(to_wav(f, sr=None).absolute())
    timer.log('FFMPEG Convert file to WAV (original sr kept)')

    print()

    # Read file
    parselmouth.Sound(fp)
    timer.log('Parselmouth: Read file.')
    sound = parselmouth.Sound(fp)
    timer.log('Parselmouth: Read file.')

    # librosa.load(fp)
    # timer.log('Librosa: Read file')
    # librosa.load(fp)
    # timer.log('Librosa: Read file')

    read_wav(fp)
    timer.log('Read file with read_wav')
    y, sr, _ = read_wav(fp)
    timer.log(f'Read file with read_wav (decoded sr = {sr})')

    torchaudio.load(fp)
    timer.log('Read file with torchaudio')
    torchaudio.load(fp)
    timer.log('Read file with torchaudio')

    print()

    # Calculate features
    pitch = sound.to_pitch(0.01)
    timer.log('Parselmouth: Pitch calculated (0.01)')
    sound.to_pitch(0.01)
    timer.log('Parselmouth: Pitch calculated again (0.01)')
    sound.to_pitch(0.032)
    timer.log('Parselmouth: Pitch calculated again (0.032)')

    print()

    sound.to_formant_burg(0.01)
    timer.log('Parselmouth: Formant calculated (0.01)')
    sound.to_formant_burg(0.01)
    timer.log('Parselmouth: Formant calculated again (0.01)')
    sound.to_formant_burg(0.032)
    timer.log('Parselmouth: Formant calculated again (0.032)')

    print()

    sound.to_spectrogram(window_length=0.128, time_step=0.032)
    timer.log('Parselmouth: Spectrogram calculated (n_fft=0.128, step=0.032)')

    print()

    librosa.core.piptrack(y=y, sr=sr)
    timer.log('Librosa: piptrack')
    spec = np.abs(librosa.stft(y, n_fft=1024, hop_length=512))
    timer.log('Librosa: STFT')
    mel_spect = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=2048, hop_length=512, htk=True)
    timer.log('Librosa: Mel spectrogram')

    print()

    _wav2feats(fp)
    timer.log('ML: Calculate mspect feats')
    mspect, loge, diff_len = _wav2feats(fp)
    timer.log('ML: Calculate mspect feats')

    seg.segment_feats(mspect, loge, diff_len, 0)
    timer.log('ML: Segment feats')
    seg.segment_feats(mspect, loge, diff_len, 0)
    timer.log('ML: Segment feats')

    # Calculate ML
    # seg(f)
    # timer.log('ML Segmented')
    # seg(f)
    # timer.log('ML Segmented again')
