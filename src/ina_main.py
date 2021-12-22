from __future__ import annotations
import os
import sys
import tempfile
import time
import wave
from subprocess import Popen, PIPE
from typing import NamedTuple, Callable

import matplotlib.pyplot as plt
import numpy as np
import scipy.io.wavfile
from matplotlib.figure import Figure
from numpy import ndarray

os.environ['KERAS_BACKEND'] = "plaidml.keras.backend"

import keras
from keras import backend

import tensorflow as tf
from inaSpeechSegmenter import *
from inaSpeechSegmenter.segmenter import featGenerator


class ResultFrame(NamedTuple):
    gender: str
    start: float
    end: float


class Result(NamedTuple):
    frames: list[ResultFrame]
    file: str


class BatchResults(NamedTuple):
    results: list[Result]
    time_full: float
    time_avg: float
    successes: int
    messages: list[tuple[str, int]]


def process(self: Segmenter, inp: list[str], tmpdir=None, verbose=False, skip_if_exist=False,
            nbtry=1, try_delay=2.) -> BatchResults:
    t_batch_start = time.time()

    results: list[Result] = []
    lmsg = []
    fg = featGenerator(inp.copy(), inp.copy(), tmpdir, self.ffmpeg, skip_if_exist, nbtry, try_delay)
    i = 0
    for feats, msg in fg:
        lmsg += msg
        i += len(msg)
        if verbose:
            print('%d/%d' % (i, len(inp)), msg)
        if feats is None:
            break
        mspec, loge, diff_len = feats
        lseg = self.segment_feats(mspec, loge, diff_len, 0)
        results.append(Result(lseg, inp[len(lmsg) - 1]))

    t_batch_dur = time.time() - t_batch_start
    nb_processed = len([e for e in lmsg if e[1] == 0])
    avg = t_batch_dur / nb_processed if nb_processed else -1
    return BatchResults(results, t_batch_dur, avg, nb_processed, lmsg)


def to_wav(file: str, callback: Callable, start_sec: float = 0, stop_sec: float = 0):
    """
    Convert media to temp wav 16k file and return features
    """
    base, _ = os.path.splitext(os.path.basename(file))

    with tempfile.TemporaryDirectory() as tmpdir_name:
        # build ffmpeg command line
        tmp_wav = tmpdir_name + '/' + base + '.wav'
        args = ['ffmpeg', '-y', '-i', file, '-ar', '16000', '-ac', '1']

        if start_sec != 0:
            args += ['-ss', '%f' % start_sec]
        if stop_sec != 0:
            args += ['-to', '%f' % stop_sec]

        args += [tmp_wav]

        # launch ffmpeg
        p = Popen(args, stdout=PIPE, stderr=PIPE)
        output, error = p.communicate()
        assert p.returncode == 0, error

        return callback(tmp_wav)


if __name__ == '__main__':
    args = sys.argv[1:]
    if len(args) < 0:
        exit(0)


    # inp = args[0]

    def wav_callback(wavfile: str):
        sample_rate, audio = scipy.io.wavfile.read(wavfile)
        _time = np.linspace(0, len(audio) / sample_rate, num=len(audio))

        cutoff = min(abs(min(audio)), abs(max(audio)))

        plt.plot(_time, audio)
        # plt.ylabel("Amplitude")
        # plt.xlabel("Time")
        # plt.title("Sample Wav")

        fig: Figure = plt.gcf()
        fig.set_size_inches(10, 1)
        fig.set_dpi(200)

        ax = plt.gca()
        ax.set_ylim([-cutoff, cutoff])

        plt.axis('off')
        plt.savefig('../image.png', bbox_inches='tight', pad_inches=0, transparent=True)


    to_wav('../test.mp3', wav_callback)

    # seg = Segmenter()
    # print(process(seg, ['../test.mp3']))
    a = (
        [([('female', 0.0, 10.48), ('male', 10.48, 12.780000000000001)], '../test.csv')],
        1.7032792568206787, 1, 1.7032792568206787,
        [('../test.csv', 0, 'ok 1.4748258590698242')])
