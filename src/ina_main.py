from __future__ import annotations

import io
import os
import shutil
import sys
import tempfile
import time
import wave
from PIL import Image
from subprocess import Popen, PIPE
from typing import NamedTuple, Callable

import matplotlib.pyplot as plt
import numpy as np
import scipy.io.wavfile
from matplotlib.figure import Figure, Axes
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
        results.append(Result([ResultFrame(*s) for s in lseg], inp[len(lmsg) - 1]))

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

        shutil.copy2(tmp_wav, './test_leohearts.wav')

        return callback(tmp_wav)


def show_image_buffer(buf):
    im = Image.open(buf)
    im.show()
    buf.close()


def draw_result(file: str, result: Result):
    """
    Draw segmentation result

    :param file: Audio file
    :param result: Segmentation result
    :return: Result image in bytes (please close it after use)
    """
    def wav_callback(wavfile: str):
        sample_rate, audio = scipy.io.wavfile.read(wavfile)
        _time = np.linspace(0, len(audio) / sample_rate, num=len(audio))

        fig: Figure = plt.gcf()
        ax: Axes = plt.gca()

        # Plot audio
        plt.plot(_time, audio, color='white')

        # Set size
        # fig.set_dpi(400)
        fig.set_size_inches(18, 6)

        # Cutoff frequency so that the plot looks centered
        cutoff = min(abs(min(audio)), abs(max(audio)))
        ax.set_ylim([-cutoff, cutoff])
        ax.set_xlim([result.frames[0].start, result.frames[-1].end])

        # Draw segmentation areas
        colors = {'female': '#F5A9B8', 'male': '#5BCEFA', 'default': 'gray'}
        for r in result.frames:
            color = colors[r.gender] if r.gender in colors else colors['default']
            ax.axvspan(r.start, r.end - 0.01, alpha=.5, color=color)

        # Savefig to bytes
        buf = io.BytesIO()
        plt.axis('off')
        plt.savefig(buf, bbox_inches='tight', pad_inches=0, transparent=False)
        buf.seek(0)
        plt.clf()
        plt.close()
        return buf

    return to_wav(file, wav_callback)


def get_result_percentages(result: Result) -> tuple[float, float, float]:
    """
    Get percentages

    :param result: Result
    :return: %female, %male, %other
    """
    # Count total and categorical durations
    total_dur = 0
    durations: dict[str, int] = {f.gender: 0 for f in result.frames}
    for f in result.frames:
        dur = f.end - f.start
        durations[f.gender] += dur
        total_dur += dur

    # Convert durations to ratios
    for d in durations:
        durations[d] /= total_dur

    # Return results
    f = durations.get('female', 0)
    m = durations.get('male', 0)
    return f, m, 1 - f - m


# def test():
#     results: BatchResults = BatchResults(
#         [Result([ResultFrame('female', 0.0, 10.48), ResultFrame('male', 10.48, 12.780000000000001)],
#                 '../test.csv')],
#         1.7032792568206787, 1.7032792568206787, 1,
#         [('../test.csv', 0)])
#
#     with draw_result('../test.mp3', results.results[0]) as buf:
#         show_image_buffer(buf)
#         print(get_result_percentages(results.results[0]))
#
#     # seg = Segmenter()
#     # print(process(seg, ['../test.mp3']))

if __name__ == '__main__':
    to_wav('../audio_tmp/2021-12-22 05-32 leph1art5.mp3', print)
    # test()
    pass
