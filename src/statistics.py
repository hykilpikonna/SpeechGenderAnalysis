from __future__ import annotations

import csv
import json
import os
from dataclasses import dataclass
from json import JSONDecodeError
from multiprocessing import Pool
from os import PathLike
from pathlib import Path
from typing import Iterable, Literal, Callable
import jsonpickle as jsonpickle
import matplotlib.pyplot as plt
import numpy
import numpy as np
import pandas as pd
import parselmouth
import tqdm
import seaborn as sns
from matplotlib.patches import Patch

from spectral_tilt import tilt

ASAB = Literal['f', 'm']
COLOR_PINK = '#F5A9B8'
COLOR_BLUE = '#5BCEFA'
CPU_CORES = 36


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


def load_vox_celeb_asab_dict(path: PathLike) -> dict[str, ASAB]:
    """
    Load voxCeleb 1 or 2's metadata to gather a dictionary mapping id to assigned sex at birth.

    :param path: CSV path (Tab separated)
    :return: {id: ASAB}
    """
    with open(path, 'r', newline='') as f:
        return {row[0]: row[2] for row in csv.reader(f, delimiter='\t') if row[0].startswith('id')}


def loop_id_dirs() -> Iterable[tuple[str, Path]]:
    # Loop through all ids
    for id in agab:
        id_dir = vox_celeb_dir.joinpath(id)

        # Check if directory exists
        if not id_dir.is_dir():
            continue

        yield id, id_dir


def get_audio_paths(id_dir: Path, audio_suffix: str = 'wav') -> list[str]:
    """
    Get all audio paths under one person

    :param id_dir: Person ID directory
    :param audio_suffix: Select only files with this suffix
    :return: audio paths
    """
    audios = []

    # Loop through all videos
    for vid in os.listdir(id_dir):
        vid_dir = id_dir.joinpath(vid)

        # Check if it's a video directory
        if not vid_dir.is_dir():
            continue

        # Loop through all audios
        for aud in os.listdir(vid_dir):
            aud_dir = vid_dir.joinpath(aud)

            # Check if end with suffix
            if not aud.endswith(audio_suffix):
                continue

            # Add
            audios.append(str(aud_dir))

    return audios


def compute_audio_freq(aud_dir: str):
    """
    Compute and save the frequency info of one audio file
    """
    array = calculate_freq_info(parselmouth.Sound(aud_dir))
    numpy.save(aud_dir, array)


def compute_audio_tilt(aud_dir: str):
    """
    Compute and save the tilt info of one audio file
    """
    spectral_tilt = tilt(parselmouth.Sound(aud_dir))
    with open(Path(aud_dir).with_suffix('.json'), 'w', encoding='utf-8') as f:
        json.dump({'tilt': spectral_tilt}, f)


def compute_audio_vox_celeb(func: Callable[[str], None]) -> None:
    """
    Compute a function for each audio file in the vox celeb dataset

    :param func: The function to compute - func(aud_dir) -> None
    """
    print('Finding audio files...')
    queue: list[str] = []

    # Loop through all ids
    for id, id_dir in loop_id_dirs():
        queue += get_audio_paths(id_dir)

    print(f'There are {len(queue)} audio files to process.')
    print('Starting processing...')

    # Compute audio files in a cpu pool
    with Pool(CPU_CORES) as pool:
        for _ in tqdm.tqdm(pool.imap(func, queue), total=len(queue)):
            pass


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


def combine_id_freq(id_dir: Path):
    """
    Combine frequency data of all audio files under one person
    """
    # Load all files
    cumulative: np.ndarray = np.concatenate([np.load(f) for f in get_audio_paths(id_dir, 'npy')])

    # Remove out NaN values
    cumulative = cumulative[~np.isnan(cumulative).any(axis=1), :]
    result = calculate_freq_statistics(cumulative)

    # Write results
    with open(id_dir.joinpath('stats.json'), 'w') as jsonfile:
        jsonfile.write(jsonpickle.encode(result, jsonfile, indent=1))


def combine_id_tilt(id_dir: Path):
    """
    Combine tilt data of all audio files under one person
    """
    # Load all calculated files
    cumulative = []
    for f in get_audio_paths(id_dir, 'json'):
        try:
            cumulative.append(json.loads(Path(f).read_text('utf-8'))['tilt'])
        except JSONDecodeError:
            print(f'Error in {f}')

    # Remove out NaN values
    cumulative = [c for c in cumulative if c is not None]
    result = calc_col_stats(np.array(cumulative))

    # Write results
    with open(id_dir.joinpath('tilt.json'), 'w') as jsonfile:
        jsonfile.write(jsonpickle.encode(result, jsonfile, indent=1))


def call_id_vox_celeb(func: Callable[[Path], None]) -> None:
    """
    Call a function for each person's id in the vox celeb dataset.

    :param func: func(id_dir) -> None
    """
    id_dirs = [id_dir for id, id_dir in loop_id_dirs()]

    # Loop through all ids
    with Pool(CPU_CORES) as pool:
        for _ in tqdm.tqdm(pool.imap(func, id_dirs), total=len(id_dirs)):
            pass


def subplots(**kwargs) -> tuple[plt.Figure, plt.Axes]:
    return plt.subplots(**kwargs)


def collect_visualize_freq():
    """
    Collect statistics and draw interesting visualizations from its results
    """
    # Read stats
    stats_list: list[tuple[FrequencyStats, ASAB]] = []
    for id, id_dir in loop_id_dirs():
        stats_dir = id_dir.joinpath('stats.json')
        if not stats_dir.is_file():
            continue
        stats_list.append((jsonpickle.decode(stats_dir.read_text()), agab[id]))

    # Get AFAB and AMAB means
    headers = ['Pitch\n(Fundamental\nFrequency)', 'Formant F1', 'Formant F2', 'Formant F3', 'F1 Ratio', 'F2 Ratio', 'F3 Ratio']
    f_means = np.array([[t.mean for t in [s.pitch, s.f1, s.f2, s.f3, s.f1ratio, s.f2ratio, s.f3ratio]]
                        for s, ag in stats_list if ag == 'f'])
    m_means = np.array([[t.mean for t in [s.pitch, s.f1, s.f2, s.f3, s.f1ratio, s.f2ratio, s.f3ratio]]
                        for s, ag in stats_list if ag == 'm'])

    # Plot bar chart
    sns.set_theme(style="ticks")
    fig, ax = subplots(figsize=(10, 5))

    df = pd.DataFrame({headers[i]: f_means[:, i] for i in range(4)})
    dm = pd.DataFrame({headers[i]: m_means[:, i] for i in range(4)})
    args = dict(orient='h', scale='width', inner='quartile', linewidth=0.5)
    sns.histplot()
    sns.violinplot(data=df, color=COLOR_PINK, **args)
    sns.violinplot(data=dm, color=COLOR_BLUE, **args)
    [c.set_alpha(0.7) for c in ax.collections]

    # Create legend
    legend_elements = [
        Patch(facecolor=COLOR_PINK, edgecolor='r', label='Feminine'),
        Patch(facecolor=COLOR_BLUE, edgecolor='b', label='Masculine'),
    ]
    plt.legend(handles=legend_elements)

    ax.set_title("Distribution of Pitch and Formant Based on Gender")
    ax.xaxis.grid(True)
    ax.set_ylabel('')
    ax.set_xlabel('Frequency (Hz)')
    sns.despine(fig, ax)
    plt.show()

    # Write JSON
    data = {val: {'f': f_means[:, i].tolist(), 'm': m_means[:, i].tolist()} for i, val in enumerate(['Pitch', 'F1', 'F2', 'F3'])}
    Path('results/frequency-data.json').write_text(json.dumps(data), 'utf-8')


def collect_visualize_tilt():
    """
    Collect statistics and draw interesting visualizations from its results
    """
    # Read stats
    stats_list: list[tuple[Statistics, ASAB]] = []
    for id, id_dir in loop_id_dirs():
        stats_dir = id_dir.joinpath('tilt.json')
        if not stats_dir.is_file():
            continue
        stats_list.append((jsonpickle.decode(stats_dir.read_text()), agab[id]))

    # Get AFAB and AMAB means
    f_means = np.array([s.mean for s, ag in stats_list if ag == 'f'])
    m_means = np.array([s.mean for s, ag in stats_list if ag == 'm'])

    # Plot bar chart
    sns.set_theme(style="ticks")
    fig, ax = subplots(figsize=(10, 5))

    df = pd.DataFrame({"Tilt": f_means})
    dm = pd.DataFrame({"Tilt": m_means})
    args = dict(orient='h', scale='width', inner='quartile', linewidth=0.5)
    sns.violinplot(data=df, color=COLOR_PINK, **args)
    sns.violinplot(data=dm, color=COLOR_BLUE, **args)
    [c.set_alpha(0.7) for c in ax.collections]

    # Create legend
    legend_elements = [
        Patch(facecolor=COLOR_PINK, edgecolor='r', label='Feminine'),
        Patch(facecolor=COLOR_BLUE, edgecolor='b', label='Masculine'),
    ]
    plt.legend(handles=legend_elements)

    ax.set_title("Distribution of Spectral Tilt on Gender")
    ax.xaxis.grid(True)
    ax.set_ylabel('')
    ax.set_xlabel('Tilt Value')
    sns.despine(fig, ax)
    plt.show()

    # Write JSON
    data = {'f': f_means.tolist(), 'm': m_means.tolist()}
    Path('results/tilt-data.json').write_text(json.dumps(data), 'utf-8')


if __name__ == '__main__':
    vox_celeb_dir = Path('../Datasets/VoxCeleb1/wav')
    agab = load_vox_celeb_asab_dict(vox_celeb_dir.joinpath('../vox1_meta.csv'))

    ############
    # 1. Compute and save all the frequency (pitch, f0, f1, f2) for vox1
    #    For each audio, a file <audio-name>.npy will be saved, with each row representing 10ms data
    # compute_audio_vox_celeb(compute_audio_freq)

    # 2. Combine and save statistics for each person in vox1
    #    For each person, stats.json will be saved, containing statistics of all of their audios
    # call_id_vox_celeb(combine_id_freq)

    # 3. Collect statistics and draw visualizations
    collect_visualize_freq()

    ###########
    # 1. Compute and save all the spectral tilt for vox1
    #    For each audio, a file <audio-name>.json will be saved with tilt value in it
    # compute_audio_vox_celeb(compute_audio_tilt)

    # 2. Combine statistics for each person in vox1
    # call_id_vox_celeb(combine_id_tilt)

    # 3. Collect statistics and draw visualizations
    collect_visualize_tilt()

    # print(calculate_freq_info(parselmouth.Sound('../00001.wav')))
    # print(calculate_freq_info(parselmouth.Sound('D:/Downloads/Vowels-Extract-Z-44kHz.flac')))
    # print(calculate_freq_info(parselmouth.Sound('D:/Downloads/Vowels-Azalea.flac')))
