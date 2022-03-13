from __future__ import annotations

import csv
import os
from dataclasses import dataclass
from multiprocessing import Pool
from os import PathLike
from pathlib import Path
from typing import Iterable

import jsonpickle as jsonpickle
import matplotlib.pyplot as plt
import numpy
import numpy as np
import parselmouth
import tqdm


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


def load_vox_celeb_asab_dict(path: PathLike) -> dict[str, str]:
    """
    Load voxCeleb 1 or 2's metadata to gather a dictionary mapping id to assigned sex at birth.

    :param path: CSV path (Tab separated)
    :return: {id: ASAB}
    """
    with open(path, 'r', newline='') as f:
        return {row[0]: row[2] for row in csv.reader(f, delimiter='\t') if row[0].startswith('id')}


def loop_id_dirs() -> Iterable[Path]:
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


def compute_vox_celeb_helper(aud_dir: str):
    """
    Compute one audio file

    :param aud_dir: Audio file path
    :return: None
    """
    array = calculate_freq_info(parselmouth.Sound(aud_dir))
    numpy.save(aud_dir, array)


def compute_vox_celeb():
    print('Finding audio files...')
    queue: list[str] = []

    # Loop through all ids
    for id, id_dir in loop_id_dirs():
        queue += get_audio_paths(id_dir)

    print(f'There are {len(queue)} audio files to process.')
    print('Starting processing...')

    # Compute audio files in a cpu pool
    with Pool(8) as pool:
        for _ in tqdm.tqdm(pool.imap(compute_vox_celeb_helper, queue), total=len(queue)):
            pass


@dataclass
class FrequencyStats:
    pitch: Statistics
    f1: Statistics
    f2: Statistics
    f3: Statistics
    f1ratio: Statistics
    f2ratio: Statistics
    f3ratio: Statistics


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


def calculate_statistics(arr: np.ndarray) -> FrequencyStats:
    """
    Calculate frequency data array statistics

    :param arr: n-by-4 Array from calculate_freq_info
    :return: Statistics
    """
    def calc_col_stats(col: np.ndarray) -> Statistics:
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
            len(arr)
        )

    result = [calc_col_stats(arr[:, i]) for i in range(0, 4)] + \
             [calc_col_stats(np.divide(arr[:, i], arr[:, 0])) for i in range(1, 4)]

    return FrequencyStats(*result)


def vox_celeb_statistics_helper(id_dir: Path):
    # Load all files
    cumulative: np.ndarray = np.concatenate([np.load(f) for f in get_audio_paths(id_dir, 'npy')])

    # Remove out NaN values
    cumulative = cumulative[~np.isnan(cumulative).any(axis=1), :]
    result = calculate_statistics(cumulative)

    # Write results
    with open(f'{id_dir}/stats.json', 'w') as jsonfile:
        jsonfile.write(jsonpickle.encode(result, jsonfile, indent=1))


def vox_celeb_statistics():
    id_dirs = [id_dir for id, id_dir in loop_id_dirs()]

    # Loop through all ids
    with Pool(8) as pool:
        for _ in tqdm.tqdm(pool.imap(vox_celeb_statistics_helper, id_dirs), total=len(id_dirs)):
            pass


if __name__ == '__main__':
    vox_celeb_dir = Path('C:/Workspace/EECS 6414/Datasets/VoxCeleb1/wav')
    agab = load_vox_celeb_asab_dict(vox_celeb_dir.joinpath('../vox1_meta.csv'))

    # print(calculate_freq_info(parselmouth.Sound('../00001.wav')))
    vox_celeb_statistics()
    # print(calculate_freq_info(parselmouth.Sound('D:/Downloads/Vowels-Extract-Z-44kHz.flac')))
    # print(calculate_freq_info(parselmouth.Sound('D:/Downloads/Vowels-Azalea.flac')))
