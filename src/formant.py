from __future__ import annotations

import csv
import os
from multiprocessing import Pool
from os import PathLike
from pathlib import Path

import tqdm
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


def load_vox_celeb_asab_dict(path: PathLike) -> dict[str, str]:
    """
    Load voxCeleb 1 or 2's metadata to gather a dictionary mapping id to assigned sex at birth.

    :param path: CSV path (Tab separated)
    :return: {id: ASAB}
    """
    with open(path, 'r', newline='') as f:
        return {row[0]: row[2] for row in csv.reader(f, delimiter='\t') if row[0].startswith('id')}


def compute_vox_celeb_helper(aud_dir: str):
    """
    Compute one audio file

    :param aud_dir: Audio file path
    :return: None
    """
    array = calculate_freq_info(parselmouth.Sound(aud_dir))
    numpy.save(aud_dir, array)


def compute_vox_celeb():
    vox_celeb_dir = Path('C:/Workspace/EECS 6414/Datasets/VoxCeleb1/wav')
    audio_suffix = 'wav'

    print('Finding audio files...')

    asab = load_vox_celeb_asab_dict(vox_celeb_dir.joinpath('../vox1_meta.csv'))
    queue: list[str] = []

    # Loop through all ids
    for id in asab:
        id_dir = vox_celeb_dir.joinpath(id)

        # Check if directory exists
        if not id_dir.is_dir():
            continue

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

                # Add to queue
                queue.append(str(aud_dir))

    print(f'There are {len(queue)} audio files to process.')
    print('Starting processing...')

    # Compute audio files in a cpu pool
    with Pool(8) as pool:
        for _ in tqdm.tqdm(pool.imap(compute_vox_celeb_helper, queue), total=len(queue)):
            pass


if __name__ == '__main__':
    compute_vox_celeb()
    # print(calculate_freq_info(parselmouth.Sound('D:/Downloads/Vowels-Extract-Z-44kHz.flac')))
    # print(calculate_freq_info(parselmouth.Sound('D:/Downloads/Vowels-Azalea.flac')))

