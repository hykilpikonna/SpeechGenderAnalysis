import json
import os
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pygame
from inaSpeechSegmenter import Segmenter

from ina_main import process, get_result_percentages


def segment_all():
    # Create segmenter
    seg = Segmenter()
    np.seterr(invalid='ignore')

    # Loop through all celebrities
    for id in ids:
        id_dir = data_dir.joinpath(id)

        # Loop through all recordings (Exclude singing for now)
        utters = [r for r in os.listdir(id_dir) if r.endswith('.flac')
                  and not r.startswith('singing')]

        # Exclude existing
        utters = [id_dir.joinpath(u) for u in utters]
        utters = [u for u in utters if not u.with_suffix('.json').exists()]

        if len(utters) == 0:
            continue

        # Analyze
        results = process(seg, [str(u) for u in utters], verbose=True)

        # Write results
        total = [0, 0, 0, 0, 0]
        type_totals = {}
        for result in results.results:
            file = Path(result.file).with_suffix('.json')

            # Get results
            # f: Frames, r: Ratios
            ratios = [round(r, 3) for r in get_result_percentages(result)]
            stored = {'f': result.frames, 'r': ratios}

            # Count type total (type_totals[utter_type][-1] is the count)
            file_name = file.name
            utter_type = file_name[:file_name.index('-')]
            type_totals.setdefault(utter_type, [0, 0, 0, 0, 0])
            for i in range(4):
                type_totals[utter_type][i] += ratios[i]
                total[i] += ratios[i]
            type_totals[utter_type][-1] += 1
            total[-1] += 1

            # Write result
            file.write_text(json.dumps(stored))

        # Write type averages
        type_averages = {t: [r / type_totals[t][-1] for r in type_totals[t][:-1]] for t in type_totals}
        total_average = [r / total[-1] for r in total[:-1]]
        obj = {'type_averages': type_averages, 'total_averages': total_average}
        id_dir.joinpath('total.json').write_text(json.dumps(obj))


def graph_histogram():
    closest_to_half = 1000
    closest_to_half_id = ''
    id_pf_map = {}
    for id in ids:
        id_dir = data_dir.joinpath(id)
        json_path = id_dir.joinpath('total.json')

        if not json_path.exists():
            continue

        obj = json.loads(json_path.read_text())
        f, m, o, pf = obj['total_averages']

        # Recalculate pf (pf is actually calculated incorrectly)
        if f + m == 0:
            continue
        pf = f / (f + m)
        id_pf_map[id] = pf

        # Save fixed json
        obj['total_averages'][3] = pf
        json_path.write_text(json.dumps(obj))

        # Find pf closest to .5
        dist = abs(pf - .5)
        if dist < closest_to_half:
            closest_to_half = dist
            closest_to_half_id = id

    data_dir.joinpath('id_pf_map.json').write_text(json.dumps(id_pf_map))
    plt.hist(id_pf_map.values(), bins=50)
    plt.show()
    print(closest_to_half_id)


def manually_label_data():
    """
    Since CN-Celeb isn't labelled with the speaker's gender, this script is used to manually label
    them.
    """
    pygame.mixer.init()

    # Loop through all speaker
    id_labels = {}
    for id_i, id in enumerate(ids):
        id_dir = data_dir.joinpath(id)

        # Loop through all tracks until identified
        tracks = [f for f in os.listdir(id_dir) if f.endswith('.flac')]
        for track_i, audio in enumerate(tracks):
            # Play track
            sound = pygame.mixer.Sound(id_dir.joinpath(audio))
            sound.play()
            i = input(f'Playing speaker {id_i}/{len(ids)} - track {track_i}/{len(tracks)}\n'
                      f'- Identify gender. Press f / m, or anything else to play next track: ')\
                .lower().strip()
            sound.stop()

            # Labeled
            if i == 'f' or i == 'm':
                id_labels[id] = i
                data_dir.joinpath('id_labels.json').write_text(json.dumps(id_labels))
                break


if __name__ == '__main__':
    cn_celeb_root = Path('C:/Users/me/Workspace/Data/CN-Celeb_flac')
    data_dir = cn_celeb_root.joinpath('data')
    ids = [id for id in os.listdir(data_dir) if id.startswith('id')]

    # segment_all()
    # graph_histogram()
    manually_label_data()
