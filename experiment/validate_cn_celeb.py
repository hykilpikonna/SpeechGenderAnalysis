import json
import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from inaSpeechSegmenter import Segmenter

from ina_main import process, get_result_percentages
from utils import color, printc


gpu_devices = tf.config.experimental.list_physical_devices('GPU')
for device in gpu_devices:
    tf.config.experimental.set_memory_growth(device, True)


def segment_all():
    # Create segmenter
    seg = Segmenter()
    np.seterr(invalid='ignore')

    # Loop through all celebrities
    for id in ids[559:]:
        id_dir = data_dir / id

        if (id_dir / 'total.json').is_file():
            continue

        # Loop through all recordings (Exclude singing for now)
        utters = audio_files[id]

        # Exclude existing
        utters = [id_dir.joinpath(u) for u in utters if u.endswith('.wav')]
        # utters = [u for u in utters if not u.with_suffix('.json').exists()]


        if len(utters) == 0:
            continue

        # Analyze
        print(f'Processing {id}')
        results = process(seg, [str(u) for u in utters], verbose=True)

        # Write results
        # total = [0, 0, 0, 0, 0]
        # type_totals = {}
        total = []
        for result in results:
            file = Path(result.file).with_suffix('.json')

            # Get results
            # f: Frames, r: Ratios
            _, _, _, pf = get_result_percentages(result)
            total.append(pf)

            # Count type total (type_totals[utter_type][-1] is the count)
            # file_name = file.name
            # utter_type = file_name[:file_name.index('-')]
            # type_totals.setdefault(utter_type, [0, 0, 0, 0, 0])
            # for i in range(4):
            #     type_totals[utter_type][i] += ratios[i]
            #     total[i] += ratios[i]
            # type_totals[utter_type][-1] += 1
            # total[-1] += 1

            # Write result
            # file.write_text(json.dumps(ratios))

        # Write type averages
        # type_averages = {t: [r / type_totals[t][-1] for r in type_totals[t][:-1]] for t in type_totals}
        # total_average = [r / total[-1] for r in total[:-1]]
        # obj = {'type_averages': type_averages, 'total_averages': total_average}
        # id_dir.joinpath('total.json').write_text(json.dumps(obj))
        id_dir.joinpath('total.json').write_text(json.dumps({'ratio': np.nanmean(total)}))


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
    # pygame.mixer.init()

    # Load existing labels
    labels_json = data_dir.joinpath('id_labels.json')
    id_labels = json.loads(labels_json.read_text()) if labels_json.exists() else {}

    # Load pf table
    id_pfs = json.loads(data_dir.joinpath('id_pf_map.json').read_text())

    # Loop through all speaker
    for id in sorted(ids):
        id_dir = data_dir.joinpath(id)

        # Skip already identified labels
        if id in id_labels:
            continue

        # Get ina choice
        pf = id_pfs.get(id, -1)
        ina_choice = 'f' if pf > 0.5 else 'm'

        # Loop through all tracks until identified
        tracks = [f for f in os.listdir(id_dir) if f.endswith('.flac')]
        for track_i, audio in enumerate(tracks):
            # Play track
            # sound = pygame.mixer.Sound(id_dir.joinpath(audio))
            # sound.play()
            i = input(color(
                f'\n&7Playing speaker {id[-3:]}/{len(ids)} - track {track_i}/{len(tracks)} - {audio}&r'
                f'\n- Press f / m, or anything else to play next track: ')) \
                .lower().strip()
            # sound.stop()

            # Skip
            if i == 's':
                break

            # Labeled
            if i == 'f' or i == 'm':
                id_labels[id] = i
                labels_json.write_text(json.dumps(id_labels))

                # Print choice match
                if pf != -1:
                    agree = '&aINA agrees' if ina_choice == i else '&cINA disagree'
                    printc(f'{agree} with confidence {abs(pf - 0.5) * 200:.0f}%')
                else:
                    printc(f"&7INA didn't identify any voice")
                break


if __name__ == '__main__':
    cn_celeb_root = Path(r'C:\Datasets\VoxCeleb1\wav')

    data_dir = cn_celeb_root
    ids = [id for id in os.listdir(data_dir) if id.startswith('id1')]

    # Get all audio files for each id
    audio_files = {}
    for id in ids[559:]:
        audio_files[id] = []
        for dirpath, dirnames, filenames in os.walk(data_dir / id):
            audio_files[id] += [os.path.join(dirpath, file) for file in filenames if file.endswith('.wav')]
    # print(audio_files.keys())

    segment_all()
    # graph_histogram()
    # manually_label_data()
