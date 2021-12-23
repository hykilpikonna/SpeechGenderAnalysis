import json
import os
import warnings
from pathlib import Path

import numpy as np
from inaSpeechSegmenter import Segmenter

from ina_main import process, get_result_percentages


def segment_all():
    # Create segmenter
    seg = Segmenter()
    np.seterr(invalid='ignore')

    # Loop through all celebrities
    ids = [id for id in os.listdir(data_dir) if id.startswith('id')]
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


if __name__ == '__main__':
    cn_celeb_root = Path('C:/Users/me/Workspace/Data/CN-Celeb_flac')
    data_dir = cn_celeb_root.joinpath('data')

    segment_all()
