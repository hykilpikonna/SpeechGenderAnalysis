
from pathlib import Path
import os
import json

import numpy as np
import pandas as pd

if __name__ == '__main__':
    cn_celeb_root = Path(r'C:\Datasets\vox1_test_wav\wav')
    data_dir = cn_celeb_root
    ids = [id for id in os.listdir(data_dir) if id.startswith('id1')]
    # ids=[data_dir / id for id in os.listdir(data_dir) if id.startswith('id1')]

    appendix= Path(r'C:\Datasets\VoxCeleb1\wav')
    ids += [id for id in os.listdir(appendix) if id.startswith('id1')]
    # ids += [appendix / id for id in os.listdir(appendix) if id.startswith('id1')]
    with open(r"C:\Datasets\vox1_label.csv") as f:
        txt = f.read().strip()
        map = {}
        for row in txt.split('\n'):
            id, gender = row.split(',')
            map[id] = gender
    # female-> positive, male -> negative
    f_correct = 0 #tp
    f_incorrect = 0 #fp
    m_correct = 0 #tn
    m_incorrect = 0 #fn
    for id in ids[:40]:
        obj = json.loads((data_dir / id / 'total.json').read_text())

        label = map[id] #ground truth
        if label == 'f':
            if obj['ratio'] >= 0.5:
                f_correct += 1
            else:
                # f_incorrect += 1 #fn
                m_incorrect += 1
        if label == 'm':
            if obj['ratio'] < 0.5:
                m_correct += 1
            else:
                # m_incorrect += 1 #fp
                f_incorrect += 1
    for id in ids[40:]:
        obj = json.loads((appendix / id / 'total.json').read_text())

        label = map[id] #ground truth
        if label == 'f':
            if obj['ratio'] >= 0.5:
                f_correct += 1
            else:
                # f_incorrect += 1 #fn
                m_incorrect += 1
        if label == 'm':
            if obj['ratio'] < 0.5:
                m_correct += 1
            else:
                # m_incorrect += 1 #fp
                f_incorrect += 1


    # print(f_incorrect)
    # print(m_incorrect)

    f_precision = f_correct / (f_correct + f_incorrect)
    f_recall = f_correct / (f_correct + m_incorrect)

    m_precision = m_correct / (m_correct + m_incorrect)
    m_recall = m_correct / (m_correct + f_incorrect)

    print('Precision_f', f_precision)
    print('Recall_f', f_recall)

    print('Precision_m', m_precision)
    print('Recall_m', m_recall)

    print('total number: ', f_incorrect+f_correct+m_incorrect+m_correct)
