from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

import numpy as np
import pkg_resources
from parselmouth import Sound
from scipy.stats import gaussian_kde

from .calculations import calculate_freq_statistics, calculate_freq_info, calculate_tilt

Feature = Literal['pitch', 'f1', 'f2', 'f3', 'tilt']
Gender = Literal['f', 'm']


_kde_functions: dict[Feature, dict[Gender, gaussian_kde]] = {}
_kde_boundaries: dict[Feature, tuple[float, float]] = {}


def load_kde() -> dict[Feature, dict[Gender, gaussian_kde]]:
    """
    Load statistical results into kernel density functions

    :return: Kernel density functions for F and M for pitch, f1, f2, f3, tilt
    """
    if _kde_functions:
        return _kde_functions

    data_file = pkg_resources.resource_filename(__name__, 'data/vox1_data.json')
    data: dict[Feature, dict[Gender, list[float]]] = json.loads(Path(data_file).read_text())

    # Lowercase keys
    data = {k.lower(): data[k] for k in data}

    # Fit KDE functions
    # Also find boundaries (99th percentile for fem and 1st percentile for masc)
    for feature in data:
        _kde_functions[feature] = {}
        for gender in data[feature]:
            kde = gaussian_kde(data[feature][gender], 'scott')
            _kde_functions[feature][gender] = kde

        # Boundaries
        _kde_boundaries[feature] = (np.percentile(data[feature]['m'], 1),
                                    np.percentile(data[feature]['f'], 99))

    return _kde_functions


def calculate_feature_means(audio: Sound) -> tuple[dict[Feature, float], np.ndarray]:
    """
    Calculate frequency info and feature means

    :param audio: Audio
    :return: means, frequency array
    """
    freq_info = calculate_freq_info(audio)
    s = calculate_freq_statistics(freq_info)
    return {'pitch': s.pitch.mean, 'f1': s.f1.mean, 'f2': s.f2.mean, 'f3': s.f3.mean, 'tilt': calculate_tilt(audio)}, freq_info


def _calculate_fem_prob(feature: Feature, value: float) -> float:
    """
    Calculate probability of a feature sounding feminine

    :return: Ratio between 0 and 1
    """
    f = load_kde()[feature]['f'].evaluate([value])[0]
    m = load_kde()[feature]['m'].evaluate([value])[0]

    # Boundaries
    m1, f99 = _kde_boundaries[feature]
    if value > f99:
        return 1
    if value < m1:
        return 0

    return f / (f + m)


def calculate_feature_classification(audio: Sound) -> tuple[dict[Literal['means', 'fem_prob'], dict[Feature, float]], np.ndarray]:
    """
    Run statistical classification based on kernel density estimation.

    :param audio: Audio
    :return: Statistical results {'means': {'pitch': ..., 'f1': ...}, 'fem_prob': {'pitch': ..., 'f1': ...}}, and frequency array
    """
    means, freq_array = calculate_feature_means(audio)
    fem_prob = {feature: _calculate_fem_prob(feature, means[feature]) for feature in means}
    return {'means': means, 'fem_prob': fem_prob}, freq_array
