import json
from pathlib import Path
from typing import Literal

from parselmouth import Sound
from scipy.stats import gaussian_kde

from calculations import calculate_freq_statistics, calculate_freq_info, calculate_tilt

Feature = Literal['pitch', 'f1', 'f2', 'f3', 'tilt']
Gender = Literal['f', 'm']


_kde_functions: dict[Feature, dict[Gender, gaussian_kde]] = {}


def load_kde() -> dict[Feature, dict[Gender, gaussian_kde]]:
    """
    Load statistical results into kernel density functions

    :return: Kernel density functions for F and M for pitch, f1, f2, f3, tilt
    """
    if _kde_functions:
        return _kde_functions

    data: dict[Feature, dict[Gender, list[float]]] = {**json.loads(Path('results/frequency-data.json').read_text()),
                                                      **json.loads(Path('results/tilt-data.json').read_text())}

    # Lowercase keys
    data = {k.lower(): data[k] for k in data}

    # Fit KDE functions
    for feature in data:
        _kde_functions[feature] = {}
        for gender in data[feature]:
            kde = gaussian_kde(data[feature][gender], 'scott')
            _kde_functions[feature][gender] = kde

    return _kde_functions


def calculate_feature_means(audio: Sound) -> dict[Feature, float]:
    s = calculate_freq_statistics(calculate_freq_info(audio))
    return {'pitch': s.pitch.mean, 'f1': s.f1.mean, 'f2': s.f2.mean, 'f3': s.f3.mean, 'tilt': calculate_tilt(audio)}


def _calculate_fem_prob(feature: Feature, value: float) -> float:
    """
    Calculate probability of a feature sounding feminine

    :return: Ratio between 0 and 1
    """
    f = load_kde()[feature]['f'].evaluate([value])[0]
    m = load_kde()[feature]['m'].evaluate([value])[0]
    return f / (f + m)


def calculate_feature_classification(audio: Sound):
    """
    Run statistical classification based on kernel density estimation.

    :param audio: Audio
    :return: Statistical results {'means': {'pitch': ..., 'f1': ...}, 'fem_prob': {'pitch': ..., 'f1': ...}}
    """
    means = calculate_feature_means(audio)
    fem_prob = {feature: _calculate_fem_prob(feature, means[feature]) for feature in means}
    return {'means': means, 'fem_prob': fem_prob}
