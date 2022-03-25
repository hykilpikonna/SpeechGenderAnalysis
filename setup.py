#!/usr/bin/env python3
from pathlib import Path

from setuptools import setup

import sgs

setup(
    name="speech-gender-statistics",
    version=sgs.__version__,
    author="Azalea Gui",
    author_email="hykilpikonna@gmail.com",
    description='Using statistical methods to determine speech femininity/masculinity based on phonetic features',
    license="MIT",
    install_requires=['praat-parselmouth', 'numpy', 'scipy'],
    url="https://github.com/hykilpikonna/SpeechGenderAnalysis",
    packages=['sgs'],
    package_data={'': ['vox1_data.json']},
    long_description=Path('README.md').read_text(),
    long_description_content_type='text/markdown',
    python_requires='>=3.8',
)
