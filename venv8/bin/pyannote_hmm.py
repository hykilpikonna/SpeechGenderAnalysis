#!/Volumes/macWorkspace/CS/SpeechGenderAnalysis/venv8/bin/python3.8
# encoding: utf-8

# The MIT License (MIT)

# Copyright (c) 2014 CNRS

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# AUTHORS
# Hervé BREDIN - http://herve.niderb.fr

"""
Hidden Markov Model with (constrained) Viterbi decoding

Usage:
  hmm train [-g <gaussian>] [-c <covariance>] [-d <duration>] <uris.lst> <references.mdtm> <features.pkl> <model.pkl>
  hmm apply [-d <duration>] [-f <constraint.mdtm>] <model.pkl> <features.pkl> <hypothesis.mdtm>
  hmm -h | --help
  hmm --version

Options:
  -g <gaussian>            Number of gaussian components [default: 16].
  -c <covariance>          Covariance type (diag or full) [default: diag].
  -d <duration>            Minimum duration in seconds [default: 0.250].
  -f <constraint.mdtm>     Constrain Viterbi decoding to follow this path.
  -h --help                Show this screen.
  --version                Show version.
"""

from pyannote.algorithms.segmentation.hmm import ViterbiHMM
from pyannote.parser.util import CoParser
from pyannote.parser import MDTMParser
from docopt import docopt
import pickle


def do_train(
    uris_lst, references_mdtm, features_pkl, model_pkl,
    n_components=16, covariance_type='diag', min_duration=0.250,
):

    hmm = ViterbiHMM(
        n_components=n_components, covariance_type=covariance_type,
        random_state=None, thresh=1e-2, min_covar=1e-3, n_iter=10,
        disturb=0.05, sampling=1000, min_duration=min_duration)

    # iterate over all uris in a synchronous manner
    coParser = CoParser(uris=uris_lst,
                        reference=references_mdtm,
                        features=features_pkl)
    references, features = coParser.generators('reference', 'features')

    hmm.fit(references, features)

    with open(model_pkl, 'wb') as f:
        pickle.dump(hmm, f)


def do_apply(model_pkl, features_pkl, hypothesis_mdtm,
             min_duration=0.250, constraint_mdtm=None):

    with open(model_pkl, 'rb') as f:
        hmm = pickle.load(f)

    hmm.min_duration = min_duration

    with open(features_pkl, 'rb') as f:
        features = pickle.load(f)

    constraint = None
    if constraint_mdtm:
        constraint = MDTMParser().read(constraint_mdtm)()

    hypothesis = hmm.apply(features, constraint=constraint)

    with open(hypothesis_mdtm, 'w') as f:
        MDTMParser().write(hypothesis, f=f)


if __name__ == '__main__':

    arguments = docopt(__doc__, version='Hidden Markov Models 1.0')

    if arguments['train']:

        uris_lst = arguments['<uris.lst>']
        references_mdtm = arguments['<references.mdtm>']
        features_pkl = arguments['<features.pkl>']
        model_pkl = arguments['<model.pkl>']

        n_components = int(arguments['-g'])
        covariance_type = arguments['-c']
        min_duration = float(arguments['-d'])

        do_train(
            uris_lst, references_mdtm, features_pkl, model_pkl,
            n_components=n_components, covariance_type=covariance_type,
            min_duration=min_duration)

    elif arguments['apply']:

        model_pkl = arguments['<model.pkl>']
        features_pkl = arguments['<features.pkl>']
        hypothesis_mdtm = arguments['<hypothesis.mdtm>']
        min_duration = float(arguments['-d'])
        constraint_mdtm = arguments['-f']

        do_apply(model_pkl, features_pkl, hypothesis_mdtm,
                 min_duration=min_duration, constraint_mdtm=constraint_mdtm)
