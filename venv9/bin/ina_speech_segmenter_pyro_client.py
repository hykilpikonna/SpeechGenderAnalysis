#!/Volumes/macWorkspace/CS/SpeechGenderAnalysis/venv/bin/python3.9
# encoding: utf-8

# The MIT License

# Copyright (c) 2018 Ina (David Doukhan - http://www.ina.fr/)

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
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.


import Pyro4
import sys
import os
import socket

from inaSpeechSegmenter import Segmenter


if __name__ == '__main__':
    dname = os.path.dirname(os.path.realpath(__file__))

    hostname = socket.gethostname()

    uri = sys.argv[1]
    jobserver = Pyro4.Proxy(uri)

    ret = -1
    outname = 'init'
    
    # batch size set at 1024. Use lower values with small gpus
    g = Segmenter(batch_size=1024)
    
    while True:
        lsrc, ldst = jobserver.get_njobs('%s %s' % (hostname, ret))
            
        print(lsrc, ldst)
        if len(lsrc) == 0:
            print('job list finished')
            break
        
        ret =  g.batch_process(lsrc, ldst, skipifexist=True, nbtry=3)
