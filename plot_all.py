#!/usr/bin/env python

from subprocess import call
from os import sys

i = 0
while i<int(sys.argv[1]):
    i += 10
    call('./maps.py '+str(i), shell=True)
