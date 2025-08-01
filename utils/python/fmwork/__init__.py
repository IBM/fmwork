class Exception(Exception): pass

def banner(*s):

    print()
    print(80*'-')
    print(' '.join(list(map(str,s))))
    print(80*'-')
    print()

def sort_dict_recursive(d):

    if isinstance(d, dict):
        return {k: sort_dict_recursive(v) for k, v in sorted(d.items())}
    else:
        return d

import time

def time_get(): return time.perf_counter_ns()
def time_diff(t1, t0): return float(t1 - t0) / 1E9
def time_format(t): t = str(t).zfill(9); return '%s.%s' % (t[:-9], t[-9:])

import numpy as np

def avg(x): return np.mean(x)
def std(x): return np.std(x)
def med(x): return np.median(x)
def mad(x): return med(np.absolute(x - med(x)))

from . import args
from . import gen

