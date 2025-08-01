import time

def get(): return time.perf_counter_ns()
def diff(t1, t0): return float(t1 - t0) / 1E9
def format(t): t = str(t).zfill(9); return '%s.%s' % (t[:-9], t[-9:])


