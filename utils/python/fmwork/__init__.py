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

from . import args
from . import time
from . import gen

