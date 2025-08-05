import fmwork
import json
import shlex
import yaml

def process_opts(args, opts, sublist, namespace):

    subs = {}

    i = 0
    while i < len(opts):
        process_opt(namespace, subs, opts, i)
        i += 2

    for key in subs:
        if key not in sublist:
            raise fmwork.Exception(
                f'Sub-option "--{key}." is not supported.'
                f'\nSupported options are: {sublist}')

    for sub in sublist:
        if sub not in subs:
            subs[sub] = {}

    args.subs    = subs
    args.sublist = sublist

def process_opt(namespace, subs, opts, i):

    key = opts[i] # get key

    # check if we have a proper key

    if not key.startswith('--'):
        raise fmwork.Exception(
            f'Key expected to start with "--"; got: "{key}".')

    # check if key has a value

    if i + 1 >= len(opts):
        raise fmwork.Exception(
            f'Key "{key}" has no associated value.')

    # process key

    is_eval = False
    if key[-1] == '@':
        is_eval = True
        key = key[:-1]

    key     = key[2:]                   # remove --
    key     = key.replace(' ', '\x01')  # replace spaces by placeholder
    key     = key.replace(':', ' ')     # replace colons by spaces (to shlex)
    split   = shlex.split(key)          # split preserving quotes

    for idx in range(len(split)):                    # iterate over split
        split[idx] = split[idx].replace(' ', ':')    # revert colons
        split[idx] = split[idx].replace('\x01', ' ') # revert spaces

    # process sub

    sub = split[0]
    if sub not in subs:
        subs[sub] = {}

    # process val

    val = opts[i+1]
    if is_eval:
        val = eval(val, namespace)

    # dig into dict

    dic = subs[sub]                     # pointer to current dict
    rem = split[1:]                     # remaining keys
    for idx, key in enumerate(rem):     # iterate over remaining keys
        if idx + 1 == len(rem):         # if last key
            dic[key] = val              # set key and value
        else:                           # else
            if not key in dic:          # if sub-dict does not exist yet
                dic[key] = {}           # create next dict level
            dic = dic[key]              # set current level to it

def show(args):

    import sys

    cli = sys.argv[1:]
    tmp = vars(args)
    lst = []

    i = 0
    while i < len(cli):
        key = cli[i][2:]; i += 1
        if key in tmp and isinstance(getattr(args, key), bool):
            lst.append(f'--{key}')
        else:
            val = cli[i]; i += 1
            lst.append(f'--{key} {val}')

    for s in sorted(lst):
        print('FMWORK ARG', s)

    print()

    tmp = vars(args)
    tmp = {k: v for k, v in tmp.items() if k != 'sublist'}

    # sort subs recursively if it exists

    if 'subs' in tmp:
        tmp['subs'] = fmwork.sort_dict_recursive(tmp['subs'])

    # get all keys except subs and sort them
    regular_keys = sorted([k for k in tmp.keys() if k != 'subs'])

    # build the ordered dictionary
    ordered_tmp = {}
    for key in regular_keys:
        ordered_tmp[key] = tmp[key]

    # add subs at the end if it exists
    if 'subs' in tmp:
        ordered_tmp['subs'] = tmp['subs']

    # convert everything to evaluable strings

    evaluable_tmp = to_evaluable_strings(ordered_tmp)

    print(json.dumps(evaluable_tmp, indent=4))

def to_evaluable_strings(obj):

    if isinstance(obj, dict):
        return {k: to_evaluable_strings(v) for k, v in obj.items()}
    elif isinstance(obj, str):
        return obj
    else:
        return repr(obj)
