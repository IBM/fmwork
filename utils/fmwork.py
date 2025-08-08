import fmwork

#   ============================================================================
#
#   CORE
#   ----
#
#   Core utilities. Error handling, data / variable processing,
#   timing functions, statistics / metrics calculation, etc.
#
#   ============================================================================

class Exception(Exception): pass

def banner(*s):

    print()
    print(80*'-')
    print(' '.join(list(map(str,s))))
    print(80*'-')
    print()

#   Recursively sort a dictionary by its keys.
#   ------------------------------------------

def sort_dict_recursive(d):

    if isinstance(d, dict):
        return {k: sort_dict_recursive(v) for k, v in sorted(d.items())}

    else:
        return d

#   Recursively convert objects to strings that can be eval'ed.
#   -----------------------------------------------------------

def to_evaluable_strings(obj):

    if isinstance(obj, dict):
        return {k: to_evaluable_strings(v) for k, v in obj.items()}

    elif isinstance(obj, str):
        return obj

    else:
        return repr(obj)

#   Functions for timing.
#   ---------------------

import time

def time_get(): return time.time_ns()
def time_diff(t1, t0): return float(t1 - t0) / 1.0E9
def time_format(t): t = str(t).zfill(9); return '%s.%s' % (t[:-9], t[-9:])

#   Statistics / metrics.
#   ---------------------

import numpy as np

def avg(x): return np.mean(x)
def std(x): return np.std(x)
def med(x): return np.median(x)
def mad(x): return med(np.absolute(x - med(x)))

#   ============================================================================
#
#   ARGUMENT PROCESSING
#   -------------------
#
#   Functions to process command line arguments.
#
#   ============================================================================

import json
import shlex
import yaml

#   Process sub-options.
#   --------------------
#
#   Given the main arguments `args`,
#   sub-options `opts` (anything not captured by main arguments),
#   list `sublist` of supported / valid sub-options,
#   and additional `namespace` to consider when evaluating names,
#   process sub-options and add them to main `args` object.
#
#   Sub-options look like:
#
#   ```
#   --engine:enable_prefix_caching@ False
#   --engine:compilation_config:cudagraph_capture_sizes@ args.batch_sizes
#   --engine:max_seq_len_to_capture@ 'max(args.input_sizes)+max(args.output_sizes)'
#   --engine:max_num_seqs@ 512
#   ```
#
#   Always in the format `--SUB:KEY VAL`. Nested dictionaries are allowed,
#   as shown in the second example above -- `engine` is the sub-option name,
#   `compilation_config` is a dict name supported by vllm.LLM, and
#   `cudagraph_capture_sizes` is an option (key) name valid within this dict.
#
#   Values can either be just strings or strings to be eval'ed.
#   We suffix a sub/key name with `@` to indicate that its value must be
#   eval'ed. E.g., in the first example, the value of
#   `args.subs['engine']['enable_prefix_caching']` with be `False` -- the
#   boolean, not a string. In contrast, the value in the second example
#   will be stored as a string.
#
#   Note also that the fact that the value is eval'ed allows some further
#   / more complex logic. The third example uses values which are present
#   in the variable namespace of the module -- in this case, reusing the 
#   values of `args` itself to set `max_seq_len_to_capture` as the sum of 
#   the maximum value of provided input sizes and of output sizes.
#
#   This function should fail
#   if a particular sub-option name (key) is not supported,
#   if a key is malformed, or if it doesn't have a corresponding value.

def args_process_opts(args, opts, sublist, namespace):

    subs = {}

    i = 0
    while i < len(opts):
        args_process_opt(namespace, subs, opts, i)
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

#   Process one particular sub-option.
#   ----------------------------------

def args_process_opt(namespace, subs, opts, i):

    key = opts[i] # get key

    # check if we have a proper key

    if not key.startswith('--'):
        raise fmwork.Exception(
            f'Key expected to start with "--"; got: "{key}".')

    # check if key has a value

    if i + 1 >= len(opts):
        raise fmwork.Exception(
            f'Key "{key}" is missing a value.')

    # determine if value must be eval'ed or not

    is_eval = False
    if key[-1] == '@':
        is_eval = True
        key = key[:-1]
    
    # process key

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

#   Print values of arguments.
#   --------------------------

def args_show(args):

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

#   ============================================================================
#
#   INPUT GENERATORS
#   ----------------
#
#   Classes and functions that implement different options of input generators.
#   Inputs can be text (prompts); batched or not; as well as multimodal inputs
#   for particular models / experiments.
#
#   ============================================================================

import random
import torch

from transformers import AutoTokenizer
from transformers.tokenization_utils_base import BatchEncoding

class RandomGenerator:

    def __init__(self, model_path):

        tokenizer = AutoTokenizer.from_pretrained(model_path)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"

        self.vocab = list(range(0, tokenizer.vocab_size))
        for i in tokenizer.all_special_ids:
            if i in self.vocab:
                self.vocab.remove(i)

    def prompt(self, input_size, batch_size, return_tensors):

        tokens = [ [] for _ in range(batch_size) ]
        for b in range(batch_size):
            for i in range(input_size):
                tokens[b].append(random.choice(self.vocab))

        if return_tensors == 'np': return tokens

        input_batch = BatchEncoding({
            'input_ids' : torch.tensor(tokens),
            'attention_mask' : torch.ones(batch_size, input_size),
        })

        return input_batch

