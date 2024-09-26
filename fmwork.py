# --------------
# base utilities
# --------------

def banner(*s):
    print(); print(80*'-');
    print(' '.join(list(map(str,s))));
    print(80*'-'); print()

# ----------------
# timing utilities
# ----------------

import time

def time_get(): return time.time_ns()
def time_diff(t1, t0): return float(t1 - t0) / 1E9
def time_fmt(t): t = str(t).zfill(9); return '%s.%s' % (t[:-9], t[-9:])

# ---------------
# stats utilities
# ---------------

import numpy as np

def avg(x): return np.mean(x)
def std(x): return np.std(x)
def med(x): return np.median(x)
def mad(x): return med(np.absolute(x - med(x)))

# -------------------------
# generate synthetic inputs
# -------------------------

def input_generator(model, input_size, batch_size, return_tensors):

    import random
    import transformers

    tokenizer = transformers.AutoTokenizer.from_pretrained(model)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    vocab = list(range(0, tokenizer.vocab_size))
    for i in tokenizer.all_special_ids:
        if i in vocab:
            vocab.remove(i)

    tokens = [ [] for _ in range(batch_size) ]
    for b in range(batch_size):
        for i in range(input_size):
            tokens[b].append(random.choice(vocab))

    if return_tensors == 'np': return tokens

    import torch
    from transformers.tokenization_utils_base import BatchEncoding

    input_batch = BatchEncoding({
        'input_ids' : torch.tensor(tokens),
        'attention_mask' : torch.ones(batch_size, input_size),
    })

    return input_batch

# ------------
# benchmarking
# ------------

import datetime

class var: pass
var.t0s = None
var.t1s = None
var.dts = None

def reset():
  var.t0s = []
  var.t1s = []
  var.dts = []

def t0(): var.t0s.append(time_get())

def t1(
    rep, reps,
    input_size, output_size, batch_size,
    tensor_parallel):

    var.t1s.append(time_get())
    dt = time_diff(var.t1s[-1], var.t0s[-1])
    var.dts.append(dt)

    print(
        'REP',
        '%3d / %3d :' % (rep + 1, reps),
        '%s %s' % (time_fmt(var.t0s[-1]), time_fmt(var.t1s[-1])),
        '%.3f' % (dt),                            # rep time (s)
        '%.1f' % (1000.0 * dt / output_size),     # inter-token latency (ms)
        '%.1f' % (batch_size * output_size / dt), # throughput (tok/s)
    )

    if rep + 1 == reps:
        show(input_size, output_size, batch_size, tensor_parallel)

def show(
    input_size, output_size, batch_size,
    tensor_parallel):

    _ign = 0.2
    _ign = int(max(_ign * len(var.dts), 1))
    _rem = var.dts[_ign:]
    _med = med(_rem)
    _itl = 1000.0 * _med / output_size
    _thp = batch_size * output_size / _med

    print()

    print(
        'RES',
        datetime.datetime.now().strftime('%Y%m%d-%H%M%S.%f'),
        input_size,
        output_size,
        batch_size,
        tensor_parallel,
        '%.3f' % (_med),
        '%.1f' % (_itl),
        '%.1f' % (_thp),
    )

    print()

    print('Input size                = %d'   % (input_size))
    print('Output size               = %d'   % (output_size))
    print('Batch size                = %d'   % (batch_size))
    print('Tensor parallelism        = %d'   % (tensor_parallel))
    print('Median iteration time (s) = %.3f' % (_med))
    print('Inter-token latency (ms)  = %.1f' % (_itl))
    print('Throughput (tok/s)        = %.1f' % (_thp))

