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

# --------------
# sample dataset
# --------------

dataset = []

def input_dataset(model, dataset_mode, input_size, batch_size):

    if dataset_mode == 'expand': return input_dataset_expand(model, input_size, batch_size)
    if dataset_mode == 'real':   return input_dataset_real(input_size, batch_size)

def input_dataset_expand(model, input_size, batch_size):

    import random
    import transformers

    tokenizer = transformers.AutoTokenizer.from_pretrained(model)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    result = [ [] for _ in range(batch_size) ]

    for b in range(batch_size):
        l, prompt = random.choice(dataset)
        tokens = tokenizer(prompt)['input_ids']
        while len(result[b]) < input_size:
            result[b] += tokens
        result[b] = result[b][:input_size]

    return result

def input_dataset_real(input_size, batch_size):

    raise NotImplementedError()

def process_dataset(dataset_path, dataset_format, dataset_mode, model):

    import json
    import transformers

    from tqdm import tqdm

    tokenizer = transformers.AutoTokenizer.from_pretrained(model)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    file = open(dataset_path)
    data = json.load(file)

    for item in tqdm(data, total = len(data), desc = 'Processing dataset'):
        prompt = ''
        for message in item['conversations']:
            prompt += '[' + message['from'] + ']' + '\n\n'
            prompt += message['value'] + '\n\n'
        prompt = prompt.strip()

        if dataset_mode == 'expand':
            dataset.append((len(prompt), prompt))
        if dataset_mode == 'real':
            tokens = tokenizer(prompt)['input_ids']
            dataset.append((len(tokens), tokens))

    print()

    a = 0; b = 1024
    while a < 1024 * 1024:
        print(
            '%7d' % (b),
            sum(1 for l, v in dataset if a < l <= b)),
        a = b
        b = b * 2

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
    tensor_parallel,
    ignore_eos=True, outputs=None,
    debug_outputs=False, trace_outputs=False):

    var.t1s.append(time_get())
    dt = time_diff(var.t1s[-1], var.t0s[-1])
    var.dts.append(dt)

    print(
        'FMWORK REP',
        '%3d / %3d :' % (rep + 1, reps),
        '%s %s' % (time_fmt(var.t0s[-1]), time_fmt(var.t1s[-1])),
        '%.3f' % (dt),                            # rep time (s)
        '%.1f' % (1000.0 * dt / output_size),     # inter-token latency (ms)
        '%.1f' % (batch_size * output_size / dt), # throughput (tok/s)
    )

    if not ignore_eos or debug_outputs:
        print()
        for output in outputs:
            print(
                'FMWORK OUT',
                input_size, output_size, batch_size, tensor_parallel,
                rep + 1,
                '%.3f' % (dt),
                len(output.prompt_token_ids),
                len(output.outputs[0].token_ids),
                output.metrics.arrival_time,
                output.metrics.last_token_time,
                output.metrics.first_scheduled_time,
                output.metrics.first_token_time,
                output.metrics.time_in_queue,
                output.metrics.finished_time,
                output.metrics.scheduler_time)
            if trace_outputs:
                print('FMWORK TXT', repr(output.outputs[0].text))
                print('FMWORK TOK', output.outputs[0].token_ids)
        print()

    if rep + 1 == reps:
        return show(input_size, output_size, batch_size, tensor_parallel)

def show(
    input_size, output_size, batch_size,
    tensor_parallel):

    _ign = 0.2
    _ign = int(max(_ign * len(var.dts), 1)) if len(var.dts) > 1 else 0
    _rem = var.dts[_ign:]
    _med = med(_rem)
    _itl = 1000.0 * _med / output_size
    _thp = batch_size * output_size / _med

    print()

    etim = datetime.datetime.now().strftime('%Y%m%d-%H%M%S.%f')

    print(
        'FMWORK RES',
        etim,
        input_size,
        output_size,
        batch_size,
        tensor_parallel,
        '%.6f' % (_med),
        '%.1f' % (_itl),
        '%.1f' % (_thp),
    )

    print()

    print('Input size                = %d'   % (input_size))
    print('Output size               = %d'   % (output_size))
    print('Batch size                = %d'   % (batch_size))
    print('Median iteration time (s) = %.6f' % (_med))
    print('Inter-token latency (ms)  = %.1f' % (_itl))
    print('Throughput (tok/s)        = %.1f' % (_thp))

    print()

    return etim, _med

