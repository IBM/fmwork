# embed/tf

Embedding engine implementation using `torch` + `transformers`.

## Usage

On GPUs (e.g., CUDA):

```bash
PYTHONUNBUFFERED=1 \
/path/to/fmwork/embed/tf/driver \
    --platform cuda \
    --model_root /path/to/models \
    --model_name ibm-granite/granite-embedding-125m-english \
    --model_class RobertaModel \
    --input_sizes 512 \
    --batch_sizes 1 \
    --reps 100
```

On Spyre:

```bash
PYTHONUNBUFFERED=1 \
DTLOG_LEVEL=error \
DT_DEEPRT_VERBOSE=-1 \
DTCOMPILER_KEEP_EXPORT=-1 \
TORCH_SENDNN_LOG=CRITICAL \
/path/to/fmwork/embed/tf/driver \
    --platform spyre \
    --model_root /path/to/models \
    --model_name ibm-granite/granite-embedding-125m-english \
    --model_class RobertaModel \
    --input_sizes 512 \
    --batch_sizes 1 \
    --reps 100 \
    --torch.call:set_grad_enabled@ False \
    --compile \
    --compile:backend sendnn
```

`--torch.call:set_grad_enabled@ False`, `--compile` and `--compile:backend sendnn` are required.
Additional environment variables are optional but might help avoiding verbose outputs.
Also on Spyre, only one combination of input / batch size can be executed at a time.
In other words, `--input_sizes` and `--batch_sizes` must not be lists.

## Example of output

The `driver` can take one or more combinations of input and batch sizes.
For each combination, the following output block will be produced:

```
--------------------------------------------------------------------------------
RUN 128 / 1
--------------------------------------------------------------------------------

FMWORK REP 1 100 1108911.088153850 1108911.845211219 0.757057369
FMWORK REP 2 100 1108911.845509595 1108911.849978236 0.004468641
FMWORK REP 3 100 1108911.850187359 1108911.854049551 0.003862192
...
FMWORK REP 98 100 1108912.224051913 1108912.227749102 0.003697189
FMWORK REP 99 100 1108912.227934311 1108912.231636694 0.003702383
FMWORK REP 100 100 1108912.231828882 1108912.235531503 0.003702621

FMWORK RES 1108911.088153850 1108912.235531503 ibm-granite/granite-embedding-125m-english RobertaModel 128 1 3.691 270.9
```

The `FMWORK REP` lines provide information about each repetition (controlled by `--reps`):
* Current rep (e.g., `1`)
* Total reps to run (e.g., `100`)
* Start timestamp of rep (e.g., `1108911.088153850`)
* End timestamp of rep (e.g., `1108911.845211219`)
* Duration of rep in seconds (e.g., `0.757057369`)

The `FMWORK RES` line has a summary of the results:
* First timestamp of first rep (e.g., `1108911.088153850`)
* Last timestamp of last rep (e.g., `1108912.235531503`)
* Model name (e.g., `ibm-granite/granite-embedding-125m-english`)
* Model class (e.g., `RobertaModel`)
* Input (prompt) size (e.g., `128`)
* Batch size (e.g., `1`)
* Latency in milliseconds (e.g., `3.691`)
* Throughput (speed) in sequences per second (e.g., `270.9`)

## More on parameters

```python
    parser = argparse.ArgumentParser()
    parser.add = parser.add_argument
    parser.add('--platform',    type=str, required=True)
    parser.add('--model_class', type=str, required=True)
    parser.add('--model_root',  type=str)
    parser.add('--model_name',  type=str, required=True)
    parser.add('--compile',     action='store_true')
    parser.add('--eval',        action='store_true')
    parser.add('--input_sizes', type=str, required=True)
    parser.add('--batch_sizes', type=str, required=True)
    parser.add('--reps',        type=int, required=True)
    args, opts = parser.parse_known_args()
    fmwork.args.process_opts(args, opts, [
        'compile', 'model', 'torch.call', 'torch.set',
    ], globals())
```

The `driver` script takes a number of "fixed" parameters:

* `--platform` :
        Hardware/software platform identifier.
        Currently supported/tested `cuda` and `spyre`.
* `--model_class` :
        Model class from `transformers` library to instantiate model.
        Depends on the selected model.
        Usual values include `BertModel` and `RobertaModel`.
  `AutoModel` can also be used.
* `--model_root` :
        Path to root folder where models are located.
        This is not the path to the model itself.
        This is just a helper when pretty printing the model name.
* `--model_name` :
        Model name.
        If `--model_root` is not specified,
        this should be the full path to the model.
* `--compile` :
        Controls whether `torch.compile` is called or not.
        Further options might be required in the form of dynamic sub-options.
* `--eval` :
        Controls whether `model.eval()` is called or not.
* `--input_sizes` :
        Comma-separated list of input sizes (sequence / prompt length).
        On `spyre` this must be a single value.
* `--batch_sizes` :
        Comma-separated list of batch sizes (concurrent requests / users).
        On `spyre` this must be a single value.
* `--reps` :
        Number of repetitions to run.

The `driver` script can also take a number of dynamic sub-options.
For instance, if `--compile` is passed,
    one might specify how to compile the model using one or more subs:

```bash
PYTHONUNBUFFERED=1 \
/path/to/fmwork/embed/tf/driver \
    --platform cuda \
    --model_root /path/to/models \
    --model_name ibm-granite/granite-embedding-125m-english \
    --model_class RobertaModel \
    --input_sizes 512 \
    --batch_sizes 1 \
    --reps 100 \
    --compile:backend inductor \
    --compile:dynamic@ True \
    --compile:mode reduce-overhead
```

For each `--key val` parameter,
    the `:` in the key indicates the name of the set of sub-options
    -- in this case, `compile`.
The actual option / parameter name comes next
    -- e.g., `backend` or `dynamic`.
If the option contains a `@`,
    this indicates the value will be `eval()`;
    else, the value is `str()`.
In this case, `inductor` is just a string (name),
    while `True` is evaluated to Python's `True`.

Current `driver` defines four sets of sub-options:

* `compile` :
        Options passed to the `torch.compile` call.
        Please refer to the documentation associated to the `torch` version
            you are currently using --
            e.g., https://docs.pytorch.org/docs/stable/generated/torch.compile.html.
        Nested options are supported --
            e.g., the `options` dict that can be passed to `torch.compile`.
* `model` :
        Options passed to the `<model_class>.from_pretrained()` call.
* `torch.call` :
        `torch` functions called during engine initialization.
        For instance, `--torch.call:set_grad_enabled@ False`
            calls `torch.set_grad_enabled(False)`, which is useful to disable
            gradient computation during the execution of the benchmark
            (required on Spyre).
* `torch.set` :
        `torch` variables to be assigned.
        This can be used, for instance, to set up something like
        `torch.backends.cudnn.benchmark = True` via
        `--torch.set:backends.cudnn.benchmark@ True`.

## Processing results

A file containing results of an experiment (outputs from the `driver`)
    can be processed using the `process` script.
Usage:

```bash
/path/to/fmwork/embed/tf/process \
    --path <path> \
    --metadata_id <id>
```

* `--path` is the path to the file.
* `--metadata_id` can be used to associate the generated JSON
        to external information that, for instance, describes the environment
        where the experiment was executed.

The script will print a JSON containing a list of results present in the file.
