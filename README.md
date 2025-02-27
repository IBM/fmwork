# fmwork

FM Benchmarking Framework

## Quick start

Install conda:

```
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
```

Create environment and install deps:

```
conda create -n vllm python=3.10 -y
conda activate  vllm
pip install vllm
```

Get a model (e.g., https://huggingface.co/ibm-granite/granite-8b-code-base-128k):

```
pip install huggingface-hub
huggingface-cli download --cache-dir ./ --local-dir-use-symlinks False --revision main --local-dir models/granite-8b ibm-granite/granite-8b-code-base-128k
```

Clone repo and run experiment:

```
git clone git@github.com:IBM/fmwork.git
./fmwork/infer/vllm/driver --model_path models/granite-8b --input_size 1024 --output_size 1024 --batch_size 1,2,4 --tensor_parallel 1
```

This should produce blocks of outputs like:

```
--------------------------------------------------------------------------------
RUN 1024 / 1024 / 1 / 1
--------------------------------------------------------------------------------

FMWORK REP   1 /   3 : 1727375968.424120936 1727375976.598311213 8.174 8.0 125.3
FMWORK REP   2 /   3 : 1727375976.598364287 1727375984.859228127 8.261 8.1 124.0
FMWORK REP   3 /   3 : 1727375984.859270605 1727375993.005784506 8.147 8.0 125.7

FMWORK RES 20240926-183953.009140 1024 1024 1 1 8.204 8.0 124.8

Input size                = 1024
Output size               = 1024
Batch size                = 1
Tensor parallelism        = 1
Median iteration time (s) = 8.204
Inter-token latency (ms)  = 8.0
Throughput (tok/s)        = 124.8
```

- `FMWORK REP` lines contain stats per experiment repetition (3 repetitions by default):
    - Number of repetition
    - Total repetitions to run
    - Timestamp of rep start
    - Timestamp of rep end
    - Duration of rep (seconds)
    - Inter-token latency for rep (milliseconds per token)
    - Throughput for rep (tokens per second)

- `FMWORK RES` line contains a summary of the experiment:
    - Experiment timestamp
    - Input size
    - Output size
    - Batch size
    - Tensor parallelism size
    - Median iteration duration (seconds)
    - Inter-token latency (milliseconds per token)
    - Throughput (tokens per second)

If saved to a file, all `RES` lines can be easily grep-ed for further analysis.

```
grep -R "FMWORK RES" outputs/ | tr / ' ' | column -t
```

