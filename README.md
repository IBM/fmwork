# fmwork

⚠️ 
Code is currently being restructured. 
Release [v0.1.0](https://github.com/IBM/fmwork/releases/tag/v0.1.0) has a basic version of it to support our ongoing (internal) sweeps.
Version 1.0.0 (soon) should encompass this new structure, that should better support different hardware and software options.

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
- `FMWORK GEN` line also contains a summary of the experiment with additional metrics:
    - Experiment timestamp
    - Input size
    - Output size
    - Batch size
    - Tensor parallelism size
    - Median iteration duration (seconds)
    - Time to fist token (seconds)
    - Generation time (s)
    - Inter-token latency (milliseconds per token)
    - Throughput (tokens per second)

To save experiment results to a directory, run the fmwork `runner` script instead of the `driver` script and provide the directory path in the `rdir` argument:
```bash
./fmwork/infer/vllm/runner --rdir ~/md0/fmwork-runs --mr ~/models/fmwork --mmtps granite-3.3-8b/bf16:1 --iis 1024 batch --oos 1,1024 batch --bbs 1,2,4 batch --devs 1:3:4:5/1,3:4,5/1,3,4,5/
```
To parse the results and save to a CSV file, run `process.py` with the the above `rdir` output directory path as the first argument and the desired CSV file path as the second argument:
```bash
cd fmwork/infer/vllm
python process.py ~/md0/fmwork-runs ~/md0/fmwork-results/fmwork_data.csv
```
`process.py` will parse all of the fmwork results in the `rdir` directory and write the data to the CSV file with the following header: `work,user,host,btim,etim,hw,hwc,back,mm,prec,dp,ii,oo,bb,tp,med,ttft,gen,itl,thp,extraparams,vllmvars`

| **Field**      | **Description**                                 |
|----------------|-------------------------------------------------|
| **work**       | quarter, year                                   |
| **user**       | username                                        |
| **host**       | hostname                                        |
| **btim**       | batch (sweep) timestamp                         |
| **etim**       | experiment timestamp                            |
| **hw**         | hardware used (GPU model)                       |
| **hwc**        | hardware count (number of GPUs used)            |
| **back**       | backend inference engine (VLLM version)         |
| **mm**         | model                                           |
| **prec**       | floating point precision                        |
| **dp**         | data parallelism (usually 1)                    |
| **ii**         | input size                                      |
| **oo**         | output size                                     |
| **bb**         | batch size                                      |
| **tp**         | tensor parallel size                            |
| **med**        | inference time in seconds                       |
| **ttft**       | time to first token                             |
| **gen**        | generation time                                 |
| **itl**        | inter-token latency                             |
| **thp**        | throughput                                      |
| **extraparams**| extra VLLM parameters                           |
| **vllmvars**   | VLLM environment variables                      |





