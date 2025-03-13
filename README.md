# fmwork

FM Benchmarking Framework

## Quick start

Clone and Install vllm in a v1.20.0 Gaudi3 release docker container 

```
git clone https://github.com/HabanaAI/vllm-fork.git
cd vllm-fork
git checkout 391f101
pip install -r requirements-hpu.txt  
python setup.py develop  
```

Get a model (e.g., https://huggingface.co/ibm-granite/granite-8b-code-base-128k):

```
pip install huggingface-hub
huggingface-cli download --cache-dir ./ --local-dir-use-symlinks False --revision main --local-dir models/granite-8b ibm-granite/granite-3.1-8b-instruct
```

Clone repo and run experiment:

```
git clone git@github.com:IBM/fmwork.git
cd fmwork
./run.sh -m ibm-granite/granite-3.1-8b-instruct -b 125 --multistep 32

# vision model (include --vision flag)
./run.sh -m meta-llama/Llama-3.2-90B-Vision-Instruct -t 4 -b 64 --multistep 32 --vision

# fp8 quantization example
QUANT_CONFIG=/pathto/llama-3.3-70b-instruct/maxabs_quant_g3.json PT_HPU_LAZY_MODE=1 FUSER_ENABLE_LOW_UTILIZATION=1  ./run.sh -m meta-llama/llama-3.3-70b-instruct -t 4 -b 220 --fp8
```

Note: FP8 quantization requires calibration to be done prior to running inferencing. `QUANT_CONFIG` file will need to be passed as a variable before the `run.sh` command.  

This should produce blocks of outputs like:

```
--------------------------------------------------------------------------------
RUN 1024 / 1024 / 125 / 1
--------------------------------------------------------------------------------
FMWORK REP   1 /   3 : 1738969988.187714421 1738970013.446952411 25.259 24.7 5067.5
FMWORK REP   2 /   3 : 1738970013.447015145 1738970038.822735493 25.376 24.8 5044.2
FMWORK REP   3 /   3 : 1738970038.822796496 1738970064.211628445 25.389 24.8 5041.6

FMWORK RES 20250207-231424.212051 1024 1024 125 1 25.382 24.8 5042.9

Input size                = 1024
Output size               = 1024
Batch size                = 125
Tensor parallelism        = 1
Median iteration time (s) = 25.382
Inter-token latency (ms)  = 24.8
Throughput (tok/s)        = 5042.9

--------------------------------------------------------------------------------
DONE
--------------------------------------------------------------------------------
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

System config 
Kernel : 6.8.0-52-generic
OS : Ubuntu 24.04.5 LTS
PT version : 2.6.0 

Gaudi3 Models run command examples 

## Tested Models and Configurations

The following table contains models and configurations we have validated on Gaudi3.

# Validated Models on Gaudi3

The following table contains models and configurations we have validated on Gaudi3.

| Model | D-Type | Devices | Command |
|--------------| --------------| --------------| --------------|
| llama3.1-8b--Instruct | bf16 | 1 | PT_HPU_LAZY_MODE=1 ./run.sh -m meta-llama/Meta-Llama-3.1-8B-Instruct -b 170 |
| llama3.1-405b--Instruct | fp8 | 8 | QUANT_CONFIG=/pathto/llama-3.1-405b-instruct-v2/maxabs_quant_g3.json PT_HPU_LAZY_MODE=1 ./run.sh -m meta-llama/Llama-3.1-405B-Instruct -t 8 -b 160 --fp8 |
| granite-3.1-8b-instruct | bf16 | 1 | PT_HPU_LAZY_MODE=1 ./run.sh -m ibm-granite/granite-3.1-8b-instruct -b 132 |
| granite-20b-code-instruct-8k | bf16 | 1 | PT_HPU_LAZY_MODE=1 ENABLE_EXPERIMENTAL_FLAGS=1 AVOID_CD_SPLIT=true ./run.sh -m ibm-granite/granite-20b-code-instruct-8k -b 70 |
| Meta-Llama-3.1-70B-Instruct | bf16 | 4 | PT_HPU_LAZY_MODE=1 ./run.sh -m meta-llama/Meta-Llama-3.1-70B-Instruct -t 4 -b 200 |
| granite-3b-code-instruct-128k | bf16 | 1 | PT_HPU_LAZY_MODE=1 VLLM_CONFIG_HIDDEN_LAYERS=32 VLLM_DECODE_BLOCK_BUCKET_STEP=16 ENABLE_EXPERIMENTAL_FLAGS=1 FUSER_ENABLE_LOW_UTILIZATION=true AVOID_CD_SPLIT=true ./run.sh -m ibm-granite/granite-3b-code-instruct-128k -b 40 |
| granite-34b-code-instruct-8k | bf16 | 1 | PT_HPU_LAZY_MODE=1 VLLM_DECODE_BLOCK_BUCKET_STEP=32 VLLM_CONFIG_HIDDEN_LAYERS=20  ./run.sh  -m ibm-granite/granite-34b-code-instruct-8k  -b 125 --block_size 256 |
| Mistral-Large-Instruct-2407 | bf16 | 4 | PT_HPU_LAZY_MODE=1 ./run.sh -m mistralai/Mistral-Large-Instruct-2407 -t 4 -b 96 --block_size 256 |
| Llama-3.2-90B-Vision-Instruct | bf16 | 4 | PT_HPU_LAZY_MODE=1 ./run.sh -m meta-llama/Llama-3.2-90B-Vision-Instruct -t 4 -b 70 --vision --multistep 32 |
| llama-3.3-70b-instruct | fp8 | 4 | QUANT_CONFIG=/pathto/llama-3.3-70b-instruct/maxabs_quant_g3.json PT_HPU_LAZY_MODE=1 FUSER_ENABLE_LOW_UTILIZATION=1  ./run.sh -m meta-llama/llama-3.3-70b-instruct -t 4 -b 220 --fp8 |
| Mixtral-8x7B-Instruct-v0.1 | bf16 | 1 | PT_HPU_LAZY_MODE=1 ./run.sh -m mistralai/Mixtral-8x7B-Instruct-v0.1 -t 1 -b 110 |
| CodeLlama-34b-Instruct-hf | bf16 | 1 | PT_HPU_LAZY_MODE=1 ./run.sh  -m meta-llama/CodeLlama-34b-Instruct-hf -b 108 |
| granite-8b-code-instruct-128k | bf16 | 1 | VLLM_DECODE_BLOCK_BUCKET_STEP=8 VLLM_CONFIG_HIDDEN_LAYERS=8 VLLM_PROMPT_USE_FUSEDSDPA=true PT_HPU_LAZY_MODE=1 ./run.sh  -m ibm-granite/granite-8b-code-instruct-128k -b 140 |
