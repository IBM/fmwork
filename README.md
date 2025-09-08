# fmwork

FM Benchmarking Framework

## Quick start

Clone and Install vllm in a v1.22.0 Gaudi3 pre/release docker container 

```
git clone https://github.com/HabanaAI/vllm-fork.git
cd vllm-fork
git checkout v0.9.0.1+Gaudi-1.22.0
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
VLLM_FUSED_BLOCK_SOFTMAX=true ./run.sh -m ibm-granite/granite-3.1-8b-instruct -b 146 --split_qkv

# vision model (include --vision flag)
./run.sh -m meta-llama/Llama-3.2-90B-Vision-Instruct -t 4 -b 74 --vision --multistep 32 --block_bucket_step 128

# fp8 quantization example
QUANT_CONFIG=/pathto/llama-3.1-405b-instruct/maxabs_quant_g3.json ./run.sh -m meta-llama/Llama-3.1-405B-Instruct -t 8 -b 180 --fp8
```

Note: FP8 quantization requires calibration to be done prior to running inferencing. `QUANT_CONFIG` file will need to be passed as a variable before the `run.sh` command.  
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
PT version : 2.7.1 

Gaudi3 Models run command examples 

## Validated Models on Gaudi3

The following list contains models and configurations we have validated on Gaudi3.

Model: granite-3.1-8b-instruct | BF16 | TP=1 
```
VLLM_FUSED_BLOCK_SOFTMAX=true ./run.sh -m ibm-granite/granite-3.1-8b-instruct -b 146 --split_qkv
```

Model: Mistral-Large-Instruct-2407 | BF16 | TP=4 
```
VLLM_FUSED_BLOCK_SOFTMAX=true  ./run.sh -m mistralai/Mistral-Large-Instruct-2407 -t 4 -b 106 --block_size 256
```

Model: Llama-3.2-90B-Vision-Instruct | BF16 | TP=4
```
./run.sh -m meta-llama/Llama-3.2-90B-Vision-Instruct -t 4 -b 74 --vision --multistep 32 --block_bucket_step 128
```

Model: Meta-Llama-3.1-70B-Instruct | BF16 | TP=4
```
VLLM_FUSED_BLOCK_SOFTMAX=true  ./run.sh -m meta-llama/Meta-Llama-3.1-70B-Instruct -t 4 -b 216 --block_size 256
```

Model: granite-3b-code-instruct-128k | BF16 | TP=1
```
VLLM_FUSED_BLOCK_SOFTMAX=true  ./run.sh  -m ibm-granite/granite-34b-code-instruct-8k -b 140 --block_size 256 --split_qkv
```

Model: Mixtral-8x7B-Instruct-v0.1 | BF16 | TP=1 
```
./run.sh -m mistralai/Mixtral-8x7B-Instruct-v0.1 -t 1 -b 112 --split_qkv
```
Model: CodeLlama-34b-Instruct-hf | BF16 | TP=1
```
./run.sh -m meta-llama/CodeLlama-34b-Instruct-hf -b 112 --split_qkv
 ```
Model: granite-3b-code-instruct-128k | BF16 | TP=1
```
VLLM_FUSED_BLOCK_SOFTMAX_ADJUSTMENT=false ./run.sh -m ibm-granite/granite-3b-code-instruct-128k -b 46 --block_bucket_step 16 --split_qkv
```

Model: granite-20b-code-instruct-8k | BF16 | TP=1 
 ```
VLLM_FUSED_BLOCK_SOFTMAX=true ./run.sh -m ibm-granite/granite-20b-code-instruct-8k -b 90 --block_size 256 --split_qkv
 ```

Model: llama3.1-8b--Instruct | BF16 | TP=1
 ```
./run.sh -m meta-llama/Meta-Llama-3.1-8B-Instruct -b 180 --split_qkv
 ```

Model: granite-8b-code-instruct-128k | BF16 | TP=1
 ```
VLLM_FUSED_BLOCK_SOFTMAX=true  ./run.sh  -m ibm-granite/granite-8b-code-instruct-128k -b 162 --split_qkv
 ```

Model: llama3.1-405b--Instruct | BF16 | TP=8
 ```
QUANT_CONFIG=/software/ae/fmwork/inc/llama-3.1-405b-instruct/maxabs_quant_g3.json ./run.sh -m meta-llama/Llama-3.1-405B-Instruct -t 8 -b 180 --fp8
 ```

Model: llama-3.3-70b-instruct | BF16 | TP=4
 ```
QUANT_CONFIG=/software/ae/fmwork/inc/llama-3.3-70b-instruct/maxabs_quant_g3.json ./run.sh -m meta-llama/llama-3.3-70b-instruct -t 4 -b 256 --fp8
 ```

