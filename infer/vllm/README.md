# infer/vllm

Inference engine implementation using [vLLM](https://github.com/vllm-project/vllm).

## Usage examples

### CUDA, direct mode

```bash
/path/to/fmwork/infer/vllm/runner \
    --dir_work /path/to/workspace \
    --mode direct \
    --model_root /path/to/models \
    --model_name meta-llama/Llama-3.1-8B-Instruct \
    --env PYTHONUNBUFFERED=1 \
    --env VLLM_USE_V1=1 \
    -- \
driver \
    --platform cuda \
    --input_sizes 1024 \
    --output_sizes 1,128 \
    --batch_sizes 1 \
    --tp_size 1 \
    --reps 5 \
    --engine:enable_prefix_caching@ False \
    --engine:compilation_config:cudagraph_capture_sizes@ args.batch_sizes \
    --engine:max_seq_len_to_capture@ 131072 \
    --engine:max_num_seqs@ 64 \
    --batch_multiplier 1
```

### CUDA, server mode

```bash
/path/to/fmwork/infer/vllm/runner \
    --dir_work /path/to/workspace \
    --mode server \
    --model_root /path/to/models \
    --model_name meta-llama/Llama-3.1-8B-Instruct  \
    -- \
server \
    --env PYTHONUNBUFFERED=1 \
    --env VLLM_USE_V1=1 \
    --tensor-parallel-size 1 \
    --no-enable-prefix-caching \
    --max-num-seqs 1 \
    -- \
client \
    --env PYTHONUNBUFFERED=1 \
    --dataset-name random \
    --random-input-len 1024 \
    --random-output-len 128 \
    --num-prompts 128
```

### Spyre, direct mode, CB disabled

```bash
/path/to/fmwork/infer/vllm/runner \
    --dir_work /path/to/workspace \
    --mode direct \
    --model_root /path/to/models \
    --model_name meta-llama/Llama-3.1-8B-Instruct  \
    --env PYTHONUNBUFFERED=1 \
    --env DTLOG_LEVEL=error \
    --env DT_DEEPRT_VERBOSE=-1 \
    --env DTCOMPILER_KEEP_EXPORT=-1 \
    --env TORCH_SENDNN_LOG=CRITICAL \
    --env VLLM_USE_V1=1 \
    --env VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
    --env FLEX_RDMA_MODE_FULL=FALSE \
    --env FLEX_HDMA_MODE_FULL=1 \
    --env OMP_NUM_THREADS=32 \
    --env SENDNN_INFERENCE_WARMUP_PROMPT_LENS=1024 \
    --env SENDNN_INFERENCE_WARMUP_NEW_TOKENS=128 \
    --env SENDNN_INFERENCE_WARMUP_BATCH_SIZES=1 \
    -- \
driver \
    --platform spyre \
    --input_sizes 1024 \
    --output_sizes 1,128 \
    --batch_sizes 1 \
    --tp_size 4 \
    --engine:max_model_len@ 2048 \
    --engine:max_num_seqs@ 1 \
    --engine:enable_prefix_caching@ False \
    --engine:compilation_config:cudagraph_capture_sizes@ args.batch_sizes \
    --engine:max_seq_len_to_capture@ 131072 \
    --batch_multiplier 1 \
    --reps 5
```

### Spyre, direct mode, CB enabled

```bash
/path/to/fmwork/infer/vllm/runner \
    --dir_work /path/to/workspace \
    --mode direct \
    --model_root /path/to/models \
    --model_name meta-llama/Llama-3.1-8B-Instruct  \
    --env PYTHONUNBUFFERED=1 \
    --env DTLOG_LEVEL=error \
    --env DT_DEEPRT_VERBOSE=-1 \
    --env DTCOMPILER_KEEP_EXPORT=-1 \
    --env TORCH_SENDNN_LOG=CRITICAL \
    --env VLLM_USE_V1=1 \
    --env VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
    --env FLEX_RDMA_MODE_FULL=FALSE \
    --env FLEX_HDMA_MODE_FULL=1 \
    --env OMP_NUM_THREADS=32 \
    --env SENDNN_INFERENCE_USE_CB=1 \
    -- \
driver \
    --platform spyre \
    --input_sizes 1024 \
    --output_sizes 1,128 \
    --batch_sizes 1 \
    --tp_size 4 \
    --engine:max_model_len@ 2048 \
    --engine:max_num_seqs@ 1 \
    --engine:enable_prefix_caching@ False \
    --engine:compilation_config:cudagraph_capture_sizes@ args.batch_sizes \
    --engine:max_seq_len_to_capture@ 131072 \
    --batch_multiplier 1 \
    --reps 5
```

### Spyre, server mode, CB disabled

```bash
/path/to/fmwork/infer/vllm/runner \
    --dir_work /path/to/workspace \
    --dir_pref 20250813-tests/005 \
    --mode server \
    --model_root /path/to/models  \
    --model_name meta-llama/Llama-3.1-8B-Instruct  \
    -- \
server \
    --env PYTHONUNBUFFERED=1 \
    --env DTLOG_LEVEL=error \
    --env DT_DEEPRT_VERBOSE=-1 \
    --env DTCOMPILER_KEEP_EXPORT=-1 \
    --env TORCH_SENDNN_LOG=CRITICAL \
    --env VLLM_USE_V1=1 \
    --env VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
    --env FLEX_RDMA_MODE_FULL=FALSE \
    --env FLEX_HDMA_MODE_FULL=1 \
    --env OMP_NUM_THREADS=32 \
    --env SENDNN_INFERENCE_WARMUP_PROMPT_LENS=1024 \
    --env SENDNN_INFERENCE_WARMUP_NEW_TOKENS=128 \
    --env SENDNN_INFERENCE_WARMUP_BATCH_SIZES=1 \
    --no-enable-prefix-caching \
    --max-model-len 2048 \
    --max-num-seqs 1 \
    --tensor-parallel-size 4 \
    -- \
client \
    --env PYTHONUNBUFFERED=1 \
    --dataset-name random \
    --random-input-len 1024 \
    --random-output-len 128 \
    --num-prompts 16
```

### Spyre, server mode, CB enabled

```bash
/path/to/fmwork/infer/vllm/runner \
    --dir_work /path/to/workspace \
    --mode server \
    --model_root /path/to/models  \
    --model_name meta-llama/Llama-3.1-8B-Instruct  \
    -- \
server \
    --env PYTHONUNBUFFERED=1 \
    --env DTLOG_LEVEL=error \
    --env DT_DEEPRT_VERBOSE=-1 \
    --env DTCOMPILER_KEEP_EXPORT=-1 \
    --env TORCH_SENDNN_LOG=CRITICAL \
    --env VLLM_USE_V1=1 \
    --env VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
    --env FLEX_RDMA_MODE_FULL=FALSE \
    --env FLEX_HDMA_MODE_FULL=1 \
    --env OMP_NUM_THREADS=32 \
    --env SENDNN_INFERENCE_USE_CB=1 \
    --no-enable-prefix-caching \
    --max-model-len 2048 \
    --max-num-seqs 1 \
    --tensor-parallel-size 4 \
    -- \
client \
    --env PYTHONUNBUFFERED=1 \
    --dataset-name random \
    --random-input-len 1024 \
    --random-output-len 128 \
    --num-prompts 16
```

## Processing results

Example of outputs from first example above (executed on NVIDIA H100):

```
FMWORK SETUP 65.255954

--------------------------------------------------------------------------------
RUN 1024 / 1 / 1
--------------------------------------------------------------------------------

/net/storage149/mnt/md0/nmg/projects/fmwork/github.com/IBM/dev/fmwork/infer/vllm/driver:159: DeprecationWarning: The keyword arguments {'prompt_token_ids'} are deprecated and will be removed in a future update. Please use the 'prompts' parameter instead.
  timings = bench_combo_rep(
FMWORK REP 1 5 1755063804.569356785 1755063811.294698536 meta-llama/Llama-3.1-8B-Instruct/main 1024 1 1 1 6.725341751 6725.342 0.1
FMWORK REP 2 5 1755063811.295244652 1755063811.324200424 meta-llama/Llama-3.1-8B-Instruct/main 1024 1 1 1 0.028955772 28.956 34.5
FMWORK REP 3 5 1755063811.325461863 1755063811.352333914 meta-llama/Llama-3.1-8B-Instruct/main 1024 1 1 1 0.026872051 26.872 37.2
FMWORK REP 4 5 1755063811.353573998 1755063811.380560835 meta-llama/Llama-3.1-8B-Instruct/main 1024 1 1 1 0.026986837 26.987 37.1
FMWORK REP 5 5 1755063811.381792484 1755063811.409563334 meta-llama/Llama-3.1-8B-Instruct/main 1024 1 1 1 0.027770850 27.771 36.0

Timestamp start               = 1755063804.569356785
Timestamp end                 = 1755063811.409563334
Model name                    = meta-llama/Llama-3.1-8B-Instruct/main
Input size                    = 1024
Output size                   = 1
Batch size                    = 1
Batch size multiplier         = 1
Tensor parallel size          = 1
Relative med. abs. dev.       = 0.016
RES: Inference time (s)       = 0.027
RES: Inter-token latency (ms) = 27.379
RES: Throughput (tok/s)       = 36.5

FMWORK RES 1755063804.569356785 1755063811.409563334 meta-llama/Llama-3.1-8B-Instruct/main 1024 1 1 1 0.016 0.027 27.379 36.5

--------------------------------------------------------------------------------
RUN 1024 / 128 / 1
--------------------------------------------------------------------------------

FMWORK REP 1 5 1755063811.412279354 1755063812.404891550 meta-llama/Llama-3.1-8B-Instruct/main 1024 128 1 1 0.992612196 7.755 129.0
FMWORK REP 2 5 1755063812.406126905 1755063813.361175403 meta-llama/Llama-3.1-8B-Instruct/main 1024 128 1 1 0.955048498 7.461 134.0
FMWORK REP 3 5 1755063813.362408263 1755063814.279993320 meta-llama/Llama-3.1-8B-Instruct/main 1024 128 1 1 0.917585057 7.169 139.5
FMWORK REP 4 5 1755063814.281200124 1755063815.197474959 meta-llama/Llama-3.1-8B-Instruct/main 1024 128 1 1 0.916274835 7.158 139.7
FMWORK REP 5 5 1755063815.198693247 1755063816.115997318 meta-llama/Llama-3.1-8B-Instruct/main 1024 128 1 1 0.917304071 7.166 139.5

Timestamp start               = 1755063811.412279354
Timestamp end                 = 1755063816.115997318
Model name                    = meta-llama/Llama-3.1-8B-Instruct/main
Input size                    = 1024
Output size                   = 128
Batch size                    = 1
Batch size multiplier         = 1
Tensor parallel size          = 1
Relative med. abs. dev.       = 0.001
RES: Inference time (s)       = 0.917
RES: Inter-token latency (ms) = 7.168
RES: Throughput (tok/s)       = 139.5

FMWORK RES 1755063811.412279354 1755063816.115997318 meta-llama/Llama-3.1-8B-Instruct/main 1024 128 1 1 0.001 0.917 7.168 139.5

Timestamp start                        = 1755063811.412279354
Timestamp end                          = 1755063816.115997318
Model name                             = meta-llama/Llama-3.1-8B-Instruct/main
Input size                             = 1024
Output size                            = 128
Batch size                             = 1
Batch size multiplier                  = 1
Tensor parallel size                   = 1
Relative med. abs. dev. TTFT           = 0.016
Relative med. abs. dev. INF            = 0.001
GEN: [ INF  ] Inference time (s)       = 0.917
GEN: [ GEN  ] Generation time (s)      = 0.890
GEN: [ TTFT ] Time to first token (s)  = 0.027
GEN: [ ITL  ] Inter-token latency (ms) = 6.954
GEN: [ THP  ] Throughput (tok/s)       = 143.8

FMWORK GEN 1755063811.412279354 1755063816.115997318 meta-llama/Llama-3.1-8B-Instruct/main 1024 128 1 1 0.016 0.001 0.917 0.890 0.027 6.954 143.8
```

### Logs to JSON

The script automatically detects whether to parse logs.

```
python process.py --path <PATH_TO_EXPERIMENT_FOLDER> [OPTIONAL_ARGS]
```
| Argument | Required | Description |
| :--- | :--- | :--- |
| **`--path`** | **Yes** | The directory containing the benchmark log and command files (`*.log`, `*.cmd`). |
| `--model` | No | Expected base model name for version extraction. |
| `--precision` | No | Default model precision (e.g., `fp16`). |
| `--enable_prom_metrics` | No | Flag to enable processing of Prometheus CPU/Memory metrics. |

#### JSON Output Structure (Key Fields)

The script outputs a list of JSON objects, one for each run (or each client in multi-client mode).

| Field | Description | Source |
| :--- | :--- | :--- |
| `model` | Model name and version. | Command/Log |
| `input`, `output` | Input and output token lengths. | Command/Log |
| `batch`, `tp` | Batch size and Tensor Parallel size. | Command/Log |
| `ttft` | **Time To First Token** (s). | Log Metric |
| `itl` | **Inter-Token Latency** (ms). | Log Metric |
| `thp` | **Output Token Throughput** (tok/s). | Log Metric |
| `req_thp` | **Request Throughput** (req/s). | Calculated |
| `status` | Run status (`OK`, `PART_`, `ERR_`). | Log Check |

```json
[
    {
        "timestamp": "1755063811.412279354",
        "metadata_id": null,
        "engine": "fmwork/infer/vllm",
        "model": "meta-llama/Llama-3.1-8B-Instruct/main",
        "precision": null,
        "input": 1024,
        "output": 128,
        "batch": 1,
        "tp": 1,
        "opts": [
            "--env PYTHONUNBUFFERED=1",
            "--env VLLM_USE_V1=1",
            "--batch_multiplier 1",
            "--batch_sizes 1",
            "--engine:compilation_config:cudagraph_capture_sizes@ args.batch_sizes",
            "--engine:enable_prefix_caching@ False",
            "--engine:max_num_seqs@ 64",
            "--engine:max_seq_len_to_capture@ 131072",
            "--input_sizes 1024",
            "--model_name meta-llama/Llama-3.1-8B-Instruct/main",
            "--model_root /net/storage149/autofs/css22/nmg/models/hf",
            "--output_sizes 1,128",
            "--platform cuda",
            "--reps 5",
            "--tp_size 1"
        ],
        "warmup": null,
        "setup": 65.255954,
        "ttft": 0.027,
        "itl": 6.954,
        "thp": 143.8
    }
]
```
### Logs to CSV

`log2csv.sh` is a **Shell script** to search a top-level directory, extract key performance metrics and run status from all individual test runs, and prints the result to **stdout** in **CSV** format.

#### Usage

Provide the path to the **top-level directory** containing multiple experiment folders.

```bash
# Redirect the output to a CSV file
./log2csv.sh <PATH_TO_TOP_LEVEL_EXPERIMENT_DIR> > results.csv
```

#### Output CSV Fields (Example Headers)

The script outputs a fixed set of fields for each run:

| Field Name | Description | Source |
| :--- | :--- | :--- |
| `path` | Full path to the experiment directory. | `find` |
| `status` | Run status (`OK`, `OOR`, `RPC`, etc.). | Log Check |
| `E2E` | End-to-end benchmark duration (Server Mode). | `client.log` |
| `RAW_TTFT` | Raw HW (Direct Mode) or Server-side Average (Server Mode) (seconds).| `metrics.log` calculated |
| `Median_TTFT` | Median Time To First Token (milliseconds). | `client.log` |
| `Median_ITL` | Median Inter-Token Latency (milliseconds). | `client.log` |