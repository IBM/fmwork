# infer/vllm

Inference engine implementation using [vLLM](https://github.com/vllm-project/vllm).

## Usage

### CUDA

```
/path/to/fmwork/infer/vllm/runner
    --mode direct
    --dir_work /path/to/workspace
    --
driver
    --platform cuda
    --model_root /path/to/models
    --model_name meta-llama/Llama-3.1-8B-Instruct/main
    --input_sizes 1024
    --output_sizes 1,128
    --batch_sizes 1,2,4
    --tp_size 1
    --reps 5
    --engine:enable_prefix_caching@ False
    --engine:compilation_config:cudagraph_capture_sizes@ args.batch_sizes
    --engine:max_seq_len_to_capture@ 131072
    --engine:max_num_seqs@ 64
    --batch_multiplier 1
```

The vLLM integration currently has the following scripts:
- `runner`: Environment and experiment set up based on execution `--mode`.
- `driver`: Implementation of vLLM benchmark in direct (offline, static) mode.
- `client`: Client piece of server-mode benchmarking. 
- `server`: Server piece of server-mode benchmarking.
- `process`: Process results.

### Spyre

## Example of output

## More on parameters

## Processing results
