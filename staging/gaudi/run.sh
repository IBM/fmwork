#!/bin/bash

set -ex


usage() {
    echo "Usage: $0"
    echo "Options:"
    echo "  --model|-m                  Model path lub model stub"
    echo "  --bs|-b                     Specify the batch size, default: 128"
    echo "  --tp_size|-t                Specify the number of HPUs, default: 8"
    echo "  --fp8                    Enable or Disable fp8/quantization, default disabled"
    echo "  --multistep               Enable multi-step scheduling feature by setting number of scheduled steps larger than 1, default: 1 (feature off)"
    echo "  --help                   Display this help message"
    exit 1
}

get_flavor() {
    flavor=$(python3 -c "import habana_frameworks.torch.hpu as h; print(h.get_device_name()[-1])") || (echo "Detecting device failed" && exit 1)
    echo "g$flavor"
}

function extract_last_folder_name() {
    local path="$1"

    path="${path%/}"
    last_folder="$(basename "$path")"
    last_folder="${last_folder,,}"

    echo "$last_folder"
}

EXTRA_FLAGS=""
tp_size=1
scheduled_steps=1

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model|-m)
            model=$2
            shift 2
            ;;
        --bs|-b)
            batch_size=$2
            shift 2
            ;;
        --fp8)
            fp8="On"
            shift 1
            ;;
        --multistep)
            scheduled_steps=$2
            shift 2
            ;;
        --tp_size|-t)
            tp_size=$2
            shift 2
            ;;
        --help)
            usage
            ;;
        *)
            echo "Invalid option: $1"
            exit 1
            ;;
    esac
done
if [[ -n $HELP ]]; then
    usage
fi

model_name=$(extract_last_folder_name "$model")

export PT_HPU_ENABLE_LAZY_COLLECTIVES=true
export EXPERIMENTAL_WEIGHT_SHARING=0

max_prompt_batch_size=16
input_sizes=1024
output_sizes=1024
block_size=128
max_model_len=$((input_sizes + output_sizes))
max_num_batched_tokens=$((input_sizes * max_prompt_batch_size))

# setup vLLM buckets for static shapes
export VLLM_PROMPT_SEQ_BUCKET_MIN=$input_sizes
export VLLM_PROMPT_SEQ_BUCKET_MAX=$input_sizes
export VLLM_PROMPT_BS_BUCKET_MAX=$max_prompt_batch_size

export VLLM_DECODE_BS_BUCKET_MIN=$batch_size
export VLLM_DECODE_BS_BUCKET_STEP=$batch_size
export VLLM_DECODE_BS_BUCKET_MAX=$batch_size

max_model_len=$((input_sizes + output_sizes))
last_block=$((batch_size * max_model_len / block_size))
last_block=$((last_block + block_size - 1))
last_block=$((last_block / block_size))
last_block=$((last_block * block_size))
last_block=$((last_block + block_size))
export VLLM_DECODE_BLOCK_BUCKET_MAX=$last_block

flavor=$(get_flavor)
if [[ -n "$fp8" ]]; then
    EXTRA_FLAGS+="--quantization inc --kv_cache_dtype fp8_inc --weights_load_device cpu "
    if [[ $model_name =~ llama ]]; then
        export QUANT_CONFIG=/software/data/vllm-benchmarks/inc/$model_name/maxabs_quant_$flavor.json
    else
        export QUANT_CONFIG=inc/maxabs_quant.json
    fi
fi

python driver -m $model -i $input_sizes -o $output_sizes -t $tp_size --dtype bfloat16 --device hpu -b $batch_size --max_num_batched_tokens $max_num_batched_tokens --num_scheduler_steps $scheduled_steps $EXTRA_FLAGS
