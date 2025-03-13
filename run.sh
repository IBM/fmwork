#!/bin/bash

set -ex

usage() {
    echo "Usage: $0"
    echo "Options:"
    echo "  --model|-m               Model path lub model stub"
    echo "  --bs|-b                  Specify the batch size, default: 128"
    echo "  --tp_size|-t             Specify the number of HPUs, default: 1"
    echo "  --fp8                    Enable or Disable fp8/quantization, default disabled"
    echo "  --vision                 Enable vision model, default disabled"
    echo "  --multistep              Enable multi-step scheduling feature by setting number of scheduled steps larger than 1, default: 1 (feature off)"
    echo "  --block_size             Specify the block size, default is 128 for bf16, 256 for fp8"
    echo "  --mpbs                   Max prompt batch size"
    echo "  --help                   Display this help message"
    exit 1
}

get_flavor() {
    flavor=$(python3 -c "import habana_frameworks.torch.hpu as h; print(h.get_device_name()[-1])") || (echo "Detecting device failed" && exit 1)
    echo "g$flavor"
}

extract_last_folder_name() {
    local path="$1"

    path="${path%/}"
    last_folder="$(basename "$path")"
    last_folder="${last_folder,,}"

    echo "$last_folder"
}

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
        --vision)
            vision="On"
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
        --block_size)
            block_size=$2
            shift 2
            ;;
        --mpbs)
            max_prompt_batch_size=$2
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

# setup vLLM buckets for static shapes
input_sizes=1024
output_sizes=1024
max_model_len=$((input_sizes + output_sizes))
if [[ -z "$block_size" ]]; then
    if [[ -n "$fp8" ]]; then
        block_size=256
    else
        block_size=128
    fi
fi

if [[ -n "$vision" ]]; then
    image_size=256
    num_encoder_tokens=1601 # for current image size
    last_block=$((batch_size * (max_model_len + num_encoder_tokens) / block_size))
else
    last_block=$((batch_size * max_model_len / block_size))
fi

first_block=$((batch_size * input_sizes / block_size))
first_block=$((first_block / block_size * block_size + block_size))

last_block=$((last_block + block_size - 1))
last_block=$((last_block / block_size * block_size + block_size))

if [ -z ${max_prompt_batch_size+x} ]; then
    max_prompt_batch_size=4
fi
max_num_batched_tokens=$((input_sizes * max_prompt_batch_size))

export VLLM_PROMPT_SEQ_BUCKET_MIN=$input_sizes
export VLLM_PROMPT_SEQ_BUCKET_MAX=$input_sizes
export VLLM_PROMPT_BS_BUCKET_MAX=$max_prompt_batch_size
export VLLM_DECODE_BS_BUCKET_MIN=$batch_size
export VLLM_DECODE_BS_BUCKET_STEP=$batch_size
export VLLM_DECODE_BS_BUCKET_MAX=$batch_size
export VLLM_DECODE_BLOCK_BUCKET_MIN=$first_block
export VLLM_DECODE_BLOCK_BUCKET_MAX=$last_block

# setup fp8
EXTRA_FLAGS=""
flavor=$(get_flavor)
if [[ -n "$fp8" ]]; then
    EXTRA_FLAGS+="--quantization inc --kv_cache_dtype fp8_inc --weights_load_device cpu "

    # Check for QUANT_CONFIG environment variable
    if [[ -n "$QUANT_CONFIG" ]]; then  # If set, use it *exclusively*
        export QUANT_CONFIG="$QUANT_CONFIG" # Use the env variable directly
        echo "Using QUANT_CONFIG from environment: $QUANT_CONFIG"
    else  # If not set, error out if fp8 is enabled
        echo "Error: QUANT_CONFIG environment variable must be set when fp8 is enabled."
        exit 1
    fi 
fi

# enable delayed sampling or multi-step scheduling
if [[ $scheduled_steps -eq 1 ]]; then
    export VLLM_DELAYED_SAMPLING=true
fi

# run benchmark
export PT_HPU_ENABLE_LAZY_COLLECTIVES=true
export EXPERIMENTAL_WEIGHT_SHARING=0
export FUSER_ENABLE_LOW_UTILIZATION=true

if [[ -n "$vision" ]]; then
    python driver_vision \
      --model_path $model \
      --image_width $image_size \
      --image_height $image_size \
      --input_size $input_sizes \
      --output_size $output_sizes \
      --batch_size $batch_size \
      --tensor_parallel_size $tp_size \
      --max_num_prefill_seqs $max_prompt_batch_size \
      --num_scheduler_steps $scheduled_steps \
      --quiet_mode \
      $EXTRA_FLAGS
else
    python driver \
      -m $model \
      -i $input_sizes \
      -o $output_sizes \
      -t $tp_size \
      --dtype bfloat16 \
      --device hpu \
      -b $batch_size \
      --block_size $block_size \
      --max_num_batched_tokens $max_num_batched_tokens \
      --num_scheduler_steps $scheduled_steps \
      $EXTRA_FLAGS
fi
