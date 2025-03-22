#!/usr/bin/bash
current_date=$(date +"%Y%m%d")
flagsuffix="mi300_run"
OUT_DIR=${current_date}_${flagsuffix}
mkdir -p ${OUT_DIR}

MODEL_BATCH_CONFIG=(
        # IBM Granite models
        "ibm-granite/granite-3.1-8b-instruct 128"
        "ibm-granite/granite-3b-code-instruct-128k 56"
        "ibm-granite/granite-8b-code-instruct-128k 128"
        "ibm-granite/granite-20b-code-instruct-8k 64"
        "ibm-granite/granite-34b-code-instruct-8k 88"

        # Meta and Mistral Models
        "meta-llama/Llama-3.1-70B-Instruct 96"
        "meta-llama/Llama-3.2-90B-Vision-Instruct 4" # Check Batch Size
        "mistralai/Mixtral-8x7B-Instruct-v0.1 96"
        "meta-llama/CodeLlama-34b-Instruct-hf 32"    # Check Batch Size
        "meta-llama/Llama-3.1-8B-Instruct 104"
        "mistralai/Mistral-Large-Instruct-2407 64"

        # FP8 Models
        "meta-llama/Llama-3.1-405B-Instruct 120"
        "meta-llama/llama-3-3-70b-instruct 158"
        )

# Model Parameters
INP_SIZE=1024
OUT_SIZE=1024

#Scheduler Steps
NUM_SCHEDULER_STEPS=32

for MODEL_BATCH in  "${MODEL_BATCH_CONFIG[@]}"
do
    set -- $MODEL_BATCH
    MODEL_NAME=$1
    BATCH_SIZE=$2
    MODEL="${MODEL_NAME##*/}"
    QUANT=""
    TENSOR_PARALLEL=1
    if [[ $MODEL == "Llama-3.1-405B-Instruct" ]]; then
        TENSOR_PARALLEL=8
    elif [[ $MODEL == "Mistral-Large-Instruct-2407" ]] || [[ $MODEL == "Llama-3.1-70B-Instruct" ]] || [[ $MODEL == "Llama-3.2-90B-Vision-Instruct" ]] || [[ $MODEL == "llama-3-3-70b-instruct" ]]; then
        TENSOR_PARALLEL=4
    fi

    DTYPE="bfloat16"
    if [[ $MODEL == "Llama-3.1-405B-Instruct" ]] || [[ $MODEL == "llama-3-3-70b-instruct" ]]; then
        DTYPE="bfloat16"
        QUANT=" --quantization fp8 --kv-cache-dtype fp8 "
    fi
    #echo ${MODEL}, ${NUM_SCHEDULER_STEPS}, ${TENSOR_PARALLEL}, ${INP_SIZE}, ${OUT_SIZE}, ${BATCH_SIZE}, ${DTYPE}
    huggingface-cli download --cache-dir ./ --local-dir-use-symlinks False --revision main --local-dir models/${MODEL} ${MODEL_NAME}
    VLLM_USE_AITER_PAGED_ATTN=1 VLLM_USE_AITER=1 VLLM_USE_TRITON_FLASH_ATTN=0 ./fmwork/infer/vllm/driver --model_path models/${MODEL} --num_scheduler_steps ${NUM_SCHEDULER_STEPS} --input_size ${INP_SIZE} --output_size ${OUT_SIZE} --batch_size ${BATCH_SIZE} --tensor_parallel ${TENSOR_PARALLEL} --dtype ${DTYPE}  ${QUANT} 2>&1 | tee ${OUT_DIR}/${MODEL}.txt
done
