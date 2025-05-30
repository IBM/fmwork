#!/usr/bin/bash
FMWORK_PATH=/myworkspace/fmwork

current_date=$(date +"%Y%m%d")
flagsuffix="mi300_run"
OUT_DIR=${current_date}_${flagsuffix}
mkdir -p ${OUT_DIR}

# Parse named arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --model_name) MODEL_NAME="$2"; shift ;;
        --model_path) DATA_PATH="$2"; shift ;;
        --gradlibTuning) GRADLIB_TUNING="$2"; shift ;;
        --tunableOp) TUNABLEOP="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; usage ;;
    esac
    shift
done

if [[ -z "$MODEL_NAME" || -z "$DATA_PATH" ]]; then
    echo "ERROR: required parameter missing --model_name or --model_path"
    exit
fi

declare -A MODEL_BATCH_CONFIG

MODEL_BATCH_CONFIG=(
    ["granite-3.1-8b"]="128"
    ["granite-3b-code-instruct"]="56"
    ["granite-8b-code-instruct"]="128"
    ["granite-20b-code-instruct"]="64"
    ["granite-34b-code-instruct"]="88"
    ["llama-3.1-70b"]="96"
    ["llama-3.1-8b"]="104"
    ["mixtral8x7b"]="96"
    ["codellama34b-instruct-hf"]="96"
    ["mistral-large-2407"]="64"
    ["amd-llama3.1-405b-fp8"]="120"
    ["amd-llama3.3-70b-fp8"]="72"
)

if [[ -v MODEL_BATCH_CONFIG["$MODEL_NAME"] ]]; then
    BATCH_SIZE="${MODEL_BATCH_CONFIG[$MODEL_NAME]}"
    echo "$BATCH_SIZE selected"
else
    echo "ERROR: $MODEL_NAME not found"
    echo "Available options : granite-3.1-8b"
    exit

fi

#
# Model Parameters
INP_SIZE=1024
OUT_SIZE=1024

#Scheduler Steps
NUM_SCHEDULER_STEPS=32


# Tensor Parallel 1 as default.
TENSOR_PARALLEL=1
if [[ $MODEL_NAME == "llama-3.1-405b" ]]; then
    TENSOR_PARALLEL=8
elif [[ $MODEL_NAME == "mistral-large-2407" ]]; then
    TENSOR_PARLLEL=4
fi


DTYPE="bfloat16"
QUANT=""
if [[ $MODEL_NAME == "amd-llama3.1-405b-fp8" ]] || [[ $MODEL_NAME == "amd-llama3.3-70b-fp8" ]]; then
    QUANT=" --quantization fp8 --kv_cache_dtype fp8"
fi
ENABLE_PREFIX_CACHING="--enable_prefix_caching"

DIST_BACKEND=""
if [[ $MODEL_NAME == "llama-3.1-8b" ]]; then
    DIST_BACKEND="--distributed_executor_backend ray"
fi

echo "Running model=$MODEL_NAME, num_scheduler_steps=$NUM_SCHEDULER_STEPS, tensor_parallel=$TENSOR_PARALLEL, input_size=$INP_SIZE, output_size=$OUT_SIZE, batch_size=$BATCH_SIZE, dtype=$DTYPE, quant=$QUANT"


## Use CK backend.
export VLLM_USE_TRITON_FLASH_ATTN=0

# use AITER Paged Attention.
if [[ $MODEL_NAME == "granite-20b-code-instruct" ]] || [[ $MODEL_NAME == "granite-34b-code-instruct" ]] || [[ $MODEL_NAME == "amd-llama3.3-70b-fp8" ]]; then
    export VLLM_USE_AITER=1
    export VLLM_USE_AITER_PAGED_ATTN=1 
fi

# Enable AITER Fused MoE
if [[ $MODEL_NAME == "mixtral8x7b" ]]; then
    export VLLM_USE_AITER=1
fi

## StreamK Math library option.
if [[ $MODEL_NAME == "granite-3b-code-instruct" ]] || [[ $MODEL_NAME == "amd-llama3.1-405b-fp8" ]]; then
    export TENSILE_SOLUTION_SELECTION_METHOD=2
fi

if [[ $MODEL_NAME == "amd-llama3.1-405b-fp8" ]]; then
    export NCCL_MIN_NCHANNELS=112
fi

MODEL_PARAMS="--model_path $DATA_PATH --num_scheduler_steps $NUM_SCHEDULER_STEPS --input_size $INP_SIZE --output_size $OUT_SIZE --batch_size $BATCH_SIZE --tensor_parallel $TENSOR_PARALLEL --dtype $DTYPE $ENABLE_PREFIX_CACHING $QUANT $DIST_BACKEND"

# Run the workload without gradlib tuning and tunableop tuning
${FMWORK_PATH}/infer/vllm/driver ${MODEL_PARAMS} 2>&1 | tee ${OUT_DIR}/${MODEL_NAME}.txt

# Gradlib Tuning Procedure
if [[ $GRADLIB_TUNING == "True" ]]; then
    # First Generate the untuned csv file
    VLLM_UNTUNE_FILE=/app/vllm_untuned.csv VLLM_TUNE_GEMM=1 ${FMWORK_PATH}/infer/vllm/driver ${MODEL_PARAMS} 2>&1 | tee ${OUT_DIR}/${MODEL_NAME}_untuned_gradlib.txt

    # git clone --recursive https://github.com/ROCm/vllm.git
    # Do offline gradlib tuning 
    python3 gemm_tuner.py --outdtype bf16 --input_file /app/vllm_untuned.csv --tuned_file /app/${MODEL_NAME}_gradlib_tuned.csv

    # Run with tuned to get improved numbers
    VLLM_TUNE_FILE=/app/${MODEL_NAME}_gradlib_tuned.csv ${FMWORK_PATH}/infer/vllm/driver ${MODEL_PARAMS} 2>&1 | tee ${OUT_DIR}/${MODEL_NAME}_tuned_gradlib.txt
fi

# TunableOp Procedure
if [[ $TUNABLEOP == "True" ]]; then
    # Following command will generate tunableop_untuned0.csv
    PYTORCH_TUNABLEOP_ENABLED=1 PYTORCH_TUNABLEOP_TUNING=0 PYTORCH_TUNABLEOP_RECORD_UNTUNED=1 ${FMWORK_PATH}/infer/vllm/driver ${MODEL_PARAMS} 2>&1 | tee ${OUT_DIR}/${MODEL_NAME}_untuned_tunableops.txt

    # Next run offline tuning to generate tuned csv file
    python3 tunableops_offline_tuning.py

    # Run with tuned to get improved numbers
    PYTORCH_TUNABLEOP_ENABLED=1 PYTORCH_TUNABLEOP_TUNING=0 PYTORCH_TUNABLEOP_FILENAME=tunableop_results_full%d.csv ${FMWORK_PATH}/infer/vllm/driver ${MODEL_PARAMS} 2>&1 | tee ${OUT_DIR}/${MODEL_NAME}_tuned_tunableops.txt
fi

# Both Tunableops and gradlib tuning
if [[ $GRADLIB_TUNING == "True" ]] && [[ $TUNABLEOP == "True" ]]; then
    VLLM_TUNE_FILE=/app/${MODEL_NAME}_gradlib_tuned.csv PYTORCH_TUNABLEOP_ENABLED=1 PYTORCH_TUNABLEOP_TUNING=0 PYTORCH_TUNABLEOP_FILENAME=tunableop_results_full%d.csv ${FMWORK_PATH}/infer/vllm/driver ${MODEL_PARAMS} 2>&1 | tee ${OUT_DIR}/${MODEL_NAME}_tuned_gradlib_tunableops.txt
fi
