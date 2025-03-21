## Docker container used:

For all of the workloads execpt below use the container:

```rocm/vllm-dev:nightly_aiter_integration_final_20250312```

**For IBM-Granite 3.1-8B**

```rocm/vllm-dev:nightly_main_20250225```

**For LLaMa 3.2-90B Vision Workload** 

```rocm/vllm-dev:nightly_main_20250317```

## Steps to run the workloads.

1. Clone the below repository for non-vision workloads. 
The below repo contains code to have unique inputs and enable prefix caching support. 

```
git clone --recursive -b rocm_repro https://github.com/lcskrishna/fmwork.git
cd fmwork
```

2. Download the dataset

Example for granite-8b


```
pip install huggingface-hub
huggingface-cli download --cache-dir ./ --local-dir-use-symlinks False --revision main --local-dir models/granite-8b ibm-granite/granite-8b-code-base-128k
```

2. Run the workloads:

```
bash scripts/run_workload.sh --model_name granite-3.1-8b  --model_path <PATH>
bash scripts/run_workload.sh --model_name granite-3b-code-instruct  --model_path <PATH>
bash scripts/run_workload.sh --model_name granite-8b-code-instruct  --model_path <PATH>
bash scripts/run_workload.sh --model_name granite-20b-code-instruct  --model_path <PATH>
bash scripts/run_workload.sh --model_name granite-34b-code-instruct  --model_path <PATH>
bash scripts/run_workload.sh --model_name llama-3.1-70b  --model_path <PATH>
bash scripts/run_workload.sh --model_name llama-3.1-8b  --model_path <PATH>
bash scripts/run_workload.sh --model_name mixtral8x7b  --model_path <PATH>
bash scripts/run_workload.sh --model_name codellama34b-instruct-hf --model_path <PATH>
bash scripts/run_workload.sh --model_name mistral-large-2407  --model_path <PATH>
bash scripts/run_workload.sh --model_name amd-llama3.1-405b-fp8  --model_path <PATH>
bash scripts/run_workload.sh --model_name amd-llama3.3-70b-fp8 --model_path <PATH>
```

Note: MODEL_NAME is a required parameter and MODEL_PATH is optional.



## Steps to run the Vision Workload.

1. git clone -b vision https://github.com/lcskrishna/fmwork.git (Contains David Sandler's changes to input_generate function)
2. cd fmwork
3. VLLM_USE_TRITON_FLASH_ATTN=0 ./driver --model_path models/Llama-3.2-90B-Vision-Instruct --input_size 1024 --output_size 1024  --batch_size 40  --tensor_parallel_size 4  --image_width 256 --image_height 256 



## Steps to do Gradlib Tuning:

An example on Granite-3.1-8B Instruct workload. Make sure to be in /app/fmwork folder.

```
export VLLM_UNTUNE_FILE=/app/vllm_untuned.csv
VLLM_TUNE_GEMM=1 VLLM_USE_TRITON_FLASH_ATTN=0 ./infer/vllm/driver --model_path /data/granite-3.1-8b/ --input_size 1024 --output_size 1024 --batch_size 128 --tensor_parallel 1 --dtype bfloat16 --num_scheduler_steps 32 --enable_prefix_caching
git clone --recursive https://github.com/ROCm/vllm.git
python vllm/gradlib/gemm_tuner.py --outdtype bf16 --input_file /app/vllm_untuned.csv --tuned_file /app/tuned_ibm_granite8b.csv
unset VLLM_TUNE_GEMM
VLLM_TUNE_FILE=/app/tuned_ibm_granite8b.csv VLLM_USE_TRITON_FLASH_ATTN=0 ./infer/vllm/driver --model_path /data/granite-3.1-8b/ --input_size 1024 --output_size 1024 --batch_size 128 --tensor_parallel 1 --dtype bfloat16 --num_scheduler_steps 32 --enable_prefix_caching

```

## Steps to Perform tunable ops.

Please refer to the tunableops documentation in PyTorch:
https://github.com/pytorch/pytorch/tree/main/aten/src/ATen/cuda/tunable

### Online Tuning: (While running the workload)

```
PYTORCH_TUNABLEOP_ENABLED=1 PYTORCH_TUNABLEOP_TUNING=1 <cmd>
```

The above will generate a tunableop_results%d.csv file which has all the tuned solutions.

To use the above tuning file run the command using the below.

```
PYTORCH_TUNABLEOP_ENABLED=1 PYTORCH_TUNABLEOP_TUNING=0 <cmd>
```


### Offline Tuning:

Set the below environment variables:
```
PYTORCH_TUNABLEOP_ENABLED=1
PYTORCH_TUNABLEOP_TUNING=0
PYTORCH_TUNABLEOP_RECORD_UNTUNED=1
```

This will dump out a tunableop_untuned0.csv file which can be used for tuning with the below script. 

```
import torch.cuda.tunable as tunable
import os

os.putenv('PYTORCH_TUNABLEOP_ENABLED', '1')
os.putenv('PYTORCH_TUNABLEOP_TUNING', '1')
os.putenv('PYTORCH_TUNABLEOP_RECORD_UNTUNED', '0')

if __name__ == "__main__":
    num_gpus = 8 # number of GPUs that will be used during the tuning process
    tunable.mgpu_tune_gemm_in_file("tunableop_untuned0.csv", num_gpus)

```

After the dumping is finished - it generates couple of CSV files with tunableop_results_full%d.csv. 
Make sure to pass the CSV file using: PYTORCH_TUNABLEOP_FILENAME=tunableop_results_full%d.csv

**A generic command combined with Grad Lib + Tunable Ops will look like this.**

```
VLLM_TUNE_FILE=/app/tuned_gemm_gradlib_granite20b.csv PYTORCH_TUNABLEOP_ENABLED=1 PYTORCH_TUNABLEOP_TUNING=0 PYTORCH_TUNABLEOP_FILENAME=tunableop_results_full%d.csv <CMD>

```
