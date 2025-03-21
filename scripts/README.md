## Docker container used:

For all of the workloads except for below two use this container:
```
rocm/vllm-dev:nightly_aiter_integration_final_20250312
```

**For IBM-Granite 3.1-8B** 

```rocm/vllm-dev:nightly_main_20250225```


**For LLaMa-3.2-90B-Vision-Instruct Workload **

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
