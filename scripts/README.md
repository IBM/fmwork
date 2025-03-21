## Docker container used:

rocm/vllm-dev:nightly_aiter_integration_final_20250312

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
bash scripts/run_workload.sh <MODEL_NAME> [MODEL_PATH]
```

Note: MODEL_NAME is a required parameter and MODEL_PATH is optional.
