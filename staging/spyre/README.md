# fmwork

FM Benchmarking Framework

## Quick start on Openshift Cluster

### 1. Create a pod on cluster:
```bash
oc apply -f pod-fmwork.yaml  # updated on July
oc apply -f pod-fmwork-wxpe.yaml # it's used before July 2025
```
**Note:**
In this yaml, please modify `name`, `namespace`, `imagePullSecrets`, `persistentVolumeClaim`

### 2. Steps required by cluster and image:
```
bash -l
# Check if .senlib.json exists under ${HOME}/.senlib.json, if not please follow the next step and if it exists please skip it
# need to add https://github.ibm.com/ai-foundation/aiu-inference-dev/blob/main/dd2/.senlib.json in $HOME 
vi ${HOME}/.senlib.json
source /opt/vllm/bin/activate
```
### 3. Downloaded HF model: `ibm-granite/granite-3.3-8b-instruct`
```
pip install huggingface-hub
huggingface-cli download --cache-dir ./ --local-dir-use-symlinks False --revision main --local-dir <model_path> ibm-granite/granite-3.3-8b-instruct
```
### 4. Environment variables we ran with (06/25/2025):
```
export VLLM_USE_V1=1 # should be by default
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
# to run with Host DMA due to the switch structure of the cluster
export FLEX_RDMA_MODE_FULL=FALSE
export FLEX_HDMA_MODE_FULL=1

export OMP_NUM_THREADS=32
```
### 5. Run experiments in the pod:
- the command if you want to have TTFT numbers:

```
./infer/vllm/driver --model_path ${model_path} --input_size 1024 --output_size 1,128 --batch_size 4 --tensor_parallel 4 --rep 5 --batch_size_times 100
```
Note on 07/07/2025:
We introduce a new benchmarking methodology to address batch size scaling fidelity.
For batch size N > 1, we now feed significantly more than N samples (instead of exactly N) during benchmarking.
This saturates the engine and amortizes efficiency losses from the first and last batch, ensuring more representative TTFT/ITL results.

* In this setup, `--batch_size_times` controls how many requests are sent to the engine.

* Specifically, the engine receives `batch_size_times × batch_size` total requests.

* Importantly, only one compiled engine is used, shaped to the specified `--batch_size`.

* For all reported numbers (from both `FMWORK GEN` and `FMWORK RES`), we have already divided the raw metrics by `batch_size_times`, so the values are properly normalized.

e.g., `--batch_size 4 --batch_size_times 100` will issue 400 requests, all using a batch size-4 compiled engine. Reported TTFT/ITL will reflect per-inference averages.

**Note** on 07/11/2025: You can now enable PyTorch profiler via `--enable_profiler [optional_output_dir]`.  
Tracing runs only at **rep == 1** and does **not** affect reported TTFT/ITL metrics.

* You can simply run:  
  `--enable_profiler` → traces will be saved to `./vllm_profile` (default)

* Or specify a custom output path:  
  `--enable_profiler /tmp/trace_dir`

* Trace filenames are auto-generated (e.g., include timestamps); no fixed naming.

* If profiling fails, the benchmark continues normally without interruption.


###  6. output example:
This should produce blocks of outputs like:
```
FMWORK GEN 20250621-014836.577621 1024 128 1 4 6.831 6.473 0.358 50.6 19.8

TTFT / GEN Metrics
------------------

Experiment timestamp:            20250621-014836.577621
Model path:                      /mnt/home/zhuoran/fmwork/staging/spyre/models/ibm-granite/granite-3.3-8b-instruct
Input size:                      1024
Output size:                     128
Batch size:                      1
Tensor parallel size:            4
MED  - Median iteration (s):     6.831
TTFT - Time to first token (s):  0.358
GEN  - Generation time (s):      6.473
ITL  - Inter-token latency (ms): 50.6
THP  - Throughput (tok/s):       19.8

Settings for Spyre
------------------

OMP_NUM_THREADS:                 32
FLEX_HDMA_MODE_FULL:             1
FLEX_RDMA_MODE_FULL:             None
COLL_ALLREDUCE_ALGO:             None
```

### 7. Convert logs to JSON payloads json (for database)

Assume we ran inference with `vLLM=v1` and captured benchmark logs as `exp.log` files, e.g.:

**Note**: `v1/<model_name>/<precision>` is required in the structure for the following parsing steps.
````
./infer/vllm/driver --model_path ${model_path} --input_size 1024 --output_size 1,128 --batch_size 4 --tensor_parallel 4 --rep 5   2>&1  | tee  <output_path>/v1/<model_name>/<precision>/<name>.txt
````
To generate metadata_id, we need to run this inside the image:

```
git clone https://github.ibm.com/ai-foundation/transformer-ft-eval.git -b meta_gen
cd transformer-ft-eval
export IMAGE_URL=<image url>
python gen_metadata.py
```
To generate a JSON file from all logs under a specified folder (automatically finding all `exp.log` files recursively):
```
find <path to the folder>/ -name exp.log | xargs python jsonfy-metadataid.py \
  --metadata_id <your_metadata_id> \
  --model ibm-granite/granite-3.3-8b-instruct \
  --precision bf16 \
  --output <output_json_file>
```

* `--path`: Root directory containing model/precision folders with FMWORK logs.



* `--metadata_id`: generated as the above.
* `--model`: Required. Hugging Face model ID.
* `--precision`: Required. Model precision (e.g., bf16, fp16, fp8).
* `--output`: Path to save the final JSON file.
* `--opts` (optional): Custom configuration string. If present, it will be appended to system settings parsed from the logs (under `Settings for Spyre`). 
* `--debug` (optional): Print parsed results for inspection.

**Note**: 
* The value of the "model" field in the JSON payload must be a **standard Hugging Face model ID**, as required by the metadata database schema.
* If `Settings for Spyre` is found in the logs, it will be parsed into `opts` automatically. The `--opts` string will be appended to those values. `opts` is now stored as a list of strings.

Example command to upload:
```
curl -X POST \
  'https://aiu-benchmark.apps.cash-washington-01-satellite.cash.res.ibm.com/v1/benchmark' \
  -H 'accept: application/json' \
  -H 'X-API-Key: <API-key>' \
  -H 'Content-Type: application/json' \
  -d @<output json file>
```


## Quick start on bare metal (TBD)

Get a model (e.g., https://huggingface.co/ibm-granite/granite-3.0-8b-instruct):

```
pip install huggingface-hub
huggingface-cli download --cache-dir ./ --local-dir-use-symlinks False --revision main --local-dir models/granite-3.0-8b ibm-granite/granite-3.0-8b-instruct
```

Need to login with your `APIkey`:

```
podman login -u iamapikey -p $(cat $HOME/.ssh/ibmcloudAPI.key) us.icr.io/wxpe-cicd-internal
```

Create container with podman 

```
podman run --rm -d --name fmwork --privileged --pids-limit -1 --tz=local --user root --shm-size 16g -v /dev/vfio:/dev/vfio -v models/granite-3.0-8b:models/granite-3.0-8b -w /home/senuser -e AIU_SETUP_MULTI_AIU=1 -e FLEX_COMPUTE=SENTIENT -e FLEX_DEVICE=VFIO -e FLEX_OVERWRITE_NMB_FRAME=1 -e FLEX_UNLINK_DEVMEM=false us.icr.io/wxpe-cicd-internal/dd2/aiu-vllm-dev:latest
```
You need to get access to the image ```us.icr.io/wxpe-cicd-internal/dd2/aiu-vllm-dev:latest```

Log in to container and activate vllm env

```
podman exec -it fmwork bash -l
# Check if .senlib.json exists under /root, if not please follow the next step and if it exists please skip it
cp /home/senuser/.senlib.json /root/.senlib.json
source /opt/vllm/bin/activate
```

Clone repo:

```
git clone https://github.com/IBM/fmwork.git
```
By default, Host DMA is enabled. For P2P (R)DMA, use

```
unset FLEX_HDMA_MODE_FULL
export FLEX_RDMA_MODE_FULL=1 
```

By default, V1 engine is enabled. For V0 engine, set 

```
export VLLM_USE_V1=0
```

Run experiment in container:

```
./fmwork/infer/vllm/driver --model_path models/granite-3.0-8b --input_size 1024 --output_size 128 --batch_size 1 --tensor_parallel 1 --rep 1
```

This should produce blocks of outputs like:

```
update_lazyhandle() done (duration: 529.6827290058136s)
[SpyreWorker] ... warmup finished.
	warmup took 749.3301029205322s (for prompt length1024 and max output tokens 128)
[SpyreWorker] All warmups for 1 different prompt/decode/batchsize-shape combinations finished. Total warmup time 749.3324112892151s.
INFO 05-13 23:28:53 [llm_engine.py:438] Overriding num_gpu_blocks=512 with num_gpu_blocks_override=512
INFO 05-13 23:28:53 [executor_base.py:112] # cpu blocks: 512, # CPU blocks: 0
INFO 05-13 23:28:53 [executor_base.py:117] Maximum concurrency for 1152 tokens per request: 512.00x
INFO 05-13 23:28:53 [llm_engine.py:449] init engine (profile, create kv cache, warmup model) took 0.00 seconds

FMWORK SETUP 760.088849

--------------------------------------------------------------------------------
RUN 1024 / 128 / 1 / 1
--------------------------------------------------------------------------------

/home/senuser/./fmwork/infer/vllm/driver:146: DeprecationWarning: The keyword arguments {'prompt_token_ids'} are deprecated and will be removed in a future update. Please use the 'prompts' parameter instead.
  etim, med = run(input_size, output_size, batch_size)
[spyre_model_runner:execute_model] t_token: 792.11ms
[spyre_model_runner:execute_model] t_token: 137.86ms
[spyre_model_runner:execute_model] t_token: 137.85ms
[spyre_model_runner:execute_model] t_token: 137.32ms
[spyre_model_runner:execute_model] t_token: 139.71ms
[spyre_model_runner:execute_model] t_token: 142.41ms
[spyre_model_runner:execute_model] t_token: 143.29ms
[spyre_model_runner:execute_model] t_token: 144.89ms
[spyre_model_runner:execute_model] t_token: 145.77ms
[spyre_model_runner:execute_model] t_token: 145.02ms
[spyre_model_runner:execute_model] t_token: 144.86ms
[spyre_model_runner:execute_model] t_token: 145.59ms
[spyre_model_runner:execute_model] t_token: 145.02ms
[spyre_model_runner:execute_model] t_token: 145.81ms
[spyre_model_runner:execute_model] t_token: 148.56ms
[spyre_model_runner:execute_model] t_token: 151.14ms
[spyre_model_runner:execute_model] t_token: 153.61ms
[spyre_model_runner:execute_model] t_token: 152.48ms
[spyre_model_runner:execute_model] t_token: 152.01ms
[spyre_model_runner:execute_model] t_token: 150.51ms
[spyre_model_runner:execute_model] t_token: 150.57ms
[spyre_model_runner:execute_model] t_token: 149.28ms
[spyre_model_runner:execute_model] t_token: 148.82ms
[spyre_model_runner:execute_model] t_token: 149.81ms
[spyre_model_runner:execute_model] t_token: 149.42ms
[spyre_model_runner:execute_model] t_token: 149.10ms
[spyre_model_runner:execute_model] t_token: 156.54ms
[spyre_model_runner:execute_model] t_token: 149.45ms
[spyre_model_runner:execute_model] t_token: 150.10ms
[spyre_model_runner:execute_model] t_token: 150.47ms
[spyre_model_runner:execute_model] t_token: 152.09ms
[spyre_model_runner:execute_model] t_token: 148.37ms
[spyre_model_runner:execute_model] t_token: 148.76ms
[spyre_model_runner:execute_model] t_token: 148.62ms
[spyre_model_runner:execute_model] t_token: 148.03ms
[spyre_model_runner:execute_model] t_token: 148.92ms
[spyre_model_runner:execute_model] t_token: 148.62ms
[spyre_model_runner:execute_model] t_token: 148.15ms
[spyre_model_runner:execute_model] t_token: 151.31ms
[spyre_model_runner:execute_model] t_token: 149.06ms
[spyre_model_runner:execute_model] t_token: 147.93ms
[spyre_model_runner:execute_model] t_token: 148.52ms
[spyre_model_runner:execute_model] t_token: 148.30ms
[spyre_model_runner:execute_model] t_token: 148.14ms
[spyre_model_runner:execute_model] t_token: 150.19ms
[spyre_model_runner:execute_model] t_token: 148.72ms
[spyre_model_runner:execute_model] t_token: 148.13ms
[spyre_model_runner:execute_model] t_token: 149.29ms
[spyre_model_runner:execute_model] t_token: 148.73ms
[spyre_model_runner:execute_model] t_token: 149.29ms
[spyre_model_runner:execute_model] t_token: 148.96ms
[spyre_model_runner:execute_model] t_token: 148.93ms
[spyre_model_runner:execute_model] t_token: 148.93ms
[spyre_model_runner:execute_model] t_token: 153.12ms
[spyre_model_runner:execute_model] t_token: 148.84ms
[spyre_model_runner:execute_model] t_token: 147.99ms
[spyre_model_runner:execute_model] t_token: 148.63ms
[spyre_model_runner:execute_model] t_token: 149.23ms
[spyre_model_runner:execute_model] t_token: 148.97ms
[spyre_model_runner:execute_model] t_token: 151.51ms
[spyre_model_runner:execute_model] t_token: 149.59ms
[spyre_model_runner:execute_model] t_token: 147.96ms
[spyre_model_runner:execute_model] t_token: 148.63ms
[spyre_model_runner:execute_model] t_token: 148.94ms
[spyre_model_runner:execute_model] t_token: 151.30ms
[spyre_model_runner:execute_model] t_token: 152.72ms
[spyre_model_runner:execute_model] t_token: 149.03ms
[spyre_model_runner:execute_model] t_token: 148.38ms
[spyre_model_runner:execute_model] t_token: 149.16ms
[spyre_model_runner:execute_model] t_token: 149.01ms
[spyre_model_runner:execute_model] t_token: 148.95ms
[spyre_model_runner:execute_model] t_token: 149.31ms
[spyre_model_runner:execute_model] t_token: 150.07ms
[spyre_model_runner:execute_model] t_token: 148.17ms
[spyre_model_runner:execute_model] t_token: 149.66ms
[spyre_model_runner:execute_model] t_token: 149.51ms
[spyre_model_runner:execute_model] t_token: 150.20ms
[spyre_model_runner:execute_model] t_token: 149.09ms
[spyre_model_runner:execute_model] t_token: 151.60ms
[spyre_model_runner:execute_model] t_token: 148.30ms
[spyre_model_runner:execute_model] t_token: 149.49ms
[spyre_model_runner:execute_model] t_token: 149.95ms
[spyre_model_runner:execute_model] t_token: 148.95ms
[spyre_model_runner:execute_model] t_token: 150.80ms
[spyre_model_runner:execute_model] t_token: 149.07ms
[spyre_model_runner:execute_model] t_token: 148.22ms
[spyre_model_runner:execute_model] t_token: 148.75ms
[spyre_model_runner:execute_model] t_token: 148.90ms
[spyre_model_runner:execute_model] t_token: 150.24ms
[spyre_model_runner:execute_model] t_token: 148.72ms
[spyre_model_runner:execute_model] t_token: 148.93ms
[spyre_model_runner:execute_model] t_token: 148.08ms
[spyre_model_runner:execute_model] t_token: 148.63ms
[spyre_model_runner:execute_model] t_token: 148.75ms
[spyre_model_runner:execute_model] t_token: 147.49ms
[spyre_model_runner:execute_model] t_token: 148.83ms
[spyre_model_runner:execute_model] t_token: 146.75ms
[spyre_model_runner:execute_model] t_token: 145.45ms
[spyre_model_runner:execute_model] t_token: 145.45ms
[spyre_model_runner:execute_model] t_token: 145.51ms
[spyre_model_runner:execute_model] t_token: 145.56ms
[spyre_model_runner:execute_model] t_token: 145.06ms
[spyre_model_runner:execute_model] t_token: 146.14ms
[spyre_model_runner:execute_model] t_token: 146.13ms
[spyre_model_runner:execute_model] t_token: 145.20ms
[spyre_model_runner:execute_model] t_token: 147.46ms
[spyre_model_runner:execute_model] t_token: 147.36ms
[spyre_model_runner:execute_model] t_token: 145.84ms
[spyre_model_runner:execute_model] t_token: 145.44ms
[spyre_model_runner:execute_model] t_token: 145.73ms
[spyre_model_runner:execute_model] t_token: 144.59ms
[spyre_model_runner:execute_model] t_token: 145.55ms
[spyre_model_runner:execute_model] t_token: 146.53ms
[spyre_model_runner:execute_model] t_token: 145.65ms
[spyre_model_runner:execute_model] t_token: 146.38ms
[spyre_model_runner:execute_model] t_token: 145.81ms
[spyre_model_runner:execute_model] t_token: 145.27ms
[spyre_model_runner:execute_model] t_token: 145.18ms
[spyre_model_runner:execute_model] t_token: 146.75ms
[spyre_model_runner:execute_model] t_token: 146.39ms
[spyre_model_runner:execute_model] t_token: 147.40ms
[spyre_model_runner:execute_model] t_token: 147.89ms
[spyre_model_runner:execute_model] t_token: 146.60ms
[spyre_model_runner:execute_model] t_token: 149.01ms
[spyre_model_runner:execute_model] t_token: 147.28ms
[spyre_model_runner:execute_model] t_token: 145.76ms
[spyre_model_runner:execute_model] t_token: 147.10ms
[spyre_model_runner:execute_model] t_token: 147.02ms
```

Parse results  

```
./parse.sh <result directory> | xargs -n 9
```
Output: < model mode iis oos bbs tp warmup(s) setup ttft(ms) itl(ms) >

Exit and stop container

```
exit
podman stop fmwork
```
