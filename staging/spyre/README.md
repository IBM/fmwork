# fmwork

FM Benchmarking Framework

## Quick start

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
