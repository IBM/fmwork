# fmwork

FM Benchmarking Framework

## Quick start

Get a model (e.g., https://huggingface.co/ibm-granite/granite-3.0-8b-instruct):

```
pip install huggingface-hub
huggingface-cli download --cache-dir ./ --local-dir-use-symlinks False --revision main --local-dir models/granite-3.0-8b ibm-granite/granite-3.0-8b-instruct
```

Create container with podman

```
podman run --rm -d --name fmwork --privileged --pids-limit -1 --tz=local --user root --shm-size 16g -v /dev/vfio:/dev/vfio -v models/granite-3.0-8b:models/granite-3.0-8b -w /home/senuser -e AIU_SETUP_MULTI_AIU=1 -e FLEX_COMPUTE=SENTIENT -e FLEX_DEVICE=VFIO -e FLEX_OVERWRITE_NMB_FRAME=1 -e FLEX_UNLINK_DEVMEM=false docker-na-public.artifactory.swg-devops.com/wcp-ai-foundation-team-docker-virtual/aiu-vllm-dev
```
Check https://github.ibm.com/ai-foundation/aiu-inference-dev?tab=readme-ov-file#artifactory-access to get access to the image

Log in to container and activate vllm env


```
podman exec -it fmwork bash -l
source /opt/vllm/bin/activate
```

Clone repo and run experiment in container:

```
git clone -b spyre https://github.com/shwetasalaria/fmwork.git
./fmwork/infer/vllm/driver --model_path models/granite-3.0-8b --input_size 512 --output_size 8 --batch_size 1 --tensor_parallel 1
```
By default, Host DMA is enabled. For P2P (R)DMA, use

```
unset FLEX_HDMA_MODE_FULL
export FLEX_RDMA_MODE_FULL=1 
```

This should produce blocks of outputs like:

```
--------------------------------------------------------------------------------
RUN 64 / 3 / 1 / 1
--------------------------------------------------------------------------------

FMWORK REP   1 /   3 : 1741737889.267835572 1741737889.903656112 0.636 211.9 4.7
FMWORK REP   2 /   3 : 1741737889.903777288 1741737890.545321112 0.642 213.8 4.7
FMWORK REP   3 /   3 : 1741737890.545434292 1741737891.203066833 0.658 219.2 4.6


FMWORK RES 20250312-000451.204555 64 3 1 1 0.649588 216.5 4.6

Input size                = 64
Output size               = 3
Batch size                = 1
Median iteration time (s) = 0.649588
Inter-token latency (ms)  = 216.5
Throughput (tok/s)        = 4.6

```

- `FMWORK REP` lines contain stats per experiment repetition (3 repetitions by default):
    - Number of repetition
    - Total repetitions to run
    - Timestamp of rep start
    - Timestamp of rep end
    - Duration of rep (seconds)
    - Inter-token latency for rep (milliseconds per token)
    - Throughput for rep (tokens per second)

- `FMWORK RES` line contains a summary of the experiment:
    - Experiment timestamp
    - Input size
    - Output size
    - Batch size
    - Tensor parallelism size
    - Median iteration duration (seconds)
    - Inter-token latency (milliseconds per token)
    - Throughput (tokens per second)

If saved to a file, all `RES` lines can be easily grep-ed for further analysis.


```
grep -R "FMWORK RES" outputs/ | tr / ' ' | column -t
```
Run with 2 spyre cards:

```
--------------------------------------------------------------------------------
RUN 64 / 3 / 1 / 2
--------------------------------------------------------------------------------

FMWORK REP   1 /   3 : 1741738282.836189964 1741738283.242006582 0.406 135.3 7.4
FMWORK REP   2 /   3 : 1741738283.242051507 1741738283.634623773 0.393 130.9 7.6
FMWORK REP   3 /   3 : 1741738283.634658861 1741738284.031624817 0.397 132.3 7.6

FMWORK RES 20250312-001124.032077 64 3 1 2 0.394769 131.6 7.6

Input size                = 64
Output size               = 3
Batch size                = 1
Median iteration time (s) = 0.394769
Inter-token latency (ms)  = 131.6
Throughput (tok/s)        = 7.6

```

Stop container

```
podman stop fmwork
```
