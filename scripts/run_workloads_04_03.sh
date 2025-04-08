#!/usr/bin/bash

#amd-meta-llama-3.3-70b-fp8-kv
VLLM_FP8_PADDING=1 VLLM_FP8_ACT_PADDING=1 VLLM_FP8_WEIGHT_PADDING=1 VLLM_FP8_REDUCE_CONV=1 VLLM_USE_AITER=1 VLLM_USE_AITER_PAGED_ATTN=1 VLLM_USE_TRITON_FLASH_ATTN=0 ./infer/vllm/driver --model_path /data/amd-meta-llama-3.3-70b-fp8-kv/ --input_size 1024 --output_size 1024 --batch_size 72 --tensor_parallel 1 --dtype bfloat16 --quantization fp8 --enable_prefix_caching --num_scheduler_steps 32 --kv_cache_dtype fp8 2>&1 | tee amd-meta-llama-3.3-70b-fp8-tp1-bs72-aiter-attn-kvcache-fp8.txt

VLLM_USE_AITER=1 VLLM_USE_AITER_PAGED_ATTN=1 VLLM_USE_TRITON_FLASH_ATTN=0 ./infer/vllm/driver --model_path /data/granite-3.1-8b-instruct --num_scheduler_steps 32 --input_size 1024 --output_size 1024 --batch_size 160 --tensor_parallel 1 --dtype bfloat16 --enable_prefix_caching --kv_cache_dtype fp8 2>&1 | tee granite-3.1-8b-bf16-tp1-bs160-aiter-attn-kvcache-fp8.txt

#amd-Meta-Llama-3.1-405B-Instruct-FP8-KV
VLLM_FP8_PADDING=1 VLLM_FP8_ACT_PADDING=1 VLLM_FP8_WEIGHT_PADDING=1 VLLM_FP8_REDUCE_CONV=1 NCCL_MIN_NCHANNELS=112 TENSILE_SOLUTION_SELECTION_METHOD=2  VLLM_USE_TRITON_FLASH_ATTN=0 ./infer/vllm/driver --model_path /data/amd-Meta-Llama-3.1-405B-Instruct-FP8-KV/ --input_size 1024 --output_size 1024 --batch_size 120 --tensor_parallel 8 --dtype bfloat16 --quantization fp8  --kv_cache_dtype fp8 --num_scheduler_steps 32 --enable_prefix_caching --max_num_seqs 300 2>&1 | tee amd-Meta-Llama-3.1-405B-fp8-tp8-bs120-kvcache-fp8.txt

VLLM_USE_AITER=1 VLLM_USE_AITER_PAGED_ATTN=1 VLLM_USE_TRITON_FLASH_ATTN=0 ./infer/vllm/driver --model_path /data/mistral-large-2407/ --input_size 1024 --output_size 1024  --batch_size 64 --tensor_parallel 4 --dtype bfloat16  --num_scheduler_steps 32 --kv_cache_dtype fp8 2>&1 | tee mistral-large-2407-bf16-tp4-bs64-aiter-attn-kvcache-fp8.txt

VLLM_USE_TRITON_FLASH_ATTN=0  ./infer/vllm/driver --model_path /data/Llama-3.2-90B-Vision-Instruct --input_size 1024 --output_size 1024  --batch_size 40  --tensor_parallel_size 4  --image_width 256 --image_height 256  2>&1 | tee Llama-3.2-90B-Vision-tp4-bs40.txt

VLLM_USE_TRITON_FLASH_ATTN=0 ./infer/vllm/driver --model_path /data/Llama-3.1-70B-Instruct --input_size 1024 --output_size 1024 --batch_size 96 --tensor_parallel 4 --dtype bfloat16 --num_scheduler_steps 32 --enable_prefix_caching --distributed_executor_backend ray 2>&1 | tee Llama-3.1-70B-bf16-tp4-bs96.txt

#Rocm 6.4
VLLM_USE_AITER=1 VLLM_USE_AITER_PAGED_ATTN=1 VLLM_USE_TRITON_FLASH_ATTN=0 ./infer/vllm/driver --model_path /data/granite-34b-code-instruct-8k/ --input_size 1024 --output_size 1024 --tensor_parallel 1 --batch_size 88 --dtype bfloat16 --num_scheduler_steps 32 --enable_prefix_caching  --kv_cache_dtype fp8 2>&1 | tee granite-34b-code-bf16-tp1-bs88-aiter-attn-kvcache-fp8.txt

VLLM_USE_AITER=1 VLLM_USE_TRITON_FLASH_ATTN=0 ./infer/vllm/driver --model_path /data/mixtral-8x7b/ --input_size 1024 --output_size 1024 --batch_size 96 --tensor_parallel 1 --dtype bfloat16 --num_scheduler_steps 32  2>&1 | tee  mixtral-8x7b-bf16-tp1-bs96-aiter.txt

VLLM_USE_AITER=1 VLLM_USE_AITER_PAGED_ATTN=1 VLLM_USE_TRITON_FLASH_ATTN=0  ./infer/vllm/driver --model_path /data/CodeLlama-34b-Instruct-hf --input_size 1024 --output_size 1024  --batch_size 96 --tensor_parallel 1 --dtype bfloat16  --num_scheduler_steps 32  --kv_cache_dtype fp8 2>&1 | tee CodeLlama-34b-bf16-tp1-bs96-aiter-attn-kvcache-fp8.txt

VLLM_USE_AITER=1 VLLM_USE_AITER_PAGED_ATTN=1 VLLM_USE_TRITON_FLASH_ATTN=0 ./infer/vllm/driver --model_path /data/granite-3b-code-instruct-128k/ --input_size 1024 --output_size 1024 --batch_size 56 --tensor_parallel 1 --dtype bfloat16 --num_scheduler_steps 32 --enable_prefix_caching --kv_cache_dtype fp8 2>&1 | tee granite-3b-code-bf16-tp1-bs56-aiter-attn-kvcache-fp8.txt

VLLM_USE_AITER=1 VLLM_USE_AITER_PAGED_ATTN=1 VLLM_USE_TRITON_FLASH_ATTN=0 ./infer/vllm/driver --model_path /data/Llama-3.1-8B-Instruct --input_size 1024 --output_size 1024 --batch_size 192 --tensor_parallel 1 --dtype bfloat16 --kv_cache_dtype fp8 --num_scheduler_steps 32 --enable_prefix_caching --gpu_memory_utilization 0.95 --distributed_executor_backend ray 2>&1 | tee Llama-3.1-8B-bf16-tp1-bs192-aiter-attn-kvcache-fp8.txt

#Rocm 6.4
VLLM_USE_AITER=1 VLLM_USE_AITER_PAGED_ATTN=1 VLLM_USE_TRITON_FLASH_ATTN=0 ./infer/vllm/driver --model_path /data/granite-20b-code-instruct-8k/ --input_size 1024 --output_size 1024 --tensor_parallel 1 --batch_size 64 --dtype bfloat16 --num_scheduler_steps 32 --enable_prefix_caching  --kv_cache_dtype fp8 2>&1 | tee granite-20b-code-bf16-tp1-bs64-aiter-attn-kvcache-fp8.txt

VLLM_USE_AITER=1 VLLM_USE_AITER_PAGED_ATTN=1 VLLM_USE_TRITON_FLASH_ATTN=0 ./infer/vllm/driver --model_path /data/granite-8b-code-instruct-128k/ --input_size 1024 --output_size 1024 --tensor_parallel 1 --batch_size 176 --dtype bfloat16 --num_scheduler_steps 32 --enable_prefix_caching  --kv_cache_dtype fp8 2>&1 | tee granite-8b-code-bf16-tp1-bs128-aiter-attn-kvcache-fp8.txt
