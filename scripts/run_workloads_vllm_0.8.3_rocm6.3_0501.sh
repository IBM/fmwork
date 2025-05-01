#!/usr/bin/bash

#amd-meta-llama-3.3-70b-fp8-kv
VLLM_ROCM_FP8_PADDING=1 VLLM_FP8_ACT_PADDING=1 VLLM_FP8_WEIGHT_PADDING=1 VLLM_FP8_REDUCE_CONV=1 VLLM_ROCM_USE_AITER=1 VLLM_ROCM_USE_AITER_PAGED_ATTN=1 VLLM_USE_TRITON_FLASH_ATTN=0 ../fmwork/infer/vllm/driver --model_path /data/amd-Llama-3.3-70b-Instruct-FP8-kv --input_size 1024 --output_size 1024 --batch_size 76 --tensor_parallel 1 --dtype bfloat16 --quantization fp8 --num_scheduler_steps 32 --enable_prefix_caching --kv_cache_dtype fp8 2>&1 | tee amd-meta-llama-3.3-70b-fp8-tp1-bs76-aiter-attn-kvcache-fp8.txt

#granite-3.1-8b
VLLM_ROCM_USE_AITER=1 VLLM_ROCM_USE_AITER_PAGED_ATTN=1 VLLM_USE_TRITON_FLASH_ATTN=0 ../fmwork/infer/vllm/driver --model_path /data/granite-3.1-8b --input_size 1024 --output_size 1024 --batch_size 160 --tensor_parallel 1 --dtype bfloat16 --num_scheduler_steps 32 --enable_prefix_caching --kv_cache_dtype fp8 2>&1 | tee granite-3.1-8b-bf16-tp1-bs160-aiter-attn-kvcache-fp8.txt

#amd-Meta-Llama-3.1-405B-Instruct-FP8-KV
VLLM_ROCM_USE_AITER=1 VLLM_ROCM_USE_AITER_PAGED_ATTN=1 VLLM_ROCM_FP8_PADDING=1 NCCL_MIN_NCHANNELS=112 TENSILE_SOLUTION_SELECTION_METHOD=2  VLLM_USE_TRITON_FLASH_ATTN=0 ./infer/vllm/driver --model_path /data/amd-Llama-3.1-405B-Instruct-FP8-KV/ --input_size 1024 --output_size 1024 --batch_size 128 --tensor_parallel 8 --dtype bfloat16 --quantization fp8  --kv_cache_dtype fp8 --num_scheduler_steps 32 --enable_prefix_caching 2>&1 | tee amd-Meta-Llama-3.1-405B-fp8-tp8-bs128-kvcache-fp8-aiter-pg-without-max-seq.txt

#Mistral-Large-Instruct-2407
VLLM_ROCM_USE_AITER_PAGED_ATTN=1 VLLM_ROCM_USE_AITER=1 VLLM_USE_TRITON_FLASH_ATTN=0 ../fmwork/infer/vllm/driver --model_path /data/Mistral-Large-Instruct-2407 --input_size 1024 --output_size 1024 --batch_size 76 --tensor_parallel 4 --dtype bfloat16 --num_scheduler_steps 64 --enable_prefix_caching --kv_cache_dtype fp8 --gpu_memory_utilization 0.9 --max_seq_len_to_capture 131072  --max_num_batched_tokens 131072 2>&1 | tee mistral-large-2407-bf16-tp4-bs72-aiter-attn-kvcache-fp8-maxseqlen.txt

#llama3.2-90b-vision-instruct # use vision branch
VLLM_ROCM_USE_AITER=1 VLLM_USE_TRITON_FLASH_ATTN=0  ../fmwork_vision/driver --model_path /data/llama3.2-90b-vision-instruct --input_size 1024 --output_size 1024  --batch_size 48  --tensor_parallel_size 4  --image_width 256 --image_height 256 --kv_cache_dtype fp8  2>&1 | tee Llama-3.2-90B-Vision-tp4-bs48-aiter-kvcachefp8.txt

#Llama-3.1-70B TP4
VLLM_ROCM_USE_AITER_PAGED_ATTN=1 VLLM_ROCM_USE_AITER=1 VLLM_USE_TRITON_FLASH_ATTN=0 ../fmwork/infer/vllm/driver  --model_path /data/Llama-3.1-70B-Instruct --input_size 1024  --output_size 1024  --batch_size 144  --tensor_parallel 4  --dtype bfloat16  --num_scheduler_steps 64  --enable_prefix_caching  --kv_cache_dtype fp8  --gpu_memory_utilization 0.9  --max_seq_len_to_capture 131072  --max_num_batched_tokens 131072 2>&1 | tee Llama-3.1-70B-bf16-tp4-bs144-steps64-aiter-attn-kvcache-fp8-seqlen131072.txt

#granite-34b-code-instruct-8k
VLLM_ROCM_USE_AITER=1 VLLM_ROCM_USE_AITER_PAGED_ATTN=1 VLLM_USE_TRITON_FLASH_ATTN=0 ../fmwork/infer/vllm/driver --model_path /data/granite-34b-code-instruct-8k --input_size 1024 --output_size 1024 --tensor_parallel 1 --batch_size 96  --dtype bfloat16 --num_scheduler_steps 32 --enable_prefix_caching  --kv_cache_dtype fp8 2>&1 | tee granite-34b-code-bf16-tp1-bs96-aiter-attn-kvcache-fp8.txt

#Mixtral 8x7B
VLLM_ROCM_USE_AITER=1 VLLM_ROCM_USE_AITER_PAGED_ATTN=1 VLLM_USE_TRITON_FLASH_ATTN=0 ../fmwork/infer/vllm/driver --model_path /data/Mixtral-8x7B --input_size 1024 --output_size 1024 --batch_size 152 --tensor_parallel 1 --dtype bfloat16 --num_scheduler_steps 32 --kv_cache_dtype fp8  2>&1 | tee mixtral-8x7b-bf16-tp1-bs152-aiter-attn-kvcache-fp8.txt

#CodeLlama-34b-Instruct-hf
VLLM_ROCM_USE_AITER=1 VLLM_ROCM_USE_AITER_PAGED_ATTN=1 VLLM_USE_TRITON_FLASH_ATTN=0  ../fmwork/infer/vllm/driver --model_path /data/CodeLlama-34b-Instruct-hf --input_size 1024 --output_size 1024  --batch_size 96  --tensor_parallel 1 --dtype bfloat16  --num_scheduler_steps 32  --enable_prefix_caching  --kv_cache_dtype fp8 2>&1 | tee CodeLlama-34b-bf16-tp1-bs96-aiter-attn-kvcache-fp8.txt

#granite-3b-code-instruct-128k
VLLM_ROCM_USE_AITER=1 VLLM_USE_TRITON_FLASH_ATTN=0 ../fmwork/infer/vllm/driver --model_path /data/granite-3b-code-instruct-128k/ --input_size 1024 --output_size 1024 --batch_size 56 --tensor_parallel 1 --dtype bfloat16 --num_scheduler_steps 32 --enable_prefix_caching 2>&1 | tee granite-3b-code-bf16-tp1-bs56-aiter.txt

#meta-Llama-3.1-8B-Instruct
VLLM_ROCM_USE_AITER=1 VLLM_ROCM_USE_AITER_PAGED_ATTN=1 VLLM_USE_TRITON_FLASH_ATTN=0 ../fmwork/infer/vllm/driver --model_path /data/Llama-3.1-8B-Instruct --input_size 1024 --output_size 1024 --batch_size 192 --tensor_parallel 1 --dtype bfloat16 --num_scheduler_steps 32 --enable_prefix_caching --kv_cache_dtype fp8 2>&1 | tee Llama-3.1-8B-bf16-tp1-bs192-aiter-attn-kvcache-fp8.txt

#granite-20b-code-instruct-8k
VLLM_ROCM_USE_AITER=1 VLLM_ROCM_USE_AITER_PAGED_ATTN=1 VLLM_USE_TRITON_FLASH_ATTN=0 ../fmwork/infer/vllm/driver --model_path /data/granite-20b-code-instruct-8k --input_size 1024 --output_size 1024 --tensor_parallel 1 --batch_size 80 --dtype bfloat16 --num_scheduler_steps 32 --enable_prefix_caching  --kv_cache_dtype fp8 2>&1 | tee granite-20b-code-bf16-tp1-bs80-aiter-attn-kvcache-fp8.txt

#granite-8b-code-instruct-128k
VLLM_ROCM_USE_AITER=1 VLLM_ROCM_USE_AITER_PAGED_ATTN=1 VLLM_USE_TRITON_FLASH_ATTN=0 ../fmwork/infer/vllm/driver --model_path /data/granite-8b-code-instruct-128k --input_size 1024 --output_size 1024 --tensor_parallel 1 --batch_size 176 --dtype bfloat16 --num_scheduler_steps 32 --enable_prefix_caching  --kv_cache_dtype fp8 2>&1 | tee granite-8b-code-bf16-tp1-bs176-aiter-attn-kvcache-fp8.txt
