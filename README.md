# fmwork

FM Benchmarking Framework — Vision

Please follow the instructions from `main` branch to install FMwork.

Some examples of running this benchmark:

```
./driver --model $css22/nmg/models/llama3.2-11b/instruct --input_size 20                                   --output_size 64 --batch_size 2 --tensor_parallel_size 1 --image_url "https://huggingface.co/datasets/huggingface/documentation-images/resolve/0052a70beed5bf71b92610a43a52df6d286cd5f3/diffusers/rabbit.jpg" | tee OUT.11.1
./driver --model $css22/nmg/models/llama3.2-11b/instruct --input_size 20                                   --output_size 64 --batch_size 2 --tensor_parallel_size 1 --image_width 1024 --image_height 1024                                                                                                               | tee OUT.11.2
./driver --model $css22/nmg/models/llama3.2-11b/instruct --input_text "What is the content of this image?" --output_size 64 --batch_size 2 --tensor_parallel_size 1 --image_url "https://huggingface.co/datasets/huggingface/documentation-images/resolve/0052a70beed5bf71b92610a43a52df6d286cd5f3/diffusers/rabbit.jpg" | tee OUT.11.3
./driver --model $css22/nmg/models/llama3.2-11b/instruct --input_text "What is the content of this image?" --output_size 64 --batch_size 2 --tensor_parallel_size 1 --image_width 1024 --image_height 1024                                                                                                               | tee OUT.11.4

./driver --model $css22/nmg/models/llama3.2-90b/instruct --input_size 20                                   --output_size 64 --batch_size 2 --tensor_parallel_size 4 --image_url "https://huggingface.co/datasets/huggingface/documentation-images/resolve/0052a70beed5bf71b92610a43a52df6d286cd5f3/diffusers/rabbit.jpg" | tee OUT.90.1
./driver --model $css22/nmg/models/llama3.2-90b/instruct --input_size 20                                   --output_size 64 --batch_size 2 --tensor_parallel_size 4 --image_width 1024 --image_height 1024                                                                                                               | tee OUT.90.2
./driver --model $css22/nmg/models/llama3.2-90b/instruct --input_text "What is the content of this image?" --output_size 64 --batch_size 2 --tensor_parallel_size 4 --image_url "https://huggingface.co/datasets/huggingface/documentation-images/resolve/0052a70beed5bf71b92610a43a52df6d286cd5f3/diffusers/rabbit.jpg" | tee OUT.90.3
./driver --model $css22/nmg/models/llama3.2-90b/instruct --input_text "What is the content of this image?" --output_size 64 --batch_size 2 --tensor_parallel_size 4 --image_width 1024 --image_height 1024                                                                                                               | tee OUT.90.4
```

For input prompt the script takes either `--input_text`
(which can be used to pass an actual prompt)
or `--input_size` for tokenizer-based generation.
`--output_size` determines the number of tokens to generate.
For input image the script takes either `--image_url`
or `--image_width` & `--image_height` to generate one.

To run sweeps (example):

```
./run --base_output_dir outputs/001 --model_path /path/to/llama3.2-11b/instruct --input_sizes 1024 --output_sizes 1024 --batch_sizes 1,2,4,8,12,16,20,24,28,32,40,48,56,64,80,96,112,128,160,192,224,256,320,384,448,512 --tensor_parallel_sizes 1,2,4,8 --device_sets 0:1:2:3:4:5:6:7/0,1:2,3:4,5:6,7/0,1,2,3:4,5,6,7/0,1,2,3,4,5,6,7
./run --base_output_dir outputs/002 --model_path /path/to/llama3.2-90b/instruct --input_sizes 1024 --output_sizes 1024 --batch_sizes 1,2,4,8,12,16,20,24,28,32,40,48,56,64,80,96,112,128,160,192,224,256,320,384,448,512 --tensor_parallel_sizes     4,8 --device_sets 0:1:2:3:4:5:6:7/0,1:2,3:4,5:6,7/0,1,2,3:4,5,6,7/0,1,2,3,4,5,6,7
```

