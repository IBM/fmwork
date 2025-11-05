# infer/transformers

Inference implementation using the standard [Hugging Face Transformers](https://huggingface.co/docs/transformers/index) library.

The driver is a script designed for simple, repetitive performance benchmarking of Large Language Models (LLM).

## Usage examples

### Basic CUDA Inference

This example runs a specified model (`facebook/opt-125m`) on a CUDA device, using the required input/output lengths, batch size, and the number of repetitions (`reps`) for timing.

```bash
./driver \
    --model facebook/opt-125m \
    --input_size 1024 \
    --output_size 128 \
    --batch_size 4 \
    --reps 100 \
    --device cuda
```
