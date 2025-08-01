import random
import torch

from transformers import AutoTokenizer
from transformers.tokenization_utils_base import BatchEncoding

class RandomGenerator:

    def __init__(self, model_path):

        tokenizer = AutoTokenizer.from_pretrained(model_path)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"

        self.vocab = list(range(0, tokenizer.vocab_size))
        for i in tokenizer.all_special_ids:
            if i in self.vocab:
                self.vocab.remove(i)

    def prompt(self, input_size, batch_size, return_tensors):

        tokens = [ [] for _ in range(batch_size) ]
        for b in range(batch_size):
            for i in range(input_size):
                tokens[b].append(random.choice(self.vocab))

        if return_tensors == 'np': return tokens

        input_batch = BatchEncoding({
            'input_ids' : torch.tensor(tokens),
            'attention_mask' : torch.ones(batch_size, input_size),
        })

        return input_batch

