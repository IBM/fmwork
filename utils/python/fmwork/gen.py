def input_generator(model_path, input_size, batch_size, return_tensors):

    tokenizer = transformers.AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    vocab = list(range(0, tokenizer.vocab_size))
    for i in tokenizer.all_special_ids:
        if i in vocab:
            vocab.remove(i)

    tokens = [ [] for _ in range(batch_size) ]
    for b in range(batch_size):
        for i in range(input_size):
            tokens[b].append(random.choice(vocab))

    if return_tensors == 'np': return tokens

    input_batch = BatchEncoding({
        'input_ids' : torch.tensor(tokens),
        'attention_mask' : torch.ones(batch_size, input_size),
    })

    return input_batch

