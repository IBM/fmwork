# fmwork utilities

`lakehouse_insert.py` reads fmwork data from the CSV file outputted by `process.py` and inserts data into IBM Lakehouse tables

See [main README](https://github.com/IBM/fmwork/blob/main/README.md) for instructions to run `process.py`



## Example
```bash
python lakehouse_insert.py vllm_cuda ~/md0/fmwork-results/fmwork_data.csv
```

Note: the first argument is the table name that you want to insert your data into - it must be one of the following:  `vllm_cuda`, `vllm_gaudi`, `vllm_rocm`, `vllm_spyre`, `trtllm_cuda`, `transformers_cuda`

The second argument is the path to the CSV file that is outputted by `process.py`

`lakehouse_insert.py` will read the CSV file and insert the data into the table name provided and also consolidated inference engine and hardware tables. For example, if the argument provided is `vllm_cuda`, the data will be written to the `vllm_cuda` table, the `vllm` table, and the `cuda` table in IBM Lakehouse.



