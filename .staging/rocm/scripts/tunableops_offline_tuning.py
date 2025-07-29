import torch.cuda.tunable as tunable
import os

os.putenv('PYTORCH_TUNABLEOP_ENABLED', '1')
os.putenv('PYTORCH_TUNABLEOP_TUNING', '1')
os.putenv('PYTORCH_TUNABLEOP_RECORD_UNTUNED', '0')

if __name__ == "__main__":
    num_gpus = 8 # number of GPUs that will be used during the tuning process
    tunable.mgpu_tune_gemm_in_file("tunableop_untuned0.csv", num_gpus)
