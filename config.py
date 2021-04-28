
import uuid

import torch

CUDA_NUM = 1  # the GPU device you want to use
DATA_DIR = './DEM_data'
RESULT_DIR = './results'

DEVICE = torch.device(f'cuda:{CUDA_NUM}') if torch.cuda.is_available() \
    else torch.device('cpu')
EXPERIMENT_ID = uuid.uuid4().hex
