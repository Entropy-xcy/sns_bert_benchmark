import torch as th
import time
from math import ceil
import torch

POWER_CHK_POINT = "saved_models/mobilebert_power_final.pt"
TIMING_CHK_POINT = "saved_models/mobilebert_timing_final.pt"
compiled_timing = 'ff2ff_timing_bert.pt'
compiled_power = 'ff2ff_power_bert.pt'
compiled_timing_gpu = 'ff2ff_timing_bert_gpu.pt'
compiled_power_gpu = 'ff2ff_power_bert_gpu.pt'
compiled_model = compiled_power_gpu
batchsize = 128 
num_paths = 10 * 20480

if __name__ == "__main__":
    model = th.jit.load(compiled_model, map_location=torch.device('cuda:0'))
    dummy_inputs = th.ones(batchsize, 512, dtype=int, device="cuda:0")

    # model = model.to("cuda:0")
    # dummy_inputs = dummy_inputs.to("cuda:0")
    
    out = model(dummy_inputs)

    start = time.time()
    for i in range(ceil(num_paths / batchsize)):
        out = model(dummy_inputs)
    end = time.time()

    print("Time Elapsed:", end-start)

    
