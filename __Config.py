import torch
import os
import random
import numpy as np
import socket
import platform
import torch.distributed as dst

# ==========================================
# 1. Configuration & Hyperparameters
# ==========================================

device   = "cuda" if torch.cuda.is_available() else "cpu"
universe = torch.cuda.device_count() if torch.cuda.is_available() else 1
cores    = 2

#Path
RAW_DATA = "Data/LATENTS_MOSDAC/mosdac_tir1"

# REAL latent size = [C, H, W] where C is channels per-latent (e.g. 4)
img_x   = 64
img_y   = 64
img_z   = 4      
seq_len = 4      

# Training Terms
steps_all     = 25000     
save_steps    = 1000
batch_size    = 16 * universe
lr            = 5e-5     
lr_class      = "CosAWR"
weight_decay  = 1e-4
amp_on        = True
ema_on        = True
ema_decay     = 0.9999

# Diffusion Terms
diff_timesteps     = 1000  
sampling_timesteps = 1000

chan     = 160
chan_lvl = [1, 2, 3, 4]
num_res_stg = 2
grps     = 32
attn_lvl = [False, True, False, True]
attn_heads = 16
free = 0

# ==========================================
# 2. Aux Functions
# ==========================================

def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

def format_time(seconds):
    hrs = int(seconds // 3600)
    mins = int((seconds % 3600) // 60)
    return f"{hrs:02d}.{mins:02d}"

def find_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return str(s.getsockname()[1])

def setup(rank, universe):
    control = "nccl" if platform.system() == "Linux" else "gloo"
    dst.init_process_group(control, rank=rank, world_size=universe)
    torch.cuda.set_device(rank)

def cleanup():
    if dst.is_initialized():
        dst.destroy_process_group()