import torch
import os
import random
from typing import List, Optional, Tuple
from torch.utils.data import Dataset

# --- CRITICAL: Import configuration variables from Config.py ---
try:
    from Config import img_z, seq_len
except ImportError:
    # Fallback to prevent crash if Config.py isn't found immediately
    print("Warning: Config.py not found. Using default fallbacks.")
    img_z = 4
    seq_len = 4

# ==========================================
# 3. Data Processing (Latent tensors)
# ==========================================

def compute_channel_mean_std(
    root: str,
    files: Optional[List[str]] = None,
    max_items: Optional[int] = 1000,
    skip_bad: bool = True,
    verbose: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Computes statistics for normalization.
    """
    if files is None:
        files = sorted([os.path.join(root, f) for f in os.listdir(root) if f.endswith('.pt')])

    if max_items is not None:
        files = files[:max_items]

    if len(files) == 0:
        # Return dummy values using img_z from Config.py
        return torch.zeros(img_z, 1, 1), torch.ones(img_z, 1, 1)

    # Load first valid to infer channel count
    first = None
    for p in files:
        try:
            cand = torch.load(p, map_location='cpu').float()
            if cand.dim() == 4 and cand.size(0) == 1: cand = cand.squeeze(0)
            first = cand
            break
        except Exception:
            continue

    if first is None:
         return torch.zeros(img_z, 1, 1), torch.ones(img_z, 1, 1)

    C = first.size(0)
    cnt_px = 0
    channel_sum = torch.zeros(C, dtype=torch.float64)
    channel_sumsq = torch.zeros(C, dtype=torch.float64)

    processed = 0
    for i, p in enumerate(files):
        try:
            t = torch.load(p, map_location='cpu').float()
            if t.dim() == 4 and t.size(0) == 1: t = t.squeeze(0)
            
            channel_sum += t.sum(dim=(1,2)).double()
            channel_sumsq += (t**2).sum(dim=(1,2)).double()
            cnt_px += (t.shape[1] * t.shape[2])
            processed += 1
        except Exception:
            continue

    if cnt_px == 0:
        return torch.zeros(C, 1, 1), torch.ones(C, 1, 1)

    mean = (channel_sum / cnt_px).float()
    var = (channel_sumsq / cnt_px).double() - (mean.double() ** 2)
    std = torch.sqrt(torch.clamp(var, min=1e-12)).float()

    return mean.view(C,1,1), std.view(C,1,1)


class SequentialDataset(Dataset):
    """
    Dataset for sequential latents.
    Returns:
        target: Tensor [C, H, W]  (Frame t)
        context: Tensor [seq_len, C, H, W] (Frames t-seq_len ... t-1)
    """

    def __init__(
        self,
        root: str,
        seq_len: int = seq_len, # Uses default from Config.py
        augment: bool = False,
        load_in_memory: bool = False,
        normalize_stats: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        per_sample_normalize: bool = False,
        skip_bad_files: bool = True,
        verbose: bool = True
    ):

        super().__init__()
        self.seq_len = seq_len
        self.augment = augment
        self.per_sample_normalize = per_sample_normalize
        self.load_in_memory = load_in_memory
        self.skip_bad_files = skip_bad_files
        self.verbose = verbose
        
        if self.per_sample_normalize:
             print("WARNING: per_sample_normalize is ON. This will re-scale latents that are already Tanh-constrained.")

        print(f"\nLoading latent files from: {root}")
        self.files = sorted([os.path.join(root, f) for f in os.listdir(root) if f.endswith('.pt')])
        print(f"Total Latents Found: {len(self.files)}")

        if len(self.files) <= seq_len:
            raise ValueError("❌ Not enough latent files for the given sequence length!")

        self.cache: Optional[List[torch.Tensor]] = None
        if self.load_in_memory:
            print("Loading latents into memory...")
            self.cache = []
            for i, p in enumerate(self.files):
                try:
                    self.cache.append(self._load_tensor(p))
                except Exception:
                    self.cache.append(None) # Mark bad files as None

        self._eps = 1e-8

    def __len__(self):
        return len(self.files) - self.seq_len

    def _load_tensor(self, path: str) -> torch.Tensor:
        try:
            t = torch.load(path, map_location='cpu')
        except Exception as e:
            raise RuntimeError(f"Failed loading {path}: {e}")
        
        if t.dim() == 4 and t.size(0) == 1: t = t.squeeze(0)
        return t.float().contiguous()

    def __getitem__(self, idx):
        attempt_idx = idx
        
        while attempt_idx < len(self.files) - self.seq_len:
            frames = []
            valid_sequence = True
            
            for i in range(self.seq_len + 1):
                curr_idx = attempt_idx + i
                
                try:
                    if self.cache is not None:
                        t = self.cache[curr_idx]
                        if t is None: raise RuntimeError("Cached None")
                    else:
                        t = self._load_tensor(self.files[curr_idx])
                    frames.append(t)
                except Exception:
                    valid_sequence = False
                    break 
            
            if valid_sequence:
                context_tensor = torch.stack(frames[:-1], dim=0) 
                target_tensor  = frames[-1]                      
                
                if self.per_sample_normalize:
                    c_mean = context_tensor.mean(dim=(0,2,3), keepdim=True)
                    c_std  = context_tensor.std(dim=(0,2,3), keepdim=True) + self._eps
                    context_tensor = (context_tensor - c_mean) / c_std

                    t_mean = target_tensor.mean(dim=(1,2), keepdim=True)
                    t_std  = target_tensor.std(dim=(1,2), keepdim=True) + self._eps
                    target_tensor = (target_tensor - t_mean) / t_std

                if self.augment:
                    combined = torch.cat([context_tensor, target_tensor.unsqueeze(0)], dim=0)
                    if random.random() > 0.5: combined = torch.flip(combined, dims=[-1])
                    if random.random() > 0.5: combined = torch.flip(combined, dims=[-2])
                    context_tensor = combined[:-1]
                    target_tensor = combined[-1]

                return target_tensor, context_tensor
            
            else:
                if not self.skip_bad_files:
                    raise RuntimeError(f"Bad file encountered at index {curr_idx} and skip_bad_files=False")
                
                attempt_idx += 1
                continue

        raise IndexError("Could not find a valid sequence in the remaining dataset.")