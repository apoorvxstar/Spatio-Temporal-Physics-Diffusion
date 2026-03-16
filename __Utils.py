import torch
import os
import matplotlib.pyplot as plt
import Config as org  # <--- FIXED: Changed from Origin to Config

# ==========================================
# 1. Directory & Saving Helpers
# ==========================================

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
        # print(f"[INFO] Created directory: {path}")

def save_checkpoint(state, filename="Saved/checkpoint.pth"):
    ensure_dir("Saved")
    torch.save(state, filename)

def save_graph(history, save_path):
    ensure_dir("Results")
    if len(history["steps"]) == 0:
        return

    plt.figure(figsize=(10, 6))
    plt.plot(history['steps'], history['L'], label='Current Loss', alpha=0.3)
    plt.plot(history['steps'], history['AL'], label='Avg Train Loss')
    plt.plot(history['steps'], history['EL'], label='Eval Loss', linewidth=2)
    plt.plot(history['steps'], history['XL'], label='High Noise Loss', linestyle='--')

    plt.title("Training Metrics")
    plt.xlabel("Steps")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(save_path)
    plt.close()

# ==========================================
# 2. EMA (Exponential Moving Average)
# ==========================================

class EMA:
    def __init__(self, spine, decay):
        self.spine = spine
        self.decay = decay
        self.shadow = {}
        self.store = {}
        self.register()

    def register(self):
        for name, param in self.spine.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.spine.named_parameters():
            if param.requires_grad:
                self.shadow[name].data = (1.0 - self.decay) * param.data + self.decay * self.shadow[name].data

    def apply_shadow(self):
        for name, param in self.spine.named_parameters():
            if param.requires_grad:
                self.store[name] = param.data.clone()
                param.data.copy_(self.shadow[name].data)

    def restore(self):
        for name, param in self.spine.named_parameters():
            if param.requires_grad:
                param.data.copy_(self.store[name].data)
        self.store = {}

# ==========================================
# 3. Data Prep Helpers
# ==========================================

def prepare_batch_tensors(target, context, device):
    target = target.to(device, non_blocking=True)
    context = context.to(device, non_blocking=True)

    # Target shape fix: [B, 1, C, H, W] -> [B, C, H, W]
    if target.dim() == 5 and target.shape[1] == 1:
        target = target.squeeze(1)

    # Context: [B,S,C,H,W] -> [B,S*C,H,W] (Flatten sequence)
    if context.dim() == 5:
        B, S, C, H, W = context.shape
        context = context.reshape(B, S * C, H, W)

    return target, context