import torch
import torch.nn as nn
import os
import time
import torch.multiprocessing as mp
from torch.optim import AdamW
from torch.utils.data import DataLoader, Subset
from torch.optim.lr_scheduler import LinearLR, ConstantLR, CosineAnnealingLR, CosineAnnealingWarmRestarts, SequentialLR
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torchinfo import summary

# Import custom modules
import Config as org            # <--- FIXED: Changed from Origin to Config
import Model as Spine           # Renaming Model to Spine to match your logic
from Dataset import SequentialDataset
from Utils import EMA, save_graph, prepare_batch_tensors, save_checkpoint
from Evaluate import evaluate

# ==========================================
# 1. LR Scheduler
# ==========================================

def LRscheduler(optimizer):
    acc = 1000
    l1 = LinearLR(optimizer, 0.01, 1.0, acc)

    if org.lr_class == "Constant":
        l2 = ConstantLR(optimizer, total_iters=org.steps_all-acc)
    elif org.lr_class == "CosAWR":
        l2 = CosineAnnealingWarmRestarts(optimizer, 10000)
    else:
        l2 = CosineAnnealingLR(optimizer, org.steps_all-acc, 1e-6)

    return SequentialLR(optimizer, [l1,l2], [acc])

# ==========================================
# 2. Train Loop
# ==========================================

def train_worker(rank, universe, learn_data, eval_data):

    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
    
    if universe > 1:
        org.setup(rank, universe)

    # Initialize Model
    model = Spine.UNet().to(device)
    if universe > 1:
        model = DDP(model, device_ids=[rank])

    net = model.module if universe > 1 else model

    # Optimizer & Scheduler
    optimizer = AdamW(net.parameters(), lr=org.lr, weight_decay=org.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=org.amp_on)
    scheduler = LRscheduler(optimizer)

    # Utilities
    ema = EMA(net, org.ema_decay)
    sqrt_alphas, sqrt_one_minus = Spine.Diffusion_Terms(org.diff_timesteps, device)
    loss_fn = nn.MSELoss()

    # Data Loader
    sampler = DistributedSampler(learn_data, universe, rank, shuffle=True)
    loader = DataLoader(learn_data,
                        org.batch_size // universe,
                        sampler=sampler,
                        num_workers=org.cores,
                        pin_memory=True)

    history = {"steps":[], "L":[], "AL":[], "EL":[], "XL":[]}
    
    steps = 0
    total_loss = 0.0
    t0 = time.time()

    print(f"[GPU {rank}] Starting training...")

    while steps < org.steps_all:
        sampler.set_epoch(steps)

        for batch in loader:
            # Prepare Data
            target, context = prepare_batch_tensors(batch[0], batch[1], device)

            # Noise Generation
            T = torch.randint(0, org.diff_timesteps, (target.size(0),), device=device)
            noise = torch.randn_like(target)
            x_noised = sqrt_alphas[T,None,None,None]*target + sqrt_one_minus[T,None,None,None]*noise

            # Forward & Backward
            optimizer.zero_grad()
            with torch.autocast(device_type="cuda", enabled=org.amp_on):
                out = model(x_noised, T, context)
                loss = loss_fn(out, noise)

            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            
            ema.update()

            # Logging
            steps += 1
            total_loss += loss.item()

            # Evaluation & Saving
            if steps % org.save_steps == 0 or steps == org.steps_all:
                ema.apply_shadow()
                
                # Run Evaluation
                EL, XL = evaluate(net, eval_data, device, org.diff_timesteps,
                                 universe, rank, sqrt_alphas, sqrt_one_minus)

                if rank == 0:
                    history["steps"].append(steps)
                    history["L"].append(loss.item())
                    history["AL"].append(total_loss/steps)
                    history["EL"].append(EL)
                    history["XL"].append(XL)

                    save_graph(history, "Results/Loss_Graph.png")
                    save_checkpoint(ema.shadow, "Saved/Lightning.pth")
                    print(f"\n[INFO] Saved checkpoint at step {steps}")

                ema.restore()

            # Progress Bar (Rank 0 only)
            if rank == 0:
                p = steps/org.steps_all
                bar = "■"*int(40*p) + "-"*(40-int(40*p))
                print(f"\r|{bar}| {p*100:5.1f}% [{steps}/{org.steps_all}] "
                      f"L:{loss.item():.4f} AL:{total_loss/steps:.4f} "
                      f"LR:{scheduler.get_last_lr()[0]:.2e}", end="")

            if steps >= org.steps_all:
                break

    if universe > 1:
        org.cleanup()

# ==========================================
# 3. Main Execution
# ==========================================

if __name__ == "__main__":
    org.seed_everything(42)

    # Load Data
    full_ds = SequentialDataset(root=org.RAW_DATA, seq_len=org.seq_len, augment=True)
    
    # Split Data (85/15)
    split = int(len(full_ds) * 0.85)
    learn_data = Subset(full_ds, range(split))
    eval_data  = Subset(full_ds, range(split, len(full_ds)))

    print(f"Train Size: {len(learn_data)} | Eval Size: {len(eval_data)}")

    # Model Summary (Sanity Check)
    unet_input_channels = org.img_z + (org.seq_len * org.img_z)
    dummy_x = torch.randn(1, org.img_z, org.img_x, org.img_y) 
    dummy_t = torch.randint(0, 1000, (1,)) 
    dummy_context = torch.randn(1, unet_input_channels - org.img_z, org.img_x, org.img_y)
    
    summary(Spine.UNet(), input_data=(dummy_x, dummy_t, dummy_context), depth=1)

    # Launch Training
    if org.universe > 1:
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = org.find_port()
        mp.spawn(train_worker, args=(org.universe, learn_data, eval_data), nprocs=org.universe)
    else:
        train_worker(0, 1, learn_data, eval_data)