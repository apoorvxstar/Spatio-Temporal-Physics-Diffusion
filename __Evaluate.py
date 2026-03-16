import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import Config as org  # <--- FIXED: Changed from Origin to Config
from Utils import prepare_batch_tensors

@torch.no_grad()
def evaluate(spine, eval_data, device, diff_timesteps, universe, rank, sqrt_alphas, sqrt_one_minus):
    """
    Evaluates the model on specific timesteps (EL_steps and XL_steps).
    """
    spine.eval()

    # Create sampler/loader locally for this evaluation phase
    sampler = DistributedSampler(eval_data, universe, rank, shuffle=False)
    loader  = DataLoader(eval_data,
                         org.batch_size // universe,
                         sampler=sampler,
                         num_workers=org.cores,
                         pin_memory=True)

    # Define probe steps
    el_steps = set(range(0, diff_timesteps, 100))
    xl_steps = set(range(900, diff_timesteps, 10))
    probe_steps = torch.tensor(sorted(el_steps | xl_steps), device=device)

    el_sum, xl_sum = 0.0, 0.0
    el_n, xl_n = 0, 0

    for idx, batch in enumerate(loader):
        target, context = prepare_batch_tensors(batch[0], batch[1], device)

        B = target.size(0)
        # Deterministic timestep selection for evaluation
        T = probe_steps[(torch.arange(B, device=device) + idx) % probe_steps.numel()]

        noise = torch.randn_like(target)
        # Diffusion forward process
        noised = sqrt_alphas[T, None, None, None] * target + sqrt_one_minus[T, None, None, None] * noise

        with torch.autocast("cuda", enabled=org.amp_on):
            out = spine(noised, T, context)

        mse = (out - noise).pow(2).mean(dim=[1,2,3])
        
        t_cpu = T.cpu().numpy()
        l_cpu = mse.cpu().numpy()

        for i in range(B):
            if t_cpu[i] in el_steps:
                el_sum += l_cpu[i]
                el_n += 1
            if t_cpu[i] in xl_steps:
                xl_sum += l_cpu[i]
                xl_n += 1

    # Aggregate results across GPUs
    if universe > 1:
        import torch.distributed as dist
        stats = torch.tensor([el_sum, el_n, xl_sum, xl_n], device=device)
        dist.all_reduce(stats)
        el_sum, el_n, xl_sum, xl_n = stats.tolist()

    spine.train()
    return el_sum / max(el_n, 1), xl_sum / max(xl_n, 1)