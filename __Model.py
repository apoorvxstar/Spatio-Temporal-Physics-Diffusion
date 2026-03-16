import math
import torch
import torch.nn as nn
from einops import rearrange
import Config as org  # <--- FIXED: Changed from Origin to Config

# ==========================================
# 1. Noise schedule / Time vectors
# ==========================================

def cosine_beta_schedule(time_stp, s: float = 0.008):
    """
    Generates a cosine noise schedule for diffusion.
    """
    nxt_stp = time_stp + 1
    t = torch.linspace(0, time_stp, nxt_stp)
    alphas_cumprod = torch.cos((t / time_stp + s) / (1 + s) * math.pi / 2) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)

def Diffusion_Terms(diff_timesteps, device):
    """
    Returns the alpha/beta terms needed for the forward diffusion process.
    """
    betas = cosine_beta_schedule(diff_timesteps).to(device)
    alphas = 1. - betas
    alphas_cumprod = torch.cumprod(alphas, axis=0)
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)
    return sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod

def sincos_time_vectors(time_stp: torch.Tensor, dim: int):
    """
    Sinusoidal time embeddings (Positional Encodings) for the time steps.
    """
    assert dim % 2 == 0, "Even time-dim required"
    time_vec = torch.exp(torch.arange(dim // 2, device=time_stp.device) * -math.log(10000) / (dim // 2 - 1))
    time_vec = time_stp[:, None] * time_vec[None, :]
    time_vec = torch.cat([torch.sin(time_vec), torch.cos(time_vec)], dim=-1)
    return time_vec

# ==========================================
# 2. Layers (Residual, Attention, Up/Down)
# ==========================================

class ChannelConcat(nn.Module):
    """
    Concatenate target and context robustly along channels.
    Accepts context as either [B, seq_len, C, H, W] or [B, C*seq_len, H, W].
    """
    def forward(self, x, context):
        if context is None:
            return x
        if context.dim() == 5:
            B, S, C, H, W = context.shape
            context = context.view(B, S * C, H, W)
        return torch.cat([x, context], dim=1)

class ResidualStage(nn.Module):
    def __init__(self, C_in, C_out, time_dim, grps):
        super().__init__()
        self.exnorm = nn.GroupNorm(grps, C_in)
        self.exact = nn.SiLU()
        self.exconv = nn.Conv2d(C_in, C_out, 3, padding=1)

        self.renorm = nn.GroupNorm(grps, C_out)
        self.react = nn.SiLU()
        self.reconv = nn.Conv2d(C_out, C_out, 3, padding=1)
        self.drop = nn.Dropout(org.free)

        # FiLM: produce scale & shift based on time
        self.time_nrv = nn.Sequential(nn.SiLU(), nn.Linear(time_dim, C_out * 2))

        self.resconv = nn.Conv2d(C_in, C_out, 1) if C_in != C_out else nn.Identity()

    def forward(self, X, time_vec):
        residual = self.resconv(X)
        latent = self.exnorm(X)
        latent = self.exact(latent)
        latent = self.exconv(latent)

        time_pro = self.time_nrv(time_vec)
        scale, shift = time_pro.chunk(2, dim=-1)
        scale = scale.unsqueeze(-1).unsqueeze(-1)
        shift = shift.unsqueeze(-1).unsqueeze(-1)

        latent = self.renorm(latent)
        latent = latent * (1 + scale) + shift
        latent = self.react(latent)
        latent = self.reconv(latent)
        latent = self.drop(latent)

        return latent + residual

class SelfAttention(nn.Module):
    def __init__(self, C, attn_heads, grps, free):
        super().__init__()
        assert C % attn_heads == 0, "Channels must be divisible by attention heads"
        self.attn_heads = attn_heads
        self.norm = nn.GroupNorm(grps, C)
        self.qkv = nn.Conv2d(C, C * 3, 1, bias=False)
        self.proj = nn.Sequential(nn.Conv2d(C, C, 1), nn.Dropout(free))

    def forward(self, X):
        B, C, x, y = X.shape
        qkv = self.qkv(self.norm(X)).chunk(3, dim=1)
        Q, K, V = [rearrange(tensor, "b (h d) x y -> b h (x y) d", h=self.attn_heads) for tensor in qkv]
        Y = nn.functional.scaled_dot_product_attention(Q, K, V)
        Y = rearrange(Y, "b h (x y) d -> b (h d) x y", h=self.attn_heads, x=x, y=y)
        return X + self.proj(Y)

class AttentionStage(nn.Module):
    def __init__(self, C, attn_heads, grps, free, context_dim=None):
        super().__init__()
        self.sa = SelfAttention(C, attn_heads, grps, free)

    def forward(self, X, context=None):
        return self.sa(X)

class DownStage(nn.Module):
    def __init__(self, C_in, C_out):
        super().__init__()
        # use conv stride 2 for learned downsampling
        self.conv = nn.Conv2d(C_in, C_out, 3, stride=2, padding=1)
        self.norm = nn.GroupNorm(org.grps, C_out)
        self.act = nn.SiLU()
    def forward(self, X):
        return self.conv(self.act(self.norm(X)))

class UpStage(nn.Module):
    def __init__(self, C):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv = nn.Conv2d(C, C, 3, padding=1)
        self.norm = nn.GroupNorm(org.grps, C)
        self.act = nn.SiLU()
    def forward(self, X):
        return self.conv(self.act(self.norm(self.upsample(X))))

class UNetLayers(nn.Module):
    def __init__(self, C_in, C_out, time_dim, grps, num_res_stg, attn_heads, free, context=None, is_attn=False):
        super().__init__()
        self.res = nn.ModuleList()
        for _ in range(num_res_stg):
            self.res.append(ResidualStage(C_in, C_out, time_dim, grps))
            C_in = C_out
        self.attn = AttentionStage(C_out, attn_heads, grps, free, context) if is_attn else nn.Identity()

    def forward(self, X, time_vec, context=None):
        for layer in self.res:
            X = layer(X, time_vec)
        if isinstance(self.attn, AttentionStage):
            X = self.attn(X, context)
        else:
            X = self.attn(X)
        return X

# ==========================================
# 3. Main UNet (Latent-Adapted)
# ==========================================

class UNet(nn.Module):
    def __init__(self):
        super().__init__()

        # Pull config from Config.py
        chan = org.chan
        img_z = org.img_z
        seq_len = org.seq_len

        time_dim = chan * 4
        self.time_nrv = nn.Sequential(
            nn.Linear(chan, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim)
        )

        # Context (we won't use a separate encoder; we'll concat flattened context to target)
        self.concat = ChannelConcat()
        input_channels = img_z + (img_z * seq_len)  # target + flattened context

        self.start_conv = nn.Conv2d(input_channels, chan, 3, padding=1)
        self.end_conv = nn.Conv2d(chan, img_z, 3, padding=1)

        # Build DOWN path using org.chan_lvl
        self.down = nn.ModuleList()
        C_in = chan
        for i, lvl in enumerate(org.chan_lvl):
            C_out = chan * lvl
            is_attention = org.attn_lvl[i] if i < len(org.attn_lvl) else False
            
            self.down.append(nn.ModuleList([
                UNetLayers(C_in, C_out, time_dim, org.grps, org.num_res_stg, org.attn_heads, org.free, None, is_attn=is_attention),
                DownStage(C_out, C_out)
            ]))
            C_in = C_out

        # Bottleneck (Link)
        C_out = C_in
        self.link = UNetLayers(C_in, C_out, time_dim, org.grps, org.num_res_stg, org.attn_heads, org.free, None, is_attn=True)

        # Build UP path
        self.up = nn.ModuleList()
        rev_levels = list(reversed(org.chan_lvl))
        
        # We need to know skip channel sizes in same order as created (but we pop them in reverse)
        # C_in is currently the bottleneck output
        for i, lvl in enumerate(rev_levels):
            C_out = chan * lvl
            # Calculate index for attention array (matching the reversed level)
            attn_idx = len(org.chan_lvl) - 1 - i
            is_attention = org.attn_lvl[attn_idx] if attn_idx >= 0 else False
            
            # The input to UNetLayers will be: (Input from UpStage) + (Skip Connection)
            # UpStage(C_in) outputs C_in size. 
            # Skip connection has size C_out (from the corresponding down level).
            
            self.up.append(nn.ModuleList([
                UNetLayers(C_in + C_out, C_out, time_dim, org.grps, org.num_res_stg, org.attn_heads, org.free, None, is_attn=is_attention),
                UpStage(C_in)
            ]))
            C_in = C_out

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if getattr(m, 'bias', None) is not None:
                    nn.init.zeros_(m.bias)
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight)
                if getattr(m, 'bias', None) is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, X, T, condition):
        """
        X: [B, img_z, H, W]  (noisy target)
        condition: [B, img_z*seq_len, H, W] or [B, seq_len, img_z, H, W]
        T: (B,) timesteps
        """

        # 1. Time Embedding
        T = sincos_time_vectors(T, org.chan)
        T = self.time_nrv(T)

        # 2. Condition Formatting
        if condition is not None and condition.dim() == 5:
            pass # [B, S, C, H, W] -> concat handles it
        elif condition is not None and condition.dim() == 4:
            pass # [B, C*S, H, W] -> ok
        elif condition is None:
            B, C, H, W = X.shape
            zeros = torch.zeros((B, org.img_z * org.seq_len, H, W), device=X.device, dtype=X.dtype)
            condition = zeros

        # 3. Initial Concat & Conv
        X = self.concat(X, condition)
        X = self.start_conv(X)

        # 4. Down Path
        skip = []
        for stg, down in self.down:
            X = stg(X, T, None)
            skip.append(X)
            X = down(X)

        # 5. Bottleneck
        X = self.link(X, T, None)

        # 6. Up Path
        for stg, up in self.up:
            X = up(X) # Upsample
            sc = skip.pop() # Retrieve Skip Connection
            
            # Spatial safety check
            if X.shape[-2:] != sc.shape[-2:]:
                X = nn.functional.interpolate(X, size=sc.shape[-2:], mode='bilinear', align_corners=False)
            
            X = torch.cat([X, sc], dim=1) # Concat
            X = stg(X, T, None) # Refine

        return self.end_conv(X)