"""
Diffusion Policy for short-horizon trajectory prediction
输入: 高层规划文本 (string)，原始观测 RGBD (H x W x 4 tensor)
输出: 未来 n 秒内的预测轨迹 (sequence of 2D positions or poses)

说明: 这是一个可运行的单文件示例（PyTorch）。代码实现了：
 - 文本编码器接口（可接 HuggingFace CLIP）
 - 简单的 RGBD encoder (CNN)
 - 条件 DDPM 风格 diffusion model（1D U-Net 风格的 MLP + 条件融合）
 - 训练/采样辅助函数
 - 示例用法（生成随机数据进行快速测试）

注意：这是研究/原型代码，真实部署时请替换或微调编码器、数据加载、损失与安全检查。

依赖：torch, torchvision, transformers(可选), numpy

"""

import math
import typing as T
import time
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np

# 可选的 HuggingFace CLIP 文本编码器
try:
    from transformers import CLIPTokenizer, CLIPTextModel
    HAS_CLIP = True
except Exception:
    HAS_CLIP = False

# ----------------------------- Utilities ---------------------------------

def get_timestep_embedding(timesteps: torch.Tensor, dim: int):
    """Sin/cos timestep embedding, 1D (batch, dim)"""
    assert len(timesteps.shape) == 1
    half = dim // 2
    # ensure freqs tensor on same device as timesteps
    freqs = torch.exp(
        -math.log(10000) * torch.arange(0, half, dtype=torch.float32, device=timesteps.device) / half
    )
    args = timesteps.float().unsqueeze(1) * freqs.unsqueeze(0)
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=1)
    if dim % 2:
        emb = F.pad(emb, (0,1))
    return emb

# Small MLP block
class MLPBlock(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim=None):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = max(in_dim, out_dim)
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim)
        )
    def forward(self, x):
        return self.net(x)

# ------------------------- Encoders -------------------------------------
class SimpleTextEncoder(nn.Module):
    """Fallback text encoder (learned embeddings + transformer-lite)
    If transformers.CLIP available, user can replace with pretrained CLIP encoder.
    """
    def __init__(self, vocab_size=10000, emb_dim=256, out_dim=256, max_len=64, device='cuda'):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim)
        self.pos = nn.Embedding(max_len, emb_dim)
        # tiny transformer-like stack
        self.fc = nn.Sequential(nn.Linear(emb_dim, out_dim), nn.LayerNorm(out_dim))
        self.max_len = max_len
        self.to(device=device)
    
    def forward(self, token_ids: torch.LongTensor):
        L = token_ids.shape[1]
        pos_ids = torch.arange(L, device=token_ids.device).unsqueeze(0).expand_as(token_ids)
        x = self.emb(token_ids) + self.pos(pos_ids)
        x = x.mean(dim=1)
        return self.fc(x)

class CLIPTextWrapper(nn.Module):
    def __init__(self, model_name='openai/clip-vit-base-patch32'):
        super().__init__()
        if not HAS_CLIP:
            raise RuntimeError('transformers.CLIP not available in this environment')
        self.tokenizer = CLIPTokenizer.from_pretrained(model_name)
        self.model = CLIPTextModel.from_pretrained(model_name)
    def encode(self, texts: T.List[str], device='cuda'):
        toks = self.tokenizer(texts, padding=True, truncation=True, return_tensors='pt')
        toks = {k: v.to(device) for k,v in toks.items()}
        out = self.model(**toks)
        # take pooled output
        return out.pooler_output

class RGBDEncoder(nn.Module):
    """Simple CNN to encode RGBD (4 channels) into a compact vector
    Input shape: (B, 4, H, W)
    Output: (B, feat_dim)
    """
    def __init__(self, in_ch=4, feat_dim=256):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, 32, 3, stride=2, padding=1), nn.GELU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.GELU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.GELU(),
            nn.AdaptiveAvgPool2d(1)
        )
        self.fc = nn.Linear(128, feat_dim)
    def forward(self, x):
        # x: (B,4,H,W)
        h = self.conv(x).view(x.shape[0], -1)
        return self.fc(h)

# ------------------------- Diffusion Model -------------------------------
class TrajectoryDiffusionModel(nn.Module):
    """Conditional diffusion for trajectories.
    We represent a trajectory as a flattened vector of length T * dim_per_step
    For example, T=20, dim_per_step=3 (x,y,theta) => vector length 60.

    The model predicts noise given noised trajectory x_t, timestep t, and condition vector c.
    """
    def __init__(self, traj_len: int, step_dim: int, cond_dim: int, hidden_dim=512):
        super().__init__()
        self.traj_len = traj_len
        self.step_dim = step_dim
        self.in_dim = traj_len * step_dim
        self.cond_dim = cond_dim
        self.time_embed_dim = 128

        self.time_mlp = MLPBlock(self.time_embed_dim, self.time_embed_dim)
        # conditioning projector for text + rgbd
        self.cond_proj = nn.Sequential(nn.Linear(cond_dim, hidden_dim), nn.GELU())

        # core denoiser MLP (like a tiny U-Net for 1D vector)
        self.net = nn.Sequential(
            nn.Linear(self.in_dim + hidden_dim + self.time_embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, self.in_dim)
        )

    def forward(self, x_noisy: torch.Tensor, t: torch.Tensor, cond: torch.Tensor):
        # x_noisy: (B, in_dim)
        # t: (B,) long
        # cond: (B, cond_dim)
        B = x_noisy.shape[0]
        t_emb = get_timestep_embedding(t, self.time_embed_dim).to(x_noisy.device)
        t_emb = self.time_mlp(t_emb)
        c = self.cond_proj(cond)
        h = torch.cat([x_noisy, c, t_emb], dim=1)
        return self.net(h)

# ------------------------- Diffusion Utilities ---------------------------
class DDPM(object):
    def __init__(self, model: nn.Module, betas: T.Optional[np.ndarray]=None, device='cuda'):
        self.model = model
        self.device = device
        self.in_dim = model.in_dim
        if betas is None:
            betas = np.linspace(1e-4, 0.02, 1000, dtype=np.float32)
        self.betas = torch.from_numpy(betas).to(device)
        self.alphas = 1.0 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)
        self.num_timesteps = len(betas)

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor=None):
        # x0: (B, D)
        if noise is None:
            noise = torch.randn_like(x0)
        a_bar = self.alpha_bars[t].unsqueeze(1)
        return torch.sqrt(a_bar) * x0 + torch.sqrt(1 - a_bar) * noise

    def predict_noise_from_model(self, x_t: torch.Tensor, t: torch.Tensor, cond: torch.Tensor):
        return self.model(x_t, t, cond)

    def p_sample(self, x_t: torch.Tensor, t_idx: int, cond: torch.Tensor, guidance_scale: float=1.0):
        # single step reverse sampling using simplified DDPM posterior mean estimate
        t = torch.full((x_t.shape[0],), t_idx, dtype=torch.long, device=x_t.device)
        beta_t = self.betas[t_idx]
        a_t = self.alphas[t_idx]
        a_bar_t = self.alpha_bars[t_idx]
        # predict noise
        eps_theta = self.predict_noise_from_model(x_t, t, cond)
        # estimate x0
        x0_pred = (x_t - torch.sqrt(1 - a_bar_t) * eps_theta) / torch.sqrt(a_bar_t)
        # compute posterior mean
        coef1 = (torch.sqrt(self.alpha_bars[t-1]) * beta_t) / (1 - a_bar_t)
        coef2 = (torch.sqrt(a_t) * (1 - self.alpha_bars[t-1])) / (1 - a_bar_t)
        # handle t==0 case
        coef1 = coef1.unsqueeze(1)
        coef2 = coef2.unsqueeze(1)
        mean = coef1 * x0_pred + coef2 * x_t
        # sample noise
        if t_idx > 0:
            noise = torch.randn_like(x_t)
            sigma = torch.sqrt(beta_t)
            return mean + sigma * noise
        else:
            return mean

    @torch.no_grad()
    def sample(self, cond: torch.Tensor, steps: int = 200, guidance_scale: float = 1.0, device=None):
        device = device or self.device
        B = cond.shape[0]
        x = torch.randn(B, self.in_dim, device=device)
        for i in reversed(range(steps)):
            x = self.p_sample(x, i, cond, guidance_scale=guidance_scale)
        return x

# ------------------------- Full Policy Module ---------------------------
class DiffusionPolicy(nn.Module):
    def __init__(self, traj_len=20, step_dim=2, text_dim=256, rgbd_dim=256, hidden_dim=512, device='cuda'):
        super().__init__()
        self.traj_len = traj_len
        self.step_dim = step_dim
        self.text_dim = text_dim
        self.rgbd_dim = rgbd_dim
        self.cond_dim = text_dim + rgbd_dim
        self.device = device

        self.text_encoder = SimpleTextEncoder(vocab_size=20000, emb_dim=256, out_dim=text_dim, device=device)
        self.rgbd_encoder = RGBDEncoder(in_ch=4, feat_dim=rgbd_dim).to(device)
        self.model = TrajectoryDiffusionModel(traj_len, step_dim, cond_dim=self.cond_dim, hidden_dim=hidden_dim).to(device)
        self.ddpm = DDPM(self.model, device=device)
        self.to(device)

    def encode_text_tokens(self, token_ids: torch.LongTensor):
        return self.text_encoder(token_ids.to(self.device))

    def forward(self, x0_traj: torch.Tensor, token_ids: torch.LongTensor, rgbd: torch.Tensor, t: torch.Tensor, noise=None):
        x0_traj = x0_traj.to(self.device)
        token_ids = token_ids.to(self.device)
        rgbd = rgbd.to(self.device)
        t = t.to(self.device)
        B = x0_traj.shape[0]
        text_feat = self.encode_text_tokens(token_ids)
        rgbd_feat = self.rgbd_encoder(rgbd)
        cond = torch.cat([text_feat, rgbd_feat], dim=1)
        if noise is None:
            noise = torch.randn_like(x0_traj)
        x_t = self.ddpm.q_sample(x0_traj, t, noise)
        pred_noise = self.model(x_t, t, cond)
        return pred_noise

    @torch.no_grad()
    def sample(self, token_ids: torch.LongTensor, rgbd: torch.Tensor, steps=200, device=None):
        device = device or self.device
        self.to(device)
        token_ids = token_ids.to(device)
        rgbd = rgbd.to(device)
        text_feat = self.encode_text_tokens(token_ids)
        rgbd_feat = self.rgbd_encoder(rgbd)
        cond = torch.cat([text_feat, rgbd_feat], dim=1)
        samples = self.ddpm.sample(cond, steps=steps, device=device)
        return samples.view(samples.shape[0], self.traj_len, self.step_dim)

# ------------------------- Example Dataset --------------------------------
class ToyTrajDataset(Dataset):
    """合成数据：以随机小幅度曲线作为示例"""
    def __init__(self, N=1000, traj_len=20, step_dim=2, H=64, W=64):
        super().__init__()
        self.N = N
        self.traj_len = traj_len
        self.step_dim = step_dim
        self.H = H; self.W = W
    def __len__(self):
        return self.N
    def __getitem__(self, idx):
        # random smooth curve
        t = np.linspace(0,1,self.traj_len)
        x = np.cumsum(0.05 * np.random.randn(self.traj_len))
        y = np.cumsum(0.05 * np.random.randn(self.traj_len))
        traj = np.stack([x,y], axis=1).astype(np.float32)
        traj = traj.reshape(-1)
        # random rgbd image
        rgb = np.random.rand(3, self.H, self.W).astype(np.float32)
        depth = np.random.rand(1, self.H, self.W).astype(np.float32)
        rgbd = np.concatenate([rgb, depth], axis=0)
        # toy token ids
        token_ids = np.random.randint(0, 20000, size=(16,), dtype=np.int64)
        return torch.from_numpy(traj), torch.from_numpy(token_ids), torch.from_numpy(rgbd)

# ------------------------- Training Loop (simple) -------------------------

def train_one_epoch(policy: DiffusionPolicy, dataloader: DataLoader, opt: torch.optim.Optimizer, device='cuda', profile=False):
    policy.train()
    mse = nn.MSELoss()
    total_loss = 0.0
    total_batches = 0
    total_samples = 0
    start_time = time.time()
    for traj, token_ids, rgbd in dataloader:
        traj = traj.to(device)
        token_ids = token_ids.to(device)
        rgbd = rgbd.to(device)
        B = traj.shape[0]
        t = torch.randint(0, policy.ddpm.num_timesteps, (B,), dtype=torch.long, device=device)
        noise = torch.randn_like(traj)
        x_t = policy.ddpm.q_sample(traj, t, noise)
        cond = torch.cat([policy.encode_text_tokens(token_ids), policy.rgbd_encoder(rgbd)], dim=1)
        pred_noise = policy.model(x_t, t, cond)
        loss = mse(pred_noise, noise)
        opt.zero_grad()
        loss.backward()
        opt.step()
        total_loss += loss.item() * B
        total_batches += 1
        total_samples += B
    elapsed = time.time() - start_time
    avg_loss = total_loss / len(dataloader.dataset)
    if profile:
        batches_per_sec = total_batches / elapsed if elapsed > 0 else 0.0
        samples_per_sec = total_samples / elapsed if elapsed > 0 else 0.0
        return avg_loss, elapsed, batches_per_sec, samples_per_sec
    return avg_loss

@torch.no_grad()
def benchmark_sampling(policy: DiffusionPolicy, token_ids: torch.LongTensor, rgbd: torch.Tensor, device='cuda', steps=100, repeats=10):
    policy.eval()
    token_ids = token_ids.to(device)
    rgbd = rgbd.to(device)
    # warmup
    _ = policy.sample(token_ids, rgbd, steps=steps, device=device)
    start = time.time()
    for _ in range(repeats):
        _ = policy.sample(token_ids, rgbd, steps=steps, device=device)
    elapsed = time.time() - start
    samples_per_sec = repeats / elapsed if elapsed > 0 else 0.0
    return elapsed, samples_per_sec

# ------------------------- Quick smoke test --------------------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--traj-len', type=int, default=20)
    parser.add_argument('--step-dim', type=int, default=2)
    parser.add_argument('--data-size', type=int, default=200)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--benchmark-train', action='store_true')
    parser.add_argument('--benchmark-sample', action='store_true')
    parser.add_argument('--sample-steps', type=int, default=100)
    parser.add_argument('--sample-repeats', type=int, default=10)
    args = parser.parse_args()

    device = args.device
    traj_len = args.traj_len
    step_dim = args.step_dim
    policy = DiffusionPolicy(traj_len=traj_len, step_dim=step_dim, device=device)
    dataset = ToyTrajDataset(N=args.data_size, traj_len=traj_len, step_dim=step_dim, H=64, W=64)
    dl = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    opt = torch.optim.Adam(policy.parameters(), lr=args.lr)

    if args.benchmark_train:
        for epoch in range(args.epochs):
            loss, elapsed, bps, sps = train_one_epoch(policy, dl, opt, device=device, profile=True)
            print(f'[Epoch {epoch+1}/{args.epochs}] loss={loss:.4f} time={elapsed:.3f}s batches/s={bps:.2f} samples/s={sps:.2f}')
    else:
        print('Running one training epoch (toy) ...')
        loss = train_one_epoch(policy, dl, opt, device=device)
        print('Epoch loss:', loss)

    traj, token_ids, rgbd = dataset[0]
    token_ids = token_ids.unsqueeze(0)
    rgbd = rgbd.unsqueeze(0)
    print('Sampling test ...')
    samples = policy.sample(token_ids, rgbd, steps=args.sample_steps, device=device)
    print('Sampled trajectory shape:', samples.shape)
    if args.benchmark_sample:
        elapsed, sps = benchmark_sampling(policy, token_ids, rgbd, device=device, steps=args.sample_steps, repeats=args.sample_repeats)
        print(f'Sampling benchmark: total_time={elapsed:.3f}s repeats={args.sample_repeats} samples/s={sps:.2f}')
    # print first trajectory (optional)
    for trj in samples[0]:
        print(trj)

# End of file
