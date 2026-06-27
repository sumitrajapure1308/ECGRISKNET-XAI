
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict


class ResBlock(nn.Module):
    def __init__(self, ch, k=7, dropout=0.2):
        super().__init__()
        p = k // 2
        self.net = nn.Sequential(
            nn.Conv1d(ch, ch, k, padding=p, bias=False),
            nn.GroupNorm(min(8, ch), ch), nn.GELU(), nn.Dropout(dropout),
            nn.Conv1d(ch, ch, k, padding=p, bias=False),
            nn.GroupNorm(min(8, ch), ch), nn.GELU(),
        )
    def forward(self, x): return x + self.net(x)


class InceptionBlock(nn.Module):
    def __init__(self, in_ch, out_ch_per_branch=32, bottleneck=32, dropout=0.1):
        super().__init__()
        self.bottleneck     = nn.Conv1d(in_ch, bottleneck, 1, bias=False)
        self.conv3          = nn.Conv1d(bottleneck, out_ch_per_branch, 3,  padding=1,  bias=False)
        self.conv7          = nn.Conv1d(bottleneck, out_ch_per_branch, 7,  padding=3,  bias=False)
        self.conv15         = nn.Conv1d(bottleneck, out_ch_per_branch, 15, padding=7,  bias=False)
        self.conv31         = nn.Conv1d(bottleneck, out_ch_per_branch, 31, padding=15, bias=False)
        self.maxpool_branch = nn.Sequential(
            nn.MaxPool1d(3, stride=1, padding=1),
            nn.Conv1d(in_ch, out_ch_per_branch, 1, bias=False),
        )
        total_out  = out_ch_per_branch * 5
        self.norm  = nn.GroupNorm(min(8, total_out), total_out)
        self.act   = nn.GELU()
        self.drop  = nn.Dropout(dropout)
        self.proj  = nn.Conv1d(total_out, out_ch_per_branch * 4, 1, bias=False)
    def forward(self, x):
        b = self.bottleneck(x)
        out = torch.cat([self.conv3(b), self.conv7(b), self.conv15(b),
                         self.conv31(b), self.maxpool_branch(x)], dim=1)
        return self.proj(self.drop(self.act(self.norm(out))))


class SEBlock(nn.Module):
    def __init__(self, ch, reduction=16):
        super().__init__()
        mid = max(ch // reduction, 4)
        self.fc = nn.Sequential(nn.Linear(ch, mid), nn.ReLU(), nn.Linear(mid, ch), nn.Sigmoid())
        self.last_weights: Optional[torch.Tensor] = None
    def forward(self, x):
        w = self.fc(x.mean(dim=-1)); self.last_weights = w.detach().cpu()
        return x * w.unsqueeze(-1)


class TransformerBlock(nn.Module):
    def __init__(self, d_model, heads=8, dropout=0.1, ffn_mult=4):
        super().__init__()
        self.attn  = nn.MultiheadAttention(d_model, heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model); self.norm2 = nn.LayerNorm(d_model)
        self.ffn   = nn.Sequential(
            nn.Linear(d_model, d_model * ffn_mult), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_model * ffn_mult, d_model), nn.Dropout(dropout),
        )
        self.last_attn_weights: Optional[torch.Tensor] = None
    def forward(self, x):
        out, w = self.attn(x, x, x, need_weights=True, average_attn_weights=False)
        if w is not None: self.last_attn_weights = w.detach().cpu()
        x = self.norm1(x + out); x = self.norm2(x + self.ffn(x)); return x


class AttentionPooling(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.query = nn.Linear(d_model, 1)
        self.last_pool_weights: Optional[torch.Tensor] = None
    def forward(self, x):
        w = torch.softmax(self.query(x), dim=1); self.last_pool_weights = w.detach().cpu()
        return (x * w).sum(dim=1)


class MetadataMLP(nn.Module):
    def __init__(self, in_dim=3, out_dim=16):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, 32), nn.GELU(), nn.Linear(32, out_dim), nn.GELU())
    def forward(self, m): return self.net(m)


class ECGRiskNetXAI(nn.Module):
    def __init__(self, in_ch=12, base_ch=64, num_risk_classes=4, meta_dim=3, dropout=0.3):
        super().__init__()
        self.stem      = nn.Sequential(
            nn.Conv1d(in_ch, base_ch, 15, padding=7, bias=False),
            nn.GroupNorm(min(8, base_ch), base_ch), nn.GELU(), nn.MaxPool1d(2),
        )
        self.inception = InceptionBlock(base_ch, 32, base_ch, dropout * 0.5)
        incep_ch = 128; mid_ch = base_ch * 4  # 256
        self.down      = nn.Sequential(
            nn.Conv1d(incep_ch, mid_ch, 5, stride=2, padding=2, bias=False),
            nn.GroupNorm(min(8, mid_ch), mid_ch), nn.GELU(),
            ResBlock(mid_ch, 5, dropout), ResBlock(mid_ch, 5, dropout),
        )
        self.se         = SEBlock(mid_ch, 16)
        self.pool_down  = nn.MaxPool1d(2)
        self.transformer = nn.ModuleList([TransformerBlock(mid_ch, 8, dropout * 0.5) for _ in range(4)])
        self.attn_pool   = AttentionPooling(mid_ch)
        meta_embed_dim   = 16
        self.meta_mlp    = MetadataMLP(meta_dim, meta_embed_dim)
        fused_dim        = mid_ch + meta_embed_dim
        self.shared_embed = nn.Sequential(
            nn.Linear(fused_dim, mid_ch), nn.LayerNorm(mid_ch), nn.GELU(), nn.Dropout(dropout),
        )
        self.proj_head   = nn.Sequential(nn.Linear(mid_ch, 256), nn.ReLU(), nn.Linear(256, 128))
        def _head(out): return nn.Sequential(
            nn.Linear(mid_ch, 64), nn.GELU(), nn.Dropout(dropout * 0.5), nn.Linear(64, out)
        )
        self.risk_head       = _head(4)
        self.arrhythmia_head = _head(6)
        self.mi_head         = _head(2)
        self.st_t_head       = _head(3)
        self.conduction_head = _head(4)
        self.last_attn_weights: Optional[torch.Tensor] = None
        self.last_pool_weights: Optional[torch.Tensor] = None
        self.last_se_weights:   Optional[torch.Tensor] = None

    def forward(self, ecg: torch.Tensor, meta: torch.Tensor) -> Dict[str, torch.Tensor]:
        x = self.pool_down(self.se(self.down(self.inception(self.stem(ecg)))))
        self.last_se_weights = self.se.last_weights
        x = x.transpose(1, 2)
        for blk in self.transformer: x = blk(x)
        self.last_attn_weights = self.transformer[-1].last_attn_weights
        emb  = self.attn_pool(x); self.last_pool_weights = self.attn_pool.last_pool_weights
        fused  = torch.cat([emb, self.meta_mlp(meta)], 1)
        shared = self.shared_embed(fused)
        proj   = F.normalize(self.proj_head(shared), dim=1)
        return {
            "risk": self.risk_head(shared), "arrhythmia": self.arrhythmia_head(shared),
            "mi": self.mi_head(shared), "st_t": self.st_t_head(shared),
            "conduction": self.conduction_head(shared),
            "projection": proj, "embedding": shared,
        }


class ClassBalancedFocalLoss(nn.Module):
    def __init__(self, samples_per_class, gamma=2.0, beta=0.9999, label_smoothing=0.1):
        super().__init__()
        self.gamma = gamma; self.ls = label_smoothing
        s = torch.tensor(samples_per_class, dtype=torch.float32)
        w = (1.0 - beta) / (1.0 - beta ** s)
        w = w / w.sum() * len(w)
        self.register_buffer("weights", w)
    def forward(self, logits, targets):
        C = logits.size(1)
        with torch.no_grad():
            sm = torch.full_like(logits, self.ls / (C - 1))
            sm.scatter_(1, targets.unsqueeze(1), 1.0 - self.ls)
        log_p = F.log_softmax(logits, dim=1)
        ce    = -(sm * log_p).sum(dim=1)
        pt    = torch.exp(log_p).gather(1, targets.unsqueeze(1)).squeeze(1)
        return ((loss := ((1 - pt) ** self.gamma) * ce) * self.weights.to(logits.device)[targets]).mean()


class SupConLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature
    def forward(self, features, labels):
        device = features.device; B = features.shape[0]
        sim    = torch.matmul(features, features.T) / self.temperature
        labels = labels.contiguous().view(-1, 1)
        pos_mask = torch.eq(labels, labels.T).float().to(device); pos_mask.fill_diagonal_(0)
        neg_mask = 1 - pos_mask - torch.eye(B, device=device)
        sim_max, _ = sim.max(dim=1, keepdim=True); sim = sim - sim_max.detach()
        exp_sim    = torch.exp(sim)
        log_prob   = sim - torch.log((exp_sim * (pos_mask + neg_mask)).sum(dim=1, keepdim=True) + 1e-8)
        n_pos = pos_mask.sum(dim=1)
        loss  = -(pos_mask * log_prob).sum(dim=1) / (n_pos + 1e-8)
        return loss[n_pos > 0].mean() if (n_pos > 0).any() else torch.tensor(0.0, device=device)


class FocalLoss(nn.Module):
    """Legacy alias for backward compatibility."""
    def __init__(self, gamma=2.0, weight=None, label_smoothing=0.1):
        super().__init__()
        self.gamma = gamma; self.weight = weight; self.ls = label_smoothing
    def forward(self, logits, targets):
        C = logits.size(1)
        with torch.no_grad():
            sm = torch.full_like(logits, self.ls / (C - 1))
            sm.scatter_(1, targets.unsqueeze(1), 1.0 - self.ls)
        log_p = F.log_softmax(logits, dim=1)
        ce    = -(sm * log_p).sum(dim=1)
        pt    = torch.exp(log_p).gather(1, targets.unsqueeze(1)).squeeze(1)
        loss  = ((1 - pt) ** self.gamma) * ce
        if self.weight is not None: loss = loss * self.weight[targets]
        return loss.mean()
