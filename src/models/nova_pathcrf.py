# SPDX-License-Identifier: MPL-2.0
"""Pure-PyTorch adaptation of PathCRF (Nova Solutions SkillCorner fine-tuned checkpoint).

Upstream attribution (Mozilla Public License 2.0):
- Base architecture & implementation: Kim et al. (KDD 2026),
  https://github.com/hyunsungkim-ds/pathcrf (commit 1b7c6f8, MPL-2.0)
- Fine-tuned weights: Nova Solutions (`Youssef16Ahmed/nova-pathcrf-finetuned`,
  round3/final/model/state_dict_best_team.pt, MPL-2.0), fine-tuned exclusively on
  external SkillCorner Open Data (MIT) broadcast tracking.

Zero competition match data or labels are bundled or referenced here.
Adapts 2D tracking slots (including anonymous multi-slot ATD with gaps/swaps) into
ball-free 5 fps overlapping 10 s windows, runs SetTransformer + SAB + DynamicSparseCRF
marginals, and maps possession probabilities back onto the 25 fps timeline.
"""

import json
import math
import os
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.signal import savgol_filter
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODELS_DIR = os.path.join(ROOT, "models")
DEFAULT_CKPT_PATH = os.path.join(MODELS_DIR, "nova_pathcrf_best_team.pt")
DEFAULT_ARGS_PATH = os.path.join(MODELS_DIR, "nova_pathcrf_args.json")

PITCH_X = 105.0
PITCH_Y = 68.0
MAX_PLAYER_SPEED = 12.0  # m/s physical cap to suppress ATD slot-swap jumps
FPS = 25.0


# -----------------------------------------------------------------------------
# 1. SetTransformer Building Blocks (matching hyunsungkim-ds/pathcrf @ 1b7c6f8)
# -----------------------------------------------------------------------------

class Attention(nn.Module):
    """Scaled dot-product attention."""

    def __init__(self, temperature: float):
        super().__init__()
        self.temperature = temperature
        self.softmax = nn.Softmax(dim=2)

    def forward(self, queries: torch.Tensor, keys: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
        attention = torch.bmm(queries, keys.transpose(1, 2))
        attention = self.softmax(attention / self.temperature)
        return torch.bmm(attention, values)


class MultiheadAttention(nn.Module):
    """Multi-head attention layer without external dependencies."""

    def __init__(self, d: int, h: int) -> None:
        super().__init__()
        assert d % h == 0
        self.h = h
        p = d // h
        self.project_queries = nn.Linear(d, d)
        self.project_keys = nn.Linear(d, d)
        self.project_values = nn.Linear(d, d)
        self.concatenation = nn.Linear(d, d)
        self.attention = Attention(temperature=p ** 0.5)

    def forward(self, queries: torch.Tensor, keys: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
        h = self.h
        b, n, d = queries.size()
        _, m, _ = keys.size()
        p = d // h

        queries = self.project_queries(queries).view(b, n, h, p).permute(2, 0, 1, 3).contiguous().view(h * b, n, p)
        keys = self.project_keys(keys).view(b, m, h, p).permute(2, 0, 1, 3).contiguous().view(h * b, m, p)
        values = self.project_values(values).view(b, m, h, p).permute(2, 0, 1, 3).contiguous().view(h * b, m, p)

        output = self.attention(queries, keys, values)
        output = output.view(h, b, n, p).permute(1, 2, 0, 3).contiguous().view(b, n, d)
        return self.concatenation(output)


class RFF(nn.Module):
    """Row-wise 3-layer MLP feed-forward block."""

    def __init__(self, d: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(d, d),
            nn.ReLU(inplace=True),
            nn.Linear(d, d),
            nn.ReLU(inplace=True),
            nn.Linear(d, d),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class MAB(nn.Module):
    """Multihead Attention Block."""

    def __init__(self, d: int, h: int, rff: RFF) -> None:
        super().__init__()
        self.multihead = MultiheadAttention(d, h)
        self.layer_norm1 = nn.LayerNorm(d)
        self.layer_norm2 = nn.LayerNorm(d)
        self.rff = rff

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        h = self.layer_norm1(x + self.multihead(x, y, y))
        return self.layer_norm2(h + self.rff(h))


class SAB(nn.Module):
    """Set Attention Block."""

    def __init__(self, d: int, h: int, rff: RFF) -> None:
        super().__init__()
        self.mab = MAB(d, h, rff)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mab(x, x)


class ISAB(nn.Module):
    """Induced Set Attention Block."""

    def __init__(self, d: int, m: int, h: int, rff1: RFF, rff2: RFF) -> None:
        super().__init__()
        self.mab1 = MAB(d, h, rff1)
        self.mab2 = MAB(d, h, rff2)
        self.inducing_points = nn.Parameter(torch.randn(1, m, d))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b = x.size(0)
        p = self.inducing_points.repeat([b, 1, 1])
        h = self.mab1(p, x)
        return self.mab2(x, h)


class PMA(nn.Module):
    """Pooling by Multihead Attention."""

    def __init__(self, d: int, k: int, h: int, rff: RFF) -> None:
        super().__init__()
        self.mab = MAB(d, h, rff)
        self.seed_vectors = nn.Parameter(torch.randn(1, k, d))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        b = z.size(0)
        s = self.seed_vectors.repeat([b, 1, 1])
        return self.mab(s, z)


class SetTransformer(nn.Module):
    """Equivariant or invariant SetTransformer encoder."""

    def __init__(self, in_dim: int, out_dim: int, embed_type: str = "equivariant"):
        super().__init__()
        d, m, h, k = 128, 16, 4, 4
        self.embed_type = embed_type
        self.embed = nn.Sequential(nn.Linear(in_dim, d), nn.ReLU(inplace=True))
        self.encoder = nn.Sequential(ISAB(d, m, h, RFF(d), RFF(d)), ISAB(d, m, h, RFF(d), RFF(d)))
        if self.embed_type.startswith("i"):
            self.decoder = nn.Sequential(PMA(d, k, h, RFF(d)), SAB(d, h, RFF(d)))
            self.predictor = nn.Linear(k * d, out_dim)
        else:
            self.predictor = nn.Linear(d, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embed(x)
        x_enc = self.encoder(x)
        if self.embed_type.startswith("i"):
            x_dec = self.decoder(x_enc)
            b, k, d = x_dec.shape
            return self.predictor(x_dec.view(b, k * d))
        return self.predictor(x_enc)


# -----------------------------------------------------------------------------
# 2. Compressed Dynamic Sparse CRF & Exact Fast Forward-Backward Marginals
# -----------------------------------------------------------------------------

def build_edge_compression_maps(
    n_players: int = 22, n_outsides: int = 4
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compress N*N edge states (26*26=676) to valid graph edges (22*26 + 4 = 576)."""
    n_nodes = n_players + n_outsides
    valid_orig_edge_ids = []
    comp_src = []
    comp_dst = []

    for s in range(n_players):
        for d in range(n_nodes):
            valid_orig_edge_ids.append(s * n_nodes + d)
            comp_src.append(s)
            comp_dst.append(d)

    for o in range(n_players, n_nodes):
        valid_orig_edge_ids.append(o * n_nodes + o)
        comp_src.append(o)
        comp_dst.append(o)

    valid_t = torch.tensor(valid_orig_edge_ids, dtype=torch.long)
    src_t = torch.tensor(comp_src, dtype=torch.long)
    dst_t = torch.tensor(comp_dst, dtype=torch.long)
    orig2comp = torch.full((n_nodes * n_nodes,), -1, dtype=torch.long)
    orig2comp[valid_t] = torch.arange(len(valid_orig_edge_ids), dtype=torch.long)
    return valid_t, orig2comp, src_t, dst_t


class DynamicSparseCRF(nn.Module):
    """Sparse linear-chain CRF over K=576 possession edges with exact Forward-Backward."""

    def __init__(
        self,
        comp_src: torch.Tensor,
        comp_dst: torch.Tensor,
        edge_embed_dim: int = 16,
        team_size: int = 11,
        forbid_score: float = -1e4,
    ):
        super().__init__()
        self.K = comp_src.numel()
        self.comp_src = comp_src
        self.comp_dst = comp_dst
        self.edge_embed_dim = edge_embed_dim
        self.team_size = team_size
        self.forbid_score = forbid_score

        self.linear = nn.Linear(edge_embed_dim * 2, 1)

        inc_idx, inc_type, inc_mask = self._build_incoming()
        self.register_buffer("inc_idx", inc_idx)
        self.register_buffer("inc_type", inc_type)
        self.register_buffer("inc_mask", inc_mask)
        self._out_cache = None

    def _same_team(self, a: int, b: int) -> bool:
        ts = self.team_size
        if a >= 2 * ts or b >= 2 * ts:
            return False
        return (a < ts and b < ts) or (a >= ts and b >= ts)

    def _edge_type(self, sender: int, receiver: int) -> str:
        if receiver == sender:
            return "self"
        if receiver >= 2 * self.team_size:
            return "out"
        return "teammate" if self._same_team(sender, receiver) else "opponent"

    def _build_allowed_next(self) -> List[List[Tuple[int, int]]]:
        K = self.K
        src = self.comp_src.tolist()
        dst = self.comp_dst.tolist()
        allowed_next = [[] for _ in range(K)]

        for i in range(K):
            src_idx = src[i]
            dst_idx = dst[i]

            if src_idx >= 2 * self.team_size and src_idx == dst_idx:
                allowed_next[i].append((i, 11))
                continue
            if src_idx >= 2 * self.team_size and src_idx != dst_idx:
                continue

            if src_idx == dst_idx:
                for j in range(K):
                    if src[j] != src_idx:
                        continue
                    rel = self._edge_type(src_idx, dst[j])
                    theta = {"self": 0, "teammate": 1, "opponent": 2, "out": 3}[rel]
                    allowed_next[i].append((j, theta))
                continue

            if dst_idx < 2 * self.team_size:
                allowed_next[i].append((i, 4))
                for j in range(K):
                    if src[j] != dst_idx:
                        continue
                    rel = self._edge_type(dst_idx, dst[j])
                    theta = {"self": 5, "teammate": 6, "opponent": 7, "out": 8}[rel]
                    allowed_next[i].append((j, theta))
            else:
                allowed_next[i].append((i, 9))
                for j in range(K):
                    if src[j] == dst_idx and dst[j] == dst_idx:
                        allowed_next[i].append((j, 10))
                        break
        return allowed_next

    def _build_incoming(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        allowed_next = self._build_allowed_next()
        K = self.K
        incoming = [[] for _ in range(K)]
        for i in range(K):
            for j, t in allowed_next[i]:
                incoming[j].append((i, t))

        M = max(len(v) for v in incoming) if K > 0 else 1
        inc_idx = torch.full((K, M), -1, dtype=torch.long)
        inc_type = torch.full((K, M), -1, dtype=torch.long)
        inc_mask = torch.zeros((K, M), dtype=torch.bool)

        for j in range(K):
            pairs = incoming[j]
            if not pairs:
                continue
            inc_idx[j, : len(pairs)] = torch.tensor([p[0] for p in pairs], dtype=torch.long)
            inc_type[j, : len(pairs)] = torch.tensor([p[1] for p in pairs], dtype=torch.long)
            inc_mask[j, : len(pairs)] = True
        return inc_idx, inc_type, inc_mask

    def _get_outgoing(self, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """Build padded outgoing adjacency (K, M_out) on demand without altering state_dict buffers."""
        if self._out_cache is None or self._out_cache[0].device != device:
            K = self.K
            outgoing = [[] for _ in range(K)]
            inc_i = self.inc_idx.cpu().numpy()
            inc_m = self.inc_mask.cpu().numpy()
            for j in range(K):
                for m_idx in range(inc_i.shape[1]):
                    if inc_m[j, m_idx]:
                        outgoing[int(inc_i[j, m_idx])].append(j)
            M_out = max(len(v) for v in outgoing) if K > 0 else 1
            out_idx = torch.zeros((K, M_out), dtype=torch.long, device=device)
            out_mask = torch.zeros((K, M_out), dtype=torch.bool, device=device)
            for i in range(K):
                js = outgoing[i]
                if js:
                    out_idx[i, : len(js)] = torch.tensor(js, dtype=torch.long, device=device)
                    out_mask[i, : len(js)] = True
            self._out_cache = (out_idx, out_mask)
        return self._out_cache

    def _precompute_uv(self, edge_embeds: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Factor exact affine transition score w @ [e_{t-1}, e_t] + b = u_{t-1} + v_t."""
        D = self.edge_embed_dim
        W = self.linear.weight
        b = self.linear.bias
        u = F.linear(edge_embeds[:, :-1], W[:, :D]).squeeze(-1)  # (B, T-1, K)
        v = F.linear(edge_embeds[:, 1:], W[:, D:], b).squeeze(-1)  # (B, T-1, K)
        return u, v

    @torch.no_grad()
    def marginal_posteriors(self, emissions: torch.Tensor, edge_embeds: torch.Tensor) -> torch.Tensor:
        """Exact Forward-Backward state marginals P(y_t = k | x) of shape (B, T, K)."""
        B, T, K = emissions.shape
        assert B == 1, "marginal_posteriors expects B=1"
        em = emissions[0]  # (T, K)
        if T == 1:
            return torch.softmax(em, dim=-1).unsqueeze(0)

        u_all, v_all = self._precompute_uv(edge_embeds)
        u = u_all[0]  # (T-1, K)
        v = v_all[0]  # (T-1, K)

        inc_idx = self.inc_idx.clamp(min=0)
        inc_mask = self.inc_mask
        out_idx, out_mask = self._get_outgoing(em.device)

        neg_inf = float("-inf")
        alpha = torch.empty((T, K), dtype=em.dtype, device=em.device)
        alpha[0] = em[0]
        for t in range(1, T):
            prev_u = alpha[t - 1] + u[t - 1]
            cand = torch.where(inc_mask, prev_u[inc_idx] + v[t - 1].unsqueeze(1), neg_inf)
            alpha[t] = torch.logsumexp(cand, dim=1) + em[t]

        beta = torch.zeros((T, K), dtype=em.dtype, device=em.device)
        for t in range(T - 2, -1, -1):
            next_v = beta[t + 1] + em[t + 1] + v[t]
            cand = torch.where(out_mask, next_v[out_idx], neg_inf)
            beta[t] = torch.logsumexp(cand, dim=1) + u[t]

        return torch.softmax(alpha + beta, dim=-1).unsqueeze(0)

    @torch.no_grad()
    def decode(self, emissions: torch.Tensor, edge_embeds: torch.Tensor) -> torch.Tensor:
        """Fast exact Viterbi decode over compressed K=576 edges (B=1)."""
        B, T, K = emissions.shape
        assert B == 1
        em = emissions[0]
        u_all, v_all = self._precompute_uv(edge_embeds)
        u, v = u_all[0], v_all[0]

        inc_idx = self.inc_idx.clamp(min=0)
        inc_mask = self.inc_mask
        neg_inf = float("-inf")

        delta = em[0]
        back = torch.empty((max(T - 1, 0), K), dtype=torch.long, device=em.device)
        for t in range(1, T):
            prev_u = delta + u[t - 1]
            cand = torch.where(inc_mask, prev_u[inc_idx] + v[t - 1].unsqueeze(1), neg_inf)
            best_val, best_m = cand.max(dim=1)
            delta = best_val + em[t]
            back[t - 1] = best_m

        path = torch.empty(T, dtype=torch.long, device=em.device)
        st = int(delta.argmax())
        path[T - 1] = st
        for t in range(T - 1, 0, -1):
            st = int(inc_idx[st, int(back[t - 1, st])])
            path[t - 1] = st
        return path.unsqueeze(0)


# -----------------------------------------------------------------------------
# 3. Complete SetLSTM Model (exact architecture for round3/final checkpoint)
# -----------------------------------------------------------------------------

class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding."""

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)[:, 0:-1] if d_model % 2 != 0 else torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0).transpose(0, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(x + self.pe[: x.size(0), :])


class SetLSTM(nn.Module):
    """Hierarchical SetTransformer + Temporal SAB + DynamicSparseCRF for ball-free possession."""

    def __init__(
        self,
        macro_type: Optional[str] = "poss_edge",
        micro_type: str = "poss_edge",
        seq_model_type: str = "sab",
        crf_model_type: str = "dynamic_sparse_crf",
        feat_dim: int = 8,
        team_size: int = 11,
        macro_ppe_dim: int = 16,
        macro_fpe_dim: int = 16,
        micro_enc_dim: int = 64,
        macro_seq_dim: int = 128,
        micro_seq_dim: int = 128,
        crf_edge_dim: int = 16,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.macro_type = macro_type
        self.micro_type = micro_type
        self.seq_model_type = seq_model_type
        self.crf_model_type = crf_model_type
        self.feat_dim = feat_dim
        self.team_size = team_size
        self.n_nodes = team_size * 2 + 4

        self.register_buffer("pitch_size", torch.tensor([PITCH_X, PITCH_Y], dtype=torch.float32))

        self.macro_out_dim = 2
        self.macro_mask_dim = 2
        self.micro_out_dim = 2

        macro_z_dim = feat_dim + self.macro_mask_dim + macro_ppe_dim + macro_fpe_dim
        self.macro_team1_enc = SetTransformer(feat_dim + self.macro_mask_dim, macro_ppe_dim, embed_type="e")
        self.macro_team2_enc = SetTransformer(feat_dim + self.macro_mask_dim, macro_ppe_dim, embed_type="e")
        self.macro_outside_enc = nn.Linear(feat_dim + self.macro_mask_dim, macro_ppe_dim)
        self.macro_fpe_enc = SetTransformer(feat_dim + self.macro_mask_dim, macro_fpe_dim, embed_type="e")

        self.macro_seq_proj = nn.Sequential(nn.Linear(macro_z_dim, macro_seq_dim), nn.ReLU())
        self.macro_pe = PositionalEncoding(macro_seq_dim, dropout)
        self.macro_sab_t1 = SAB(macro_seq_dim, 4, RFF(macro_seq_dim))
        self.macro_sab_t2 = SAB(macro_seq_dim, 4, RFF(macro_seq_dim))
        self.macro_head = nn.Sequential(nn.Linear(macro_seq_dim, 2 * self.macro_out_dim), nn.GLU())

        micro_in_dim = feat_dim + macro_seq_dim + self.macro_out_dim
        self.micro_team1_enc = SetTransformer(micro_in_dim, micro_enc_dim, embed_type="e")
        self.micro_team2_enc = SetTransformer(micro_in_dim, micro_enc_dim, embed_type="e")
        self.micro_outside_enc = nn.Sequential(nn.Linear(micro_in_dim, micro_enc_dim), nn.ReLU())
        self.micro_fpe_enc = SetTransformer(micro_in_dim, micro_enc_dim, embed_type="e")
        self.micro_node_proj = nn.Sequential(nn.Linear(micro_enc_dim * 2, micro_enc_dim), nn.ReLU())

        self.micro_seq_proj = nn.Sequential(nn.Linear(micro_enc_dim + self.micro_out_dim, micro_seq_dim), nn.ReLU())
        self.micro_pe = PositionalEncoding(micro_seq_dim, dropout)
        self.micro_sab_t1 = SAB(micro_seq_dim, 4, RFF(micro_seq_dim))
        self.micro_sab_t2 = SAB(micro_seq_dim, 4, RFF(micro_seq_dim))

        self.micro_edge_proj = nn.Sequential(nn.Linear(micro_seq_dim * 2, micro_seq_dim), nn.ReLU())
        self.micro_edge_head = nn.Sequential(nn.Linear(micro_seq_dim, 2), nn.GLU())

        valid_orig_edge_ids, orig2comp, comp_src, comp_dst = build_edge_compression_maps(self.team_size * 2, 4)
        self.register_buffer("valid_orig_edge_ids", valid_orig_edge_ids)
        self.register_buffer("orig2comp", orig2comp)
        self.register_buffer("comp_src", comp_src)
        self.register_buffer("comp_dst", comp_dst)

        self.crf_edge_proj = nn.Sequential(nn.Linear(micro_seq_dim, crf_edge_dim), nn.ReLU())
        self.crf_edge_norm = nn.LayerNorm(crf_edge_dim)
        self.crf = DynamicSparseCRF(comp_src, comp_dst, crf_edge_dim, team_size=self.team_size)

    def _temporal_sab(self, x: torch.Tensor, block: SAB, pos_encoder: Optional[PositionalEncoding]) -> torch.Tensor:
        if pos_encoder is not None:
            x = pos_encoder(x)
        return block(x.transpose(0, 1)).transpose(0, 1)

    def forward_compressed(self, x_in: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run inference directly on K=576 compressed edges: (B, T, N, 8) -> macro, micro_comp, edge_comp."""
        x_in = x_in.transpose(0, 1)  # (T, B, N, F)
        T, B, N, feat_dim = x_in.shape
        ts = self.team_size
        masked_macro = torch.zeros(T * B, N, self.macro_mask_dim, device=x_in.device, dtype=x_in.dtype)
        masked_micro = torch.zeros(T, B * N, self.micro_out_dim, device=x_in.device, dtype=x_in.dtype)

        t1_x = x_in[:, :, :ts, :].reshape(T * B, ts, feat_dim)
        t2_x = x_in[:, :, ts : 2 * ts, :].reshape(T * B, ts, feat_dim)
        out_x = x_in[:, :, -4:, :].reshape(T * B, 4, feat_dim)

        t1_exp = torch.cat([t1_x, masked_macro[:, :ts]], -1)
        t2_exp = torch.cat([t2_x, masked_macro[:, ts:-4]], -1)
        out_exp = torch.cat([out_x, masked_macro[:, -4:]], -1)
        all_exp = torch.cat([t1_exp, t2_exp, out_exp], 1)

        macro_ppe = torch.cat([self.macro_team1_enc(t1_exp), self.macro_team2_enc(t2_exp), self.macro_outside_enc(out_exp)], 1)
        macro_z = torch.cat([all_exp, macro_ppe, self.macro_fpe_enc(all_exp)], -1).reshape(T, B * N, -1)
        macro_z = self.macro_seq_proj(macro_z)
        macro_h = self._temporal_sab(macro_z, self.macro_sab_t1, self.macro_pe)
        macro_h = self._temporal_sab(macro_h, self.macro_sab_t2, None).reshape(T * B, N, -1)
        macro_out = self.macro_head(macro_h)

        t1_in = torch.cat([t1_x, macro_h[:, :ts], macro_out[:, :ts]], -1)
        t2_in = torch.cat([t2_x, macro_h[:, ts:-4], macro_out[:, ts:-4]], -1)
        out_in = torch.cat([out_x, macro_h[:, -4:], macro_out[:, -4:]], -1)

        micro_ppe = torch.cat([self.micro_team1_enc(t1_in), self.micro_team2_enc(t2_in), self.micro_outside_enc(out_in)], 1)
        micro_fpe = self.micro_fpe_enc(torch.cat([t1_in, t2_in, out_in], 1))
        micro_z = self.micro_node_proj(torch.cat([micro_ppe, micro_fpe], -1)).reshape(T, B * N, -1)
        micro_z = self.micro_seq_proj(torch.cat([micro_z, masked_micro], -1))

        micro_h = self._temporal_sab(micro_z, self.micro_sab_t1, self.micro_pe)
        micro_h = self._temporal_sab(micro_h, self.micro_sab_t2, None).reshape(T * B, N, -1)

        # Factor micro_edge_proj nn.Linear(2D, D) across N=26 nodes instead of K=576 edges (exact up to float rounding)
        edge_lin = self.micro_edge_proj[0]
        h_dim = micro_h.size(-1)
        hs = F.linear(micro_h, edge_lin.weight[:, :h_dim], edge_lin.bias)
        hd = F.linear(micro_h, edge_lin.weight[:, h_dim:])
        edge_h = F.relu(hs[:, self.comp_src] + hd[:, self.comp_dst])
        micro_comp = self.micro_edge_head(edge_h).squeeze(-1).reshape(T, B, -1).transpose(0, 1)
        edge_comp = self.crf_edge_norm(self.crf_edge_proj(edge_h.reshape(T, B, -1, edge_h.size(-1)))).transpose(0, 1)
        macro_ret = macro_out.reshape(T, B, N, self.macro_out_dim).transpose(0, 1)
        return macro_ret, micro_comp, edge_comp


# -----------------------------------------------------------------------------
# 4. Loading & Tracking -> Ball-Free PathCRF Window Inference
# -----------------------------------------------------------------------------

def load_nova_pathcrf_model(ckpt_path: str = DEFAULT_CKPT_PATH, args_path: str = DEFAULT_ARGS_PATH) -> SetLSTM:
    """Load frozen SkillCorner fine-tuned SetLSTM + DynamicSparseCRF weights strictly."""
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Missing Nova PathCRF weights: {ckpt_path}")
    kwargs = {}
    if os.path.isfile(args_path):
        with open(args_path, encoding="utf-8") as f:
            args = json.load(f)
        kwargs = dict(
            macro_type=args.get("macro_type", "poss_edge"),
            micro_type=args.get("micro_type", "poss_edge"),
            seq_model_type=args.get("seq_model", "sab"),
            crf_model_type=args.get("crf_model", "dynamic_sparse_crf"),
            feat_dim=args.get("node_in_dim", 8),
            team_size=args.get("team_size", 11),
            macro_ppe_dim=args.get("coarse_dim", 16),
            macro_fpe_dim=args.get("coarse_dim", 16),
            micro_enc_dim=args.get("fine_dim", 64),
            macro_seq_dim=args.get("seq_dim", 128),
            micro_seq_dim=args.get("seq_dim", 128),
            crf_edge_dim=args.get("crf_edge_dim", 16),
            dropout=args.get("dropout", 0.1),
        )
    model = SetLSTM(**kwargs)
    state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    if state and next(iter(state)).startswith("module."):
        state = {k[7:]: v for k, v in state.items()}
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


def _savgol_slot_kinematics(px_m: np.ndarray, py_m: np.ndarray, fps: float = FPS) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute SavGol velocities/accelerations inside continuous runs without slot-swap artifacts."""
    T, K = px_m.shape
    vx = np.full((T, K), np.nan, dtype=np.float32)
    vy = np.full((T, K), np.nan, dtype=np.float32)
    sp = np.full((T, K), np.nan, dtype=np.float32)
    ac = np.full((T, K), np.nan, dtype=np.float32)

    for k in range(K):
        x, y = px_m[:, k], py_m[:, k]
        fin = np.isfinite(x) & np.isfinite(y)
        if fin.sum() < 2:
            continue
        # Break segments whenever missing OR step speed exceeds human cap
        step_ok = np.zeros(T, dtype=bool)
        both = fin[:-1] & fin[1:]
        dist = np.hypot(np.where(both, x[1:] - x[:-1], 0.0), np.where(both, y[1:] - y[:-1], 0.0))
        step_ok[1:] = both & (dist * fps <= MAX_PLAYER_SPEED)
        idx = np.flatnonzero(fin)
        seg_starts = [idx[0]]
        seg_ends = []
        for ii in range(1, len(idx)):
            if idx[ii] != idx[ii - 1] + 1 or not step_ok[idx[ii]]:
                seg_ends.append(idx[ii - 1] + 1)
                seg_starts.append(idx[ii])
        seg_ends.append(idx[-1] + 1)

        for a, b in zip(seg_starts, seg_ends):
            n = b - a
            if n < 2:
                vx[a:b, k] = 0.0
                vy[a:b, k] = 0.0
                sp[a:b, k] = 0.0
                ac[a:b, k] = 0.0
                continue
            dx = np.diff(x[a:b]) * fps
            dy = np.diff(y[a:b]) * fps
            if len(dx) >= 15:
                svx = savgol_filter(dx, window_length=15, polyorder=2)
                svy = savgol_filter(dy, window_length=15, polyorder=2)
            elif len(dx) >= 5:
                w = len(dx) if len(dx) % 2 == 1 else len(dx) - 1
                svx = savgol_filter(dx, window_length=w, polyorder=2)
                svy = savgol_filter(dy, window_length=w, polyorder=2)
            else:
                svx, svy = dx, dy
            svs = np.hypot(svx, svy)
            vx[a + 1 : b, k] = svx
            vy[a + 1 : b, k] = svy
            sp[a + 1 : b, k] = svs
            vx[a, k], vy[a, k], sp[a, k] = svx[0], svy[0], svs[0]

            if len(svx) >= 2:
                dax = np.diff(svx) * fps
                day = np.diff(svy) * fps
                if len(dax) >= 9:
                    ax = savgol_filter(dax, window_length=9, polyorder=2)
                    ay = savgol_filter(day, window_length=9, polyorder=2)
                elif len(dax) >= 5:
                    w = len(dax) if len(dax) % 2 == 1 else len(dax) - 1
                    ax = savgol_filter(dax, window_length=w, polyorder=2)
                    ay = savgol_filter(day, window_length=w, polyorder=2)
                else:
                    ax, ay = dax, day
                ac[a + 1 : b - 1, k] = np.hypot(ax, ay)
            ac[a, k] = 0.0
            ac[b - 1, k] = 0.0
    return vx, vy, sp, ac


def _fill_1d_window(arr: np.ndarray, t_grid: np.ndarray) -> np.ndarray:
    """Linearly interpolate with boundary clamping for a 1-D window series; zero if all NaN."""
    fin = np.isfinite(arr)
    if fin.all():
        return arr.astype(np.float32)
    if not fin.any():
        return np.zeros_like(arr, dtype=np.float32)
    return np.interp(t_grid, t_grid[fin], arr[fin]).astype(np.float32)


def _pack_half_windows(
    feats_down: np.ndarray,
    slot_teams: np.ndarray,
    home_is_left: bool,
    window_len_ds: int = 50,
    stride_ds: int = 25,
) -> Tuple[np.ndarray, List[Tuple[int, int]]]:
    """Pack overlapping 10s (50-step @ 5fps) windows into (n_win, T_w, 26, 8) left-first tensors."""
    T_ds, _, _ = feats_down.shape
    starts = list(range(0, max(1, T_ds - window_len_ds + 1), stride_ds))
    if T_ds > window_len_ds and starts[-1] + window_len_ds < T_ds:
        starts.append(T_ds - window_len_ds)

    left_code = 0 if home_is_left else 1
    right_code = 1 - left_code
    left_all = np.flatnonzero(slot_teams == left_code)
    right_all = np.flatnonzero(slot_teams == right_code)

    outside_block = np.zeros((window_len_ds, 4, 8), dtype=np.float32)
    outside_xy = ((0.0, PITCH_Y / 2.0), (PITCH_X, PITCH_Y / 2.0), (PITCH_X / 2.0, 0.0), (PITCH_X / 2.0, PITCH_Y))
    for i_o, (ox, oy) in enumerate(outside_xy):
        outside_block[:, i_o, 2] = ox
        outside_block[:, i_o, 3] = oy

    n_win = len(starts)
    batch = np.zeros((n_win, window_len_ds, 26, 8), dtype=np.float32)
    t_grid = np.arange(window_len_ds, dtype=np.float32)
    windows = []

    for wi, s0 in enumerate(starts):
        e0 = min(T_ds, s0 + window_len_ds)
        valid_len = e0 - s0
        windows.append((s0, e0))
        batch[wi, :, 22:26, :] = outside_block

        w_sub = feats_down[s0:e0]  # (valid_len, S, 6)
        w_tg = t_grid[:valid_len]

        for node_base, t_idx, team_ind, is_left in ((0, left_all, -1.0, True), (11, right_all, 1.0, False)):
            if len(t_idx) == 0:
                continue
            vis_cnt = np.sum(np.isfinite(w_sub[:, t_idx, 0]), axis=0)
            # Pick top-11 most visible slots in this window deterministically
            order = np.argsort(-vis_cnt, kind="stable")[:11]
            chosen = [t_idx[o] for o in order if vis_cnt[o] > 0]
            if not chosen:
                continue
            mean_xs = [float(np.nanmean(w_sub[:, c, 0])) for c in chosen]
            gk_local = int(np.argmin(mean_xs) if is_left else np.argmax(mean_xs))

            for k_i, slot in enumerate(chosen):
                nd = node_base + k_i
                batch[wi, :valid_len, nd, 0] = team_ind
                batch[wi, :valid_len, nd, 1] = 1.0 if k_i == gk_local else 0.0
                for c_f in range(6):
                    batch[wi, :valid_len, nd, 2 + c_f] = _fill_1d_window(w_sub[:, slot, c_f], w_tg)
                if valid_len < window_len_ds:
                    batch[wi, valid_len:, nd, :] = batch[wi, valid_len - 1 : valid_len, nd, :]

    return batch, windows


def _window_weights(windows: List[Tuple[int, int]], idx: int) -> np.ndarray:
    start, end = windows[idx]
    length = end - start
    weights = np.ones(length, dtype=np.float32)
    if idx > 0:
        ov = min(windows[idx - 1][1] - start, length)
        if ov > 0:
            weights[:ov] = np.linspace(0.0, 1.0, ov, dtype=np.float32)
    if idx < len(windows) - 1:
        ov = min(end - windows[idx + 1][0], length)
        if ov > 0:
            weights[-ov:] = np.linspace(1.0, 0.0, ov, dtype=np.float32)
    return weights


def _apply_switch_delay(p_home: np.ndarray, delay_frames: int) -> np.ndarray:
    """Shift live possession probability forward by `delay_frames`."""
    if delay_frames <= 0 or len(p_home) <= delay_frames:
        return p_home.copy()
    out = np.empty_like(p_home)
    out[:delay_frames] = p_home[0]
    out[delay_frames:] = p_home[:-delay_frames]
    return out


def apply_switch_delay_per_half(
    p_home: np.ndarray, blocks: List[Tuple[int, int, int]], switch_delay_s: float
) -> np.ndarray:
    """Shift `p_home` forward per half block without crossing halftime gaps."""
    delay_frames = int(round(switch_delay_s * FPS))
    if delay_frames <= 0:
        return np.asarray(p_home, dtype=np.float32)
    out = np.array(p_home, dtype=np.float32, copy=True)
    for _, start, last in blocks:
        stop = last + 1
        out[start:stop] = _apply_switch_delay(out[start:stop], delay_frames)
    return out


def select_pathcrf_p_home(
    probs: Dict[str, np.ndarray],
    blocks: List[Tuple[int, int, int]],
    readout: str = "crf_src",
    switch_delay_s: float = 1.0,
) -> np.ndarray:
    """Extract and optionally shift the configured P(home | live) curve from base signals."""
    if readout == "crf_src":
        raw = probs["p_home_crf_src"]
    elif readout == "crf_mid":
        raw = probs["p_home_crf_mid"]
    elif readout == "macro":
        raw = probs["p_home_macro"]
    elif readout == "crf_macro_50":
        raw = 0.5 * probs["p_home_crf_src"] + 0.5 * probs["p_home_macro"]
    else:
        raise ValueError(f"Unknown PathCRF readout: {readout}")
    return apply_switch_delay_per_half(raw, blocks, switch_delay_s)


def predict_nova_pathcrf(
    t_data: dict,
    blocks: List[Tuple[int, int, int]],
    model: Optional[SetLSTM] = None,
    window_seconds: float = 10.0,
    stride_seconds: float = 10.0,
    batch_size: int = 64,
    switch_delay_s: float = 0.0,
    readout: str = "crf_src",
    n_threads: Optional[int] = None,
    verbose: bool = False,
) -> Dict[str, np.ndarray]:
    """Run deterministic ball-free Nova PathCRF inference on normalized [0, 1] tracking data.

    Returns 25 fps per-frame probabilities matching t_data['player_x'] length:
    - p_home: selected & shifted P(home | live) in [0, 1]
    - p_live: 1 - P(outside self-loop CRF marginals) in [0, 1]
    - p_live_macro: un-absorbed 1 - P(outside macro softmax) in [0, 1]
    - p_home_crf_src, p_home_crf_mid, p_home_macro: unshifted/shifted variants for analysis
    """
    t0 = time.time()
    if model is None:
        model = load_nova_pathcrf_model()

    px_norm = np.asarray(t_data["player_x"], dtype=np.float32)
    py_norm = np.asarray(t_data["player_y"], dtype=np.float32)
    slot_teams = np.asarray(t_data["team_of_slot"])
    n_frames = len(px_norm)

    sample_freq = 5
    win_ds = int(round(window_seconds * FPS / sample_freq))
    str_ds = int(round(stride_seconds * FPS / sample_freq))
    delay_frames = int(round(switch_delay_s * FPS))

    comp_src_np = model.comp_src.cpu().numpy()
    comp_dst_np = model.comp_dst.cpu().numpy()
    left_src_m = (comp_src_np < 11).astype(np.float32)
    right_src_m = ((comp_src_np >= 11) & (comp_src_np < 22)).astype(np.float32)
    left_dst_m = (comp_dst_np < 11).astype(np.float32)
    right_dst_m = ((comp_dst_np >= 11) & (comp_dst_np < 22)).astype(np.float32)
    out_loop_m = (comp_src_np >= 22).astype(np.float32)

    out_dict = {
        "p_home": np.full(n_frames, 0.5, dtype=np.float32),
        "p_live": np.full(n_frames, 0.5, dtype=np.float32),
        "p_live_macro": np.full(n_frames, 0.5, dtype=np.float32),
        "p_home_crf_src": np.full(n_frames, 0.5, dtype=np.float32),
        "p_home_crf_mid": np.full(n_frames, 0.5, dtype=np.float32),
        "p_home_macro": np.full(n_frames, 0.5, dtype=np.float32),
    }

    prev_threads = torch.get_num_threads()
    use_threads = min(8, os.cpu_count() or 4) if n_threads is None else int(n_threads)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(max(1, use_threads))
    try:
        home_mask = slot_teams == 0
        away_mask = slot_teams == 1

        for _, start, last in blocks:
            stop = last + 1
            T_half = stop - start
            if T_half <= 2:
                continue

            px_m = px_norm[start:stop] * PITCH_X
            py_m = py_norm[start:stop] * PITCH_Y
            vx, vy, sp, ac = _savgol_slot_kinematics(px_m, py_m, fps=FPS)
            feats = np.stack([px_m, py_m, vx, vy, sp, ac], axis=-1)  # (T_half, S, 6)

            home_mx = float(np.nanmean(px_m[:, home_mask])) if home_mask.any() else 52.5
            away_mx = float(np.nanmean(px_m[:, away_mask])) if away_mask.any() else 52.5
            home_is_left = home_mx < away_mx

            down_idx = np.arange(0, T_half, sample_freq, dtype=np.int64)
            if len(down_idx) == 0 or down_idx[-1] != T_half - 1:
                down_idx = np.append(down_idx, T_half - 1)
            T_ds = len(down_idx)

            win_batch_np, windows = _pack_half_windows(
                feats[down_idx], slot_teams, home_is_left, window_len_ds=win_ds, stride_ds=str_ds
            )

            macro_acc = np.zeros((T_ds, 26, 2), dtype=np.float32)
            micro_acc = np.zeros((T_ds, 576), dtype=np.float32)
            edge_acc = np.zeros((T_ds, 576, 16), dtype=np.float32)
            w_sum = np.zeros(T_ds, dtype=np.float32)

            with torch.no_grad():
                for b0 in range(0, len(win_batch_np), batch_size):
                    xb = torch.from_numpy(win_batch_np[b0 : b0 + batch_size])
                    mac_b, mic_b, edg_b = model.forward_compressed(xb)
                    mac_np = mac_b.numpy()
                    mic_np = mic_b.numpy()
                    edg_np = edg_b.numpy()
                    for k_loc in range(len(mac_np)):
                        wi = b0 + k_loc
                        s0, e0 = windows[wi]
                        ln = e0 - s0
                        wt = _window_weights(windows, wi)
                        macro_acc[s0:e0] += mac_np[k_loc, :ln] * wt[:, None, None]
                        micro_acc[s0:e0] += mic_np[k_loc, :ln] * wt[:, None]
                        edge_acc[s0:e0] += edg_np[k_loc, :ln] * wt[:, None, None]
                        w_sum[s0:e0] += wt

            den = np.where(w_sum > 0, w_sum, 1.0)
            macro_avg = macro_acc / den[:, None, None]
            micro_avg = micro_acc / den[:, None]
            edge_avg = edge_acc / den[:, None, None]

            torch.set_num_threads(min(2, use_threads))
            try:
                with torch.no_grad():
                    post = model.crf.marginal_posteriors(
                        torch.from_numpy(micro_avg).unsqueeze(0),
                        torch.from_numpy(edge_avg).unsqueeze(0),
                    )[0].numpy()
            finally:
                torch.set_num_threads(use_threads)

            # Compute Left vs Right probabilities in 5 fps, then convert to Home vs Away
            eps = 1e-6
            p_out_ds = np.clip(post @ out_loop_m, 0.0, 1.0)
            p_live_ds = np.clip(1.0 - p_out_ds, 1e-4, 1.0 - 1e-4)

            p_l_src = post @ left_src_m
            p_r_src = post @ right_src_m
            p_left_src_ds = np.where((p_l_src + p_r_src) > eps, p_l_src / np.maximum(p_l_src + p_r_src, eps), 0.5)

            p_l_dst = post @ left_dst_m
            p_r_dst = post @ right_dst_m
            w_l = 0.5 * (p_l_src + p_l_dst)
            w_r = 0.5 * (p_r_src + p_r_dst)
            p_left_mid_ds = np.where((w_l + w_r) > eps, w_l / np.maximum(w_l + w_r, eps), 0.5)

            mac_probs = torch.softmax(torch.from_numpy(macro_avg[:, :, 0]), dim=-1).numpy()
            mac_l = mac_probs[:, :11].sum(axis=1)
            mac_r = mac_probs[:, 11:22].sum(axis=1)
            mac_live_ds = np.clip(mac_l + mac_r, 1e-4, 1.0 - 1e-4)
            p_left_mac_ds = np.where((mac_l + mac_r) > eps, mac_l / np.maximum(mac_l + mac_r, eps), 0.5)

            p_h_src_ds = p_left_src_ds if home_is_left else (1.0 - p_left_src_ds)
            p_h_mid_ds = p_left_mid_ds if home_is_left else (1.0 - p_left_mid_ds)
            p_h_mac_ds = p_left_mac_ds if home_is_left else (1.0 - p_left_mac_ds)

            # Linearly upsample from 5 fps back to full 25 fps half
            full_t = np.arange(T_half, dtype=np.float32)
            dn_t = down_idx.astype(np.float32)
            p_live_25 = np.clip(np.interp(full_t, dn_t, p_live_ds), 1e-4, 1.0 - 1e-4).astype(np.float32)
            p_lmac_25 = np.clip(np.interp(full_t, dn_t, mac_live_ds), 1e-4, 1.0 - 1e-4).astype(np.float32)
            p_src_25 = _apply_switch_delay(np.clip(np.interp(full_t, dn_t, p_h_src_ds), 1e-4, 1.0 - 1e-4).astype(np.float32), delay_frames)
            p_mid_25 = _apply_switch_delay(np.clip(np.interp(full_t, dn_t, p_h_mid_ds), 1e-4, 1.0 - 1e-4).astype(np.float32), delay_frames)
            p_mac_25 = _apply_switch_delay(np.clip(np.interp(full_t, dn_t, p_h_mac_ds), 1e-4, 1.0 - 1e-4).astype(np.float32), delay_frames)

            out_dict["p_live"][start:stop] = p_live_25
            out_dict["p_live_macro"][start:stop] = p_lmac_25
            out_dict["p_home_crf_src"][start:stop] = p_src_25
            out_dict["p_home_crf_mid"][start:stop] = p_mid_25
            out_dict["p_home_macro"][start:stop] = p_mac_25
            out_dict["p_home"][start:stop] = select_pathcrf_p_home(
                out_dict, [(0, start, last)], readout=readout, switch_delay_s=0.0
            )[start:stop]

    finally:
        torch.set_num_threads(prev_threads)

    if verbose:
        print(f"  [nova_pathcrf] inference finished in {time.time() - t0:.2f}s (threads={use_threads})", flush=True)
    return out_dict


def build_nova_pathcrf_signals(
    t_data: dict,
    blocks: List[Tuple[int, int, int]],
    game: str = "BAR-ATH",
    data_dir: Optional[str] = None,
    model: Optional[SetLSTM] = None,
    stride_seconds: float = 10.0,
    force: bool = False,
) -> Dict[str, np.ndarray]:
    """Compute or load cached unshifted Nova PathCRF curves on normalized tracking data."""
    from games import DATA_DIR

    root = data_dir or DATA_DIR
    cache_dir = os.path.join(root, "cache")
    cache_path = os.path.join(cache_dir, f"{game}_nova_pathcrf_v1_s{int(round(stride_seconds))}.npz")
    n_frames = len(t_data["player_x"])
    keys = ("p_live", "p_live_macro", "p_home_crf_src", "p_home_crf_mid", "p_home_macro")

    if not force and os.path.isfile(cache_path):
        with np.load(cache_path, allow_pickle=False) as z:
            if all(k in z for k in keys) and len(z["p_home_crf_src"]) == n_frames:
                return {k: z[k].astype(np.float32) for k in keys}

    os.makedirs(cache_dir, exist_ok=True)
    res = predict_nova_pathcrf(
        t_data,
        blocks,
        model=model,
        stride_seconds=stride_seconds,
        switch_delay_s=0.0,
        readout="crf_src",
    )
    payload = {k: res[k].astype(np.float32) for k in keys}
    tmp_path = f"{cache_path}.tmp.{os.getpid()}.npz"
    np.savez_compressed(tmp_path, **payload)
    os.replace(tmp_path, cache_path)
    return payload
