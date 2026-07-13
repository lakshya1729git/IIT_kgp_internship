"""
model.py — Temporal HAN for traffic disruption scoring (V3)
============================================================
V3 changes (matching graph_builder.py V3):
  - LOCATION_FEAT_DIM: 3 → 5  (meaningful location features now)
  - Input projection for location updated automatically via import
  - No other architecture changes needed — feature dims drive everything

Outputs:
  road_disruption_prob  : (N_road,)      ∈ [0,1]
  event_confidence_adj  : (N_event,)     ∈ [0,1]
  event_severity_logits : (N_event, 3)   raw logits → softmax → class probs
  attention_weights     : dict of tensors for explainability
"""

from __future__ import annotations

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False

# ── Feature dimensions (must match graph_builder.py) ─────────────────────────
from hgnn.graph_builder import (
    EVENT_FEAT_DIM, ROAD_FEAT_DIM, SOURCE_FEAT_DIM, LOCATION_FEAT_DIM
)

N_SEVERITY_CLASSES = 3   # low / medium / high

HIDDEN_DIM        = 64
N_ATTENTION_HEADS = 4
N_HAN_LAYERS      = 2
DROPOUT           = 0.3


def _check_torch():
    if not _TORCH_AVAILABLE:
        raise ImportError(
            "PyTorch is required for the HGNN model.\n"
            "pip install torch"
        )


# ── Semantic Attention ────────────────────────────────────────────────────────

class _SemanticAttention(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        _check_torch()
        self.proj = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, z_list: list) -> tuple:
        """Returns (fused_embedding, attention_weights)."""
        stacked = torch.stack(z_list, dim=1)            # (N, P, H)
        attn    = self.proj(torch.tanh(stacked))         # (N, P, 1)
        attn    = F.softmax(attn, dim=1)                 # (N, P, 1)
        out     = (stacked * attn).sum(dim=1)            # (N, H)
        return out, attn.squeeze(-1)                     # also return weights


# ── HAN Convolution ───────────────────────────────────────────────────────────

class _HANConv(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, n_heads: int, dropout: float):
        super().__init__()
        _check_torch()
        assert out_dim % n_heads == 0
        self.n_heads  = n_heads
        self.head_dim = out_dim // n_heads
        self.dropout  = dropout

        self.W_src = nn.Linear(in_dim, out_dim, bias=False)
        self.W_dst = nn.Linear(in_dim, out_dim, bias=False)
        self.att   = nn.Parameter(torch.Tensor(1, n_heads, 2 * self.head_dim))
        nn.init.xavier_uniform_(self.att.unsqueeze(0))
        self.bias  = nn.Parameter(torch.zeros(out_dim))

    def forward(self, x_src, x_dst, edge_index):
        if edge_index.shape[1] == 0:
            return torch.zeros(x_dst.shape[0], self.W_dst.out_features,
                               device=x_dst.device)

        src_idx = edge_index[0]
        dst_idx = edge_index[1]

        h_src = self.W_src(x_src).view(-1, self.n_heads, self.head_dim)
        h_dst = self.W_dst(x_dst).view(-1, self.n_heads, self.head_dim)

        h_src_e = h_src[src_idx]
        h_dst_e = h_dst[dst_idx]

        alpha = torch.cat([h_src_e, h_dst_e], dim=-1)
        alpha = (alpha * self.att).sum(dim=-1)
        alpha = F.leaky_relu(alpha, negative_slope=0.2)
        alpha = self._softmax_per_dst(alpha, dst_idx, x_dst.shape[0])
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        msg = h_src_e * alpha.unsqueeze(-1)
        out = torch.zeros(x_dst.shape[0], self.n_heads, self.head_dim,
                          device=x_dst.device)
        out.scatter_add_(0,
                         dst_idx.unsqueeze(-1).unsqueeze(-1).expand_as(msg),
                         msg)
        return out.view(x_dst.shape[0], -1) + self.bias

    @staticmethod
    def _softmax_per_dst(alpha, dst_idx, n_dst):
        alpha_max = torch.zeros(n_dst, alpha.shape[1], device=alpha.device)
        alpha_max.scatter_reduce_(0, dst_idx.unsqueeze(1).expand_as(alpha),
                                  alpha, reduce="amax", include_self=True)
        alpha_exp = torch.exp(alpha - alpha_max[dst_idx])
        alpha_sum = torch.zeros(n_dst, alpha.shape[1], device=alpha.device)
        alpha_sum.scatter_add_(0, dst_idx.unsqueeze(1).expand_as(alpha_exp),
                               alpha_exp)
        return alpha_exp / (alpha_sum[dst_idx] + 1e-8)


# ── Temporal Heterogeneous Attention Network ──────────────────────────────────

class TrafficHGNN(nn.Module):
    """
    Temporal HAN with three output heads:
      1. road_disruption_prob     — regression  ∈ [0,1]
      2. event_confidence_adj     — regression  ∈ [0,1]
      3. event_severity_logits    — classification (3 classes)

    Also returns attention_weights dict for explainability.
    """

    def __init__(self, hidden_dim=HIDDEN_DIM, n_heads=N_ATTENTION_HEADS,
                 n_layers=N_HAN_LAYERS, dropout=DROPOUT):
        super().__init__()
        _check_torch()

        self.hidden_dim = hidden_dim
        self.n_layers   = n_layers

        # Input projections
        self.proj_road     = nn.Linear(ROAD_FEAT_DIM,     hidden_dim)
        self.proj_event    = nn.Linear(EVENT_FEAT_DIM,    hidden_dim)
        self.proj_source   = nn.Linear(SOURCE_FEAT_DIM,   hidden_dim)
        self.proj_location = nn.Linear(LOCATION_FEAT_DIM, hidden_dim)

        # HAN conv layers per relation
        self.layers_ev_road   = nn.ModuleList([_HANConv(hidden_dim, hidden_dim, n_heads, dropout) for _ in range(n_layers)])
        self.layers_road_ev   = nn.ModuleList([_HANConv(hidden_dim, hidden_dim, n_heads, dropout) for _ in range(n_layers)])
        self.layers_ev_src    = nn.ModuleList([_HANConv(hidden_dim, hidden_dim, n_heads, dropout) for _ in range(n_layers)])
        self.layers_ev_loc    = nn.ModuleList([_HANConv(hidden_dim, hidden_dim, n_heads, dropout) for _ in range(n_layers)])
        self.layers_road_road = nn.ModuleList([_HANConv(hidden_dim, hidden_dim, n_heads, dropout) for _ in range(n_layers)])

        # Semantic attention (stores weights for explainability)
        self.sem_attn_road  = _SemanticAttention(hidden_dim)
        self.sem_attn_event = _SemanticAttention(hidden_dim)

        # Output heads
        self.road_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1), nn.Sigmoid(),
        )
        self.event_conf_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1), nn.Sigmoid(),
        )
        # NEW: severity classification head (3 classes)
        self.event_sev_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, N_SEVERITY_CLASSES),
            # No activation — use CrossEntropyLoss which applies softmax internally
        )

        self.dropout_layer = nn.Dropout(dropout)
        self.norm_road     = nn.LayerNorm(hidden_dim)
        self.norm_event    = nn.LayerNorm(hidden_dim)

    def forward(self, road_x, event_x, source_x, location_x,
                edge_ev_affects_road, edge_ev_reported_by_src,
                edge_ev_located_at_loc, edge_road_near_road,
                return_attention: bool = False):
        """
        Args:
            return_attention: If True, also return attention weight dicts
                              (used for explainability).

        Returns:
            road_prob           : (N_road,)
            event_conf          : (N_event,)
            event_sev_logits    : (N_event, 3)
            attention_weights   : dict (only if return_attention=True)
        """
        h_road = F.elu(self.proj_road(road_x))
        h_ev   = F.elu(self.proj_event(event_x))
        h_src  = F.elu(self.proj_source(source_x))
        h_loc  = F.elu(self.proj_location(location_x))

        # Store attention weights from last layer
        last_road_attn  = None
        last_event_attn = None

        for layer_i in range(self.n_layers):
            h_road_from_ev   = self.layers_ev_road[layer_i](h_ev, h_road, edge_ev_affects_road)
            rev_edge         = edge_ev_affects_road[[1, 0]]
            h_ev_from_road   = self.layers_road_ev[layer_i](h_road, h_ev, rev_edge)
            h_ev_from_src    = self.layers_ev_src[layer_i](h_src, h_ev, edge_ev_reported_by_src[[1, 0]])
            h_ev_from_loc    = self.layers_ev_loc[layer_i](h_loc, h_ev, edge_ev_located_at_loc[[1, 0]])
            h_road_from_road = self.layers_road_road[layer_i](h_road, h_road, edge_road_near_road)

            h_road_new, road_attn  = self.sem_attn_road([h_road_from_ev, h_road_from_road])
            h_road_new = self.dropout_layer(F.elu(h_road_new))
            h_road     = self.norm_road(h_road + h_road_new)

            h_ev_new, ev_attn = self.sem_attn_event([h_ev_from_road, h_ev_from_src, h_ev_from_loc])
            h_ev_new   = self.dropout_layer(F.elu(h_ev_new))
            h_ev       = self.norm_event(h_ev + h_ev_new)

            # Save last layer attention for explainability
            last_road_attn  = road_attn
            last_event_attn = ev_attn

        road_prob        = self.road_head(h_road).squeeze(-1)
        event_conf       = self.event_conf_head(h_ev).squeeze(-1)
        event_sev_logits = self.event_sev_head(h_ev)               # (N, 3)

        if return_attention:
            attn_weights = {
                "road_meta_path_weights":  last_road_attn,   # (N_road,  2) [ev_path, road_path]
                "event_meta_path_weights": last_event_attn,  # (N_event, 3) [road, src, loc]
                "road_embeddings":  h_road,
                "event_embeddings": h_ev,
            }
            return road_prob, event_conf, event_sev_logits, attn_weights

        return road_prob, event_conf, event_sev_logits


def build_model(hidden_dim=HIDDEN_DIM, n_heads=N_ATTENTION_HEADS,
                n_layers=N_HAN_LAYERS, dropout=DROPOUT) -> "TrafficHGNN":
    _check_torch()
    return TrafficHGNN(hidden_dim=hidden_dim, n_heads=n_heads,
                       n_layers=n_layers, dropout=dropout)
