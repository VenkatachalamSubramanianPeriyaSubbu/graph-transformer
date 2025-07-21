import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool
from torch_geometric.utils import to_dense_adj

import scipy.sparse as sp
import numpy as np

def laplacian_position_encoding(edge_index, num_nodes, pos_enc_dim=32):
    edge_index = edge_index.cpu()
    adj = to_dense_adj(edge_index, max_num_nodes=num_nodes)[0].numpy()
    lap = sp.csgraph.laplacian(adj, normed=True)
    eigvals, eigvecs = np.linalg.eigh(lap)
    idx = np.argsort(eigvals)[1:pos_enc_dim + 1]  # Skip the zero eigenvalue
    pos_enc = eigvecs[:, idx]  # [N, pos_enc_dim]
    pos_enc = torch.tensor(pos_enc, dtype=torch.float32)
    return pos_enc

class GraphAttentionLayer(nn.Module):
    def __init__(self, in_channels, out_channels, heads, dropout=0.5, alpha=0.2):
        super().__init__()
        self.heads = heads
        self.dim_head = out_channels // heads
        self.scale = self.dim_head ** -0.5
        self.q_proj = nn.Linear(in_channels, out_channels, bias=False)
        self.k_proj = nn.Linear(in_channels, out_channels, bias=False)
        self.v_proj = nn.Linear(in_channels, out_channels, bias=False)
        self.attn_drop = nn.Dropout(dropout)
        self.out_proj = nn.Linear(out_channels, out_channels, bias=False)
        self.alpha = alpha

    def forward(self, x, edge_index):
        N, H, D = x.size(0), self.heads, self.dim_head
        q = self.q_proj(x).view(N, H, D).transpose(0, 1)
        k = self.k_proj(x).view(N, H, D).transpose(0, 1)
        v = self.v_proj(x).view(N, H, D).transpose(0, 1)

        row, col = edge_index.to(x.device)

        attn = (q[:, row] * k[:, col]).sum(dim=-1) * self.scale
        attn = F.leaky_relu(attn, negative_slope=self.alpha)
        attn_weights = F.softmax(attn, dim=0)
        attn_weights = self.attn_drop(attn_weights)

        out = attn_weights.unsqueeze(-1) * v[:, col]
        out_nodes = torch.zeros(H, N, D, device=x.device, dtype=x.dtype)
        out_nodes.scatter_add_(1, row.unsqueeze(0).unsqueeze(-1).expand(H, -1, D), out)
        out = out_nodes.transpose(0, 1).contiguous().view(N, -1)
        return self.out_proj(out)

class MetaPathLearner(nn.Module):
    def __init__(self, num_edge_types, num_channels):
        super().__init__()
        self.weights = nn.Parameter(torch.randn(num_channels, num_edge_types))

    def forward(self, edge_index_list, num_nodes):
        A_list = []
        for ei in edge_index_list:
            if ei.ndim != 2 or ei.shape[0] != 2:
                raise ValueError(f"Expected edge_index of shape [2, num_edges], got {ei.shape}")
            ei = ei.long()
            A_dense = to_dense_adj(ei, max_num_nodes=num_nodes)[0]  # shape: [N, N]
            A_list.append(A_dense)
        
        A_stack = torch.stack(A_list, dim=0)  # shape: [E, N, N]
        soft_weights = F.softmax(self.weights, dim=-1)  # shape: [C, E]
        A_meta = torch.einsum("ce,enm->cnm", soft_weights, A_stack) 
        return A_meta, soft_weights

class GraphTransformerLayer(nn.Module):
    def __init__(self, input_features, heads=4, dropout=0.1):
        super().__init__()
        self.attn = GraphAttentionLayer(input_features, input_features, heads, dropout)
        self.norm1 = nn.LayerNorm(input_features)
        self.norm2 = nn.LayerNorm(input_features)
        self.ffn = nn.Sequential(
            nn.Linear(input_features, input_features * 4),
            nn.ReLU(),
            nn.Linear(input_features * 4, input_features)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index):
        residual = x
        x = self.attn(x, edge_index)
        x = residual + self.dropout(x)
        x = self.norm1(x)

        residual = x
        x = self.ffn(x)
        x = residual + self.dropout(x)
        x = self.norm2(x)
        return x

class GraphTransformer(nn.Module):
    def __init__(self, input_features, hidden_features, output_features,
                 num_edge_types=3, num_metachannels=10,
                 num_layers=6, heads=4, dropout=0.1, pos_enc_dim=32):
        super().__init__()
        self.pos_enc_dim = pos_enc_dim
        self.metapath = MetaPathLearner(num_edge_types, num_metachannels)
        self.input_proj = nn.Linear(input_features + pos_enc_dim, hidden_features)
        self.layers = nn.ModuleList([
            GraphTransformerLayer(hidden_features, heads, dropout) for _ in range(num_layers)
        ])
        self.final_layer = nn.Sequential(
            nn.Linear(hidden_features * num_metachannels, hidden_features),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_features, output_features)
        )

    def forward(self, x, edge_index_list):
        num_nodes = x.size(0)
        device = x.device

        # Meta-path adjacency matrices
        A_meta, soft_weights = self.metapath(edge_index_list, num_nodes)

        # Laplacian Positional Encoding
        pos_enc = laplacian_position_encoding(edge_index_list[0], num_nodes, self.pos_enc_dim).to(device)
        x = torch.cat([x, pos_enc], dim=-1)
        x = self.input_proj(x)

        # Apply transformer layers on each meta-path
        outputs = []
        for c in range(A_meta.size(0)):
            adj = A_meta[c]
            ei = (adj > 0).nonzero(as_tuple=False).t().contiguous().to(device)

            x_c = x
            for layer in self.layers:
                x_c = layer(x_c, ei)
            outputs.append(x_c)

        # Concatenate outputs across meta-path channels
        out = torch.cat(outputs, dim=-1)

        # Final MLP
        out = self.final_layer(out)
        return out, A_meta, soft_weights