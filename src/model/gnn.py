import torch
from typing import Optional
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv

class MLP(nn.Module):
    """3-layer relevance scorer mapping element-wise distance vectors to scalar
    logits."""

    def __init__(self, input_dim:int, hidden_dim:int, num_layers:int):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(input_dim, 2 * hidden_dim))
        for _ in range(num_layers - 2):
            self.layers.append(nn.Linear(2 * hidden_dim, 2 * hidden_dim))
        self.layers.append(nn.Linear(2 * hidden_dim, 1))

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        for layer in self.layers[:-1]:
            x = F.relu(layer(x))
        return(self.layers[-1](x))

def compute_dirichlet_energy(x: torch.Tensor, edge_index: torch.Tensor) -> float:
    """Compute mean squared feature difference across connected graph edges."""
    if edge_index.numel() == 0:
        return 0.0
    with torch.no_grad():
        src, dst = edge_index[0], edge_index[1]
        energy = (x[src] - x[dst]).pow(2).sum(dim=-1).mean()
        return float(energy.item())

class GAT(nn.Module):
    """Query-conditioned soft-pruned GAT encoder (configurable depth L in {2,
        4})."""

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_layers: int = 4,
        dropout: float = 0.0,
        mlp_layers: int = 3,
        num_heads: int = 4,
        operator: str="euclidean",
        edge_dim: Optional[int] = None,
    ):
        super().__init__()
        if num_layers < 2:
            raise ValueError(f"num_layers must be >=2, got {num_layers}")
        self.operator = operator
        self.dropout = dropout
        self.layer_energies: list[float] = []
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.edge_dim = edge_dim

        # First layer: in_channels -> hidden_channels (edge_dim=in_channels enables edge_attr)
        self.convs.append(
            GATConv(in_channels, hidden_channels, heads= num_heads, concat=False, edge_dim=edge_dim))
        self.bns.append(nn.BatchNorm1d(hidden_channels))

        # Intermediate layers
        for _ in range(num_layers - 2):
            self.convs.append(
                GATConv(hidden_channels, hidden_channels, heads=num_heads,
                concat=False, edge_dim=edge_dim)
            )
            self.bns.append(nn.BatchNorm1d(hidden_channels))

        # Final output layer (no BatchNorm/ReLU/Dropout after final conv)
        self.convs.append(
            GATConv(hidden_channels, out_channels, heads=num_heads,
            concat=False, edge_dim=edge_dim)
        )

        #Dual-role soft pruning MLPs (node-level and edge-level)
        self.mlp_x = MLP(in_channels, hidden_channels, mlp_layers)
        edge_mlp_in = edge_dim if edge_dim is not None else in_channels
        self.mlp_edge_attr = MLP(edge_mlp_in, hidden_channels, mlp_layers)

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()
        self.mlp_x.apply(self._reset_weights)
        self.mlp_edge_attr.apply(self._reset_weights)

    @staticmethod
    def _reset_weights(m:nn.Module):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            nn.init.constant_(m.bias, 0)

    def _compute_distance(self, feat: torch.Tensor, query: torch.Tensor) -> torch.Tensor:
        if self.operator == "euclidean":
            return torch.sqrt((feat - query) ** 2)
        if self.operator == "manhattan":
            return torch.abs(feat - query)
        if self.operator == "element":
            return feat - query
        raise ValueError("operator must be one of 'euclidean', 'manhattan', or 'element'.")

    def forward(
        self, 
        x: torch.Tensor, 
        edge_index: torch.Tensor, 
        question_node: torch.Tensor, 
        edge_attr: torch.Tensor,
        question_edge: torch.Tensor
    )-> tuple[torch.Tensor, torch.Tensor]:
        # 1. Node soft pruning (out-of-place scaling preserves autograd safety)
        x_dist = self._compute_distance(x, question_node)
        x_weights = self.mlp_x(x_dist).sigmoid()
        x = x * x_weights

        # 2. Edge soft pruning
        has_edges = edge_attr is not None and torch.any(edge_attr)
        if has_edges:
            if edge_attr.dim() == 3:
                edge_attr = edge_attr.view(-1, edge_attr.size(2))
            edge_dist = self._compute_distance(edge_attr, question_edge)
            edge_weights = self.mlp_edge_attr(edge_dist).sigmoid()
            edge_weights = edge_weights.view(*edge_attr.shape[:-1], 1)
            edge_attr = edge_attr * edge_weights
            if edge_attr.dim()>2:
                edge_attr = edge_attr.view(2, -1, edge_attr.size(-1))

        # 3. Multi-layer GAT message passing + Dirichlet telemetry
        self.layer_energies = [compute_dirichlet_energy(x, edge_index)]
        conv_edge_attr = edge_attr if has_edges else None

        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index=edge_index, edge_attr=conv_edge_attr)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            self.layer_energies.append(compute_dirichlet_energy(x, edge_index))

        x = self.convs[-1](x, edge_index=edge_index, edge_attr=conv_edge_attr)
        self.layer_energies.append(compute_dirichlet_energy(x, edge_index))
        return(x, edge_attr)

load_gnn_model={
    "gat": GAT}


if __name__ == "__main__":
    from src.dataset.expla_graphs import ExplaGraphsDataset
    from src.utils.collate import collate_fn
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = ExplaGraphsDataset()
    batch = collate_fn([dataset[0], dataset[1]])
    g = batch["graph"].to(device)
    for edim in [None, 1024]:
        model = GAT(
            in_channels=1024,
            hidden_channels=1024,
            out_channels=1024,
            num_layers=4,
            edge_dim=edim,
        ).to(device)
        model.reset_parameters()
        out_x, _ = model(g.x, g.edge_index.long(), g.question_node, g.edge_attr,
        g.question_edge)
        out_x.sum().backward()
        x_grad = model.mlp_x.layers[0].weight.grad
        e_grad = model.mlp_edge_attr.layers[0].weight.grad
        x_norm = f"{x_grad.norm().item():.6f}" if x_grad is not None else "None"
        e_norm = f"{e_grad.norm().item():.6f}" if e_grad is not None else "None"
        print(f"\n=== GAT Smoke Test (device={device}, edge_dim={edim}) ===")
        print(f"Input x: {g.x.shape} -> Output x: {out_x.shape}")
        print(f"mlp_x grad norm:    {x_norm}")
        print(f"mlp_edge grad norm: {e_norm}")
        print(f"Dirichlet energies: {[round(e, 4) for e in model.layer_energies]}")