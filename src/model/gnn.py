import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometrics.nn import GATConv

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

    def forward(self, x:torch.Tenso) -> torch.Tensor:
        for layer in self.layers[:-1]:
            x = F.relu(self.layer(x))
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
    ):
        super().__init__()
        if num_layers < 2:
            raise ValueError(f"num_layers must be >=2, got {num_layers}")
        self.operator = operator
        self.dropout = dropout
        self.layer_energies: list[float] = []
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        # First layer: in_channels -> hidden_channels (edge_dim=in_channels enables edge_attr)
        self.convs.append(
            GATConv(in_channels, hidden_channels, heads= num_heads, concat=False, edge_dim=in_channels))
        self.bns.append(nn.BatchNorm1d(hidden_channels))

        # Intermediate layers
        for _ in range(num_layers - 2):
            self.convs.append(
                GATConv(hidden_channels, hidden_channels, heads=num_heads,
                concat=False, edge_dim=in_channels)
            )
            self.bns.append(nn.BatchNorm1d(hidden_channels))

        # Final output layer (no BatchNorm/ReLU/Dropout after final conv)
        self.convs.append(
            GATConv(hidden_channels, out_channels, heads=num_heads,
            concat=False, edge_dim=in_channels)
        )

        #Dual-role soft pruning MLPs (node-level and edge-level)
        self.mlp_x = MLP(in_channels, hidden_channels, mlp_layers)
        self.mlp_edge_attr = MLP(in_channels, hidden_channels, mlp_layers)

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()
        self.mlp_x.apply(self.__reset_weights)
        self.mlp_edge_attr.apply(self.__reset_weights)
