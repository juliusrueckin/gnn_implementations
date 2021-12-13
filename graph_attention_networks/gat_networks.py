import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv


class GAT(torch.nn.Module):
    def __init__(self, num_node_features: int, num_classes: int):
        super().__init__()
        self.conv1 = GATConv(num_node_features, 16, heads=8, dropout=0.2, concat=True)
        self.conv2 = GATConv(16 * 8, num_classes, heads=1, concat=False, dropout=0.2)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, training=self.training, p=0.2)
        x = self.conv2(x, edge_index)

        return F.log_softmax(x, dim=1)
