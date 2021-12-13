import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

import constants as const


class GCN(torch.nn.Module):
    def __init__(self, num_node_features: int, num_classes: int):
        super().__init__()
        self.conv1 = GCNConv(num_node_features, const.NUM_CHANNELS)
        self.conv2 = GCNConv(const.NUM_CHANNELS, num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, training=self.training, p=const.DROPOUT)
        x = self.conv2(x, edge_index)

        return F.log_softmax(x, dim=1)
