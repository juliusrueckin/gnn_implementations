import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GATv2Conv

import constants as const


def get_gat_conv_layer(layer_version: int):
    if layer_version == 1:
        return GATConv
    elif layer_version == 2:
        return GATv2Conv
    else:
        raise NotImplementedError(f"No version '{layer_version}' of GATConv layer available!")


class GAT(torch.nn.Module):
    def __init__(self, num_node_features: int, num_classes: int, version: int):
        super().__init__()
        self.conv1 = get_gat_conv_layer(version)(
            num_node_features, const.NUM_CHANNELS, heads=const.NUM_HEADS, dropout=const.DROPOUT, concat=True
        )
        self.conv2 = get_gat_conv_layer(version)(
            const.NUM_CHANNELS * const.NUM_HEADS, num_classes, heads=1, concat=False, dropout=const.DROPOUT
        )

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, training=self.training, p=const.DROPOUT)
        x = self.conv2(x, edge_index)

        return F.log_softmax(x, dim=1)
