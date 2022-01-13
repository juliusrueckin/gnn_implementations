import torch
import torch.nn.functional as F
from torch_geometric.nn import PNAConv, BatchNorm

import constants as const


class PNA(torch.nn.Module):
    def __init__(self, num_node_features: int, num_classes: int, deg: torch.Tensor):
        super().__init__()

        self.conv_layers = torch.nn.ModuleList()
        self.batch_norm_layers = torch.nn.ModuleList()

        self.conv_layers.append(
            PNAConv(
                num_node_features,
                const.NUM_CHANNELS,
                const.AGGREGATORS,
                const.SCALERS,
                deg,
                towers=const.NUM_TOWERS,
                pre_layers=const.PRE_LAYERS,
                post_layers=const.POST_LAYERS,
            )
        )
        self.batch_norm_layers.append(BatchNorm(const.NUM_CHANNELS))

        for _ in range(const.DEPTH - 1):
            self.conv_layers.append(
                PNAConv(
                    const.NUM_CHANNELS,
                    const.NUM_CHANNELS,
                    const.AGGREGATORS,
                    const.SCALERS,
                    deg,
                    towers=const.NUM_TOWERS,
                    pre_layers=const.PRE_LAYERS,
                    post_layers=const.POST_LAYERS,
                )
            )
            self.batch_norm_layers.append(BatchNorm(const.NUM_CHANNELS))

        self.conv_layers.append(
            PNAConv(
                const.NUM_CHANNELS,
                num_classes,
                const.AGGREGATORS,
                const.SCALERS,
                deg,
                towers=1,
                pre_layers=const.PRE_LAYERS,
                post_layers=const.POST_LAYERS,
            )
        )

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        for i in range(const.DEPTH - 1):
            x = F.relu(self.batch_norm_layers[i](self.conv_layers[i](x, edge_index, edge_attr=edge_attr)))
            x = F.dropout(x, training=self.training, p=const.DROPOUT)

        x = self.conv_layers[-1](x, edge_index, edge_attr=edge_attr)

        return F.log_softmax(x, dim=1)
