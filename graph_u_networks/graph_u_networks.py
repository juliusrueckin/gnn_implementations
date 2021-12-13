import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, TopKPooling

import constants as const


class GraphUNet(torch.nn.Module):
    def __init__(self, num_node_features: int, num_classes: int):
        super().__init__()

        self.depth = const.DEPTH
        self.enc_convs = torch.nn.ModuleList()
        self.dec_convs = torch.nn.ModuleList()
        self.pools = torch.nn.ModuleList()

        self.enc_convs.append(GCNConv(num_node_features, const.NUM_CHANNELS, improved=True))
        for _ in range(self.depth):
            self.pools.append(TopKPooling(const.NUM_CHANNELS, const.TOP_K_POOLING))
            self.enc_convs.append(GCNConv(const.NUM_CHANNELS, const.NUM_CHANNELS, improved=True))

        for _ in range(self.depth - 1):
            self.dec_convs.append(GCNConv(const.NUM_CHANNELS, const.NUM_CHANNELS, improved=True))
        self.dec_convs.append(GCNConv(const.NUM_CHANNELS, num_classes, improved=True))

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = F.relu(self.enc_convs[0](x, edge_index))
        xs = [x]
        edge_indices = [edge_index]
        perms = []

        for i in range(1, self.depth + 1):
            x, edge_index, _, _, perm, _ = self.pools[i - 1](x, edge_index)
            x = F.relu(self.enc_convs[i](x, edge_index))
            x = F.dropout(x, training=self.training, p=const.DROPOUT)

            perms += [perm]
            if i < self.depth:
                xs += [x]
                edge_indices += [edge_index]

        for i in range(self.depth):
            j = self.depth - i - 1
            res, edge_index, perm = xs[j], edge_indices[j], perms[j]

            up = torch.zeros_like(res)
            up[perm] = x
            x = res + up

            x = self.dec_convs[i](x, edge_index)
            x = F.relu(x) if i < self.depth - 1 else x
            x = F.dropout(x, training=self.training, p=const.DROPOUT) if i < self.depth - 1 else x

        return F.log_softmax(x, dim=1)
