import os

import torch
from torch_geometric.data import InMemoryDataset
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import degree

from graph_attention_networks.gat_networks import GAT
from graph_conv_networks.gcn_networks import GCN
from graph_u_networks.graph_u_networks import GraphUNet
from pna_graph_networks.pna_networks import PNA


def get_model(model_name: str):
    if model_name == "GCN":
        return GCN
    elif model_name == "GAT":
        return GAT
    elif model_name == "GraphUNet":
        return GraphUNet
    elif model_name == "PNA":
        return PNA
    else:
        raise NotImplementedError(f"No graph network with name '{model_name}' implemented!")


def get_dataset(dataset_name):
    if dataset_name == "Planetoid":
        return Planetoid(root=os.path.join("datasets", "Cora"), name="Cora")
    else:
        raise NotImplementedError(f"No dataset with name '{dataset_name}' available!")


def compute_in_degree_histogram(dataset: InMemoryDataset) -> torch.Tensor:
    degree_histogram = None
    for data in dataset:
        in_degrees = degree(data.edge_index[1], num_nodes=data.num_nodes, dtype=torch.long)
        max_in_degree = torch.max(in_degrees).item()
        degree_histogram = torch.bincount(in_degrees, minlength=max_in_degree + 1)

    return degree_histogram
