import os

from torch_geometric.datasets import Planetoid

from graph_attention_networks.gat_networks import GAT, GATv2
from graph_conv_networks.gcn_networks import GCN
from graph_u_networks.graph_u_networks import GraphUNet


def get_model(model_name: str):
    if model_name == "GCN":
        return GCN
    elif model_name == "GAT":
        return GAT
    elif model_name == "GATv2":
        return GATv2
    elif model_name == "GraphUNet":
        return GraphUNet
    else:
        raise NotImplementedError(f"No graph network with name '{model_name}' implemented!")


def get_dataset(dataset_name):
    if dataset_name == "Planetoid":
        return Planetoid(root=os.path.join("datasets", "Cora"), name="Cora")
    else:
        raise NotImplementedError(f"No dataset with name '{dataset_name}' available!")
