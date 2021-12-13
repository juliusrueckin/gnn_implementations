import torch
import torch.nn.functional as F

import constants as const
from graph_network_wrappers.utils import get_model, get_dataset


class GraphNetworkWrapper:
    def __init__(self, model_name: str, dataset_name: str):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dataset = get_dataset(dataset_name)
        self.data = self.dataset[0].to(device)
        self.model = get_model(model_name)(self.dataset.num_node_features, self.dataset.num_classes).to(device)

    def train(self):
        self.model = self.model
        optimizer = torch.optim.Adam(self.model.parameters(), lr=const.LEARNING_RATE, weight_decay=const.WEIGHT_DECAY)

        self.model.train()
        for epoch in range(const.NUM_EPOCHS):
            optimizer.zero_grad()
            out = self.model(self.data)
            loss = F.nll_loss(out[self.data.train_mask], self.data.y[self.data.train_mask])
            loss.backward()
            optimizer.step()

    def evaluate(self):
        self.model.eval()
        predictions = self.model(self.data).argmax(dim=1)
        correct = (predictions[self.data.test_mask] == self.data.y[self.data.test_mask]).sum()
        accuracy = int(correct) / int(self.data.test_mask.sum())
        return accuracy
