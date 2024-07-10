import torch
import torch.nn.functional as F
import GCL.augmentors as A
from GCL.models import DualBranchContrast
from torch_geometric.data import DataLoader
from torch_geometric.datasets import TUDataset
from torch import nn
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_add_pool
import GCL.losses as L
from torch.optim import Adam


dataset = TUDataset(root='data/TUDataset', name='MUTAG')
loader = DataLoader(dataset, batch_size=32, shuffle=True)


class GCN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GCNConv(dataset.num_features, 64)
        self.conv2 = GCNConv(64, 64)

    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = global_add_pool(x, batch)
        return x

class Encoder(nn.Module):
    def __init__(self, encoder, augmentor1, augmentor2):
        super().__init__()
        self.encoder = encoder
        self.aug1 = augmentor1
        self.aug2 = augmentor2

    def forward(self, x, edge_index, batch):
        x1, edge_index1, _ = self.aug1(x, edge_index)
        x2, edge_index2, _ = self.aug2(x, edge_index)
        z1 = self.encoder(x1, edge_index1, batch)
        z2 = self.encoder(x2, edge_index2, batch)
        return z1, z2


augmentor1 = A.EdgeRemoving(pe=0.1)
augmentor2 = A.FeatureMasking(pf=0.1)

device = torch.device('cuda')
gcn = GCN().to(device)
encoder = Encoder(gcn, augmentor1, augmentor2).to(device)
contrast_model = DualBranchContrast(loss=L.InfoNCE(tau=0.2), mode='G2G').to(device)
optimizer = Adam(encoder.parameters(), lr=0.01)


def train(epoch):
    encoder.train()
    training_loss = 0
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        g1, g2 = encoder(data.x, data.edge_index, data.batch)
        loss = contrast_model(g1=g1, g2=g2)
        loss.backward()
        optimizer.step()
        training_loss += loss.item()
    avg_loss = training_loss / len(loader)
    print(f"Loss: {avg_loss:.4f}")
    return avg_loss


for epoch in range(1, 30):
    print(f"Epoch {epoch}:\n")
    train(epoch)
    print("-------------------------------------------")








