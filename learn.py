import torch
import torch.nn.functional as F
import GCL.augmentors as A
from GCL.eval import get_split, SVMEvaluator
from GCL.models import DualBranchContrast
from torch_geometric.data import DataLoader
from torch_geometric.datasets import TUDataset
from torch import nn
from torch_geometric.nn import GCNConv, BatchNorm
from torch_geometric.nn import global_add_pool
import GCL.losses as L
from torch.optim import Adam


dataset = TUDataset(root='data/TUDataset', name='MUTAG')
loader = DataLoader(dataset, batch_size=128, shuffle=True)



class GCN(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.convs = nn.ModuleList()
        self.batchNorms = nn.ModuleList()

        for i in range(3):
            if i == 0:
                self.convs.append(GCNConv(in_channel, out_channel))
            else:
                self.convs.append(GCNConv(out_channel, out_channel))
            self.batchNorms.append(BatchNorm(out_channel))

        self.project = nn.Sequential(
            nn.Linear(out_channel, out_channel),
            nn.ReLU(),
            nn.Linear(out_channel, out_channel)
        )

    def forward(self, x, edge_index, batch):
        for conv, bn in zip(self.convs, self.batchNorms):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)

        x = global_add_pool(x, batch)
        x = self.project(x)

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
gcn = GCN(dataset.num_features, 32).to(device)
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

def test():
    encoder.eval()
    x = []
    y = []
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            g = encoder.encoder(data.x, data.edge_index, data.batch)
            x.append(g)
            y.append(data.y)
    x = torch.cat(x, dim=0)
    y = torch.cat(y, dim=0)

    split = get_split(num_samples=x.size()[0], train_ratio=0.8, test_ratio=0.1)
    result = SVMEvaluator(linear=True)(x, y, split)
    return result


for epoch in range(1, 100):
    print(f"Epoch {epoch}:\n")
    train(epoch)
    print("-------------------------------------------")

test_result = test()
print(f'(E): Best test F1Mi={test_result["micro_f1"]:.4f}, F1Ma={test_result["macro_f1"]:.4f}')







