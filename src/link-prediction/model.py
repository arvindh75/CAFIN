import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv


class SAGE(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_layers):
        super().__init__()
        self.num_layers = num_layers
        self.convs = nn.ModuleList()
        self.l_relu = nn.LeakyReLU()
        self.relu = nn.ReLU()
        for i in range(num_layers):
            in_channels = in_channels if i == 0 else hidden_channels
            # Uncomment the below lines for normalization
            # if i != num_layers - 1:
            #     self.convs.append(SAGEConv(in_channels, hidden_channels))
            #     continue
            # self.convs.append(SAGEConv(in_channels, hidden_channels, normalize=True))
            self.convs.append(SAGEConv(in_channels, hidden_channels))

    def forward(self, x, adjs):
        for i, (edge_index, e_id, size) in enumerate(adjs):
            x_target = x[: size[1]]
            x = self.convs[i]((x, x_target), edge_index)
            if i != self.num_layers - 1:
                x = self.l_relu(x)
                x = F.dropout(x, p=0.5, training=self.training)
        return x

    def full_forward(self, x, edge_index):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i != self.num_layers - 1:
                x = self.l_relu(x)
                x = F.dropout(x, p=0.5, training=self.training)
        return x


if __name__ == "__main__":
    sampler = NeighborSampler(torch.tensor([(0, 1), (1, 2), (2, 3)]), sizes=[3, 2])
    model = SAGE(10, 100, 2)
