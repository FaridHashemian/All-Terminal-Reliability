import torch
from torch.nn import Embedding, Linear, ModuleList, ReLU, Sequential,Dropout, Sigmoid
from torch_geometric.utils import degree
from torch_geometric.nn import BatchNorm, PNAConv, global_add_pool
import torch.nn.functional as F



def comp_deg(train_dataset):
    max_degree = -1
    for data in train_dataset:
        d = degree(data.edge_index[1], num_nodes=data.num_nodes, dtype=torch.long)
        max_degree = max(max_degree, int(d.max()))

    # Compute the in-degree histogram tensor
    deg = torch.zeros(max_degree + 1, dtype=torch.long)
    for data in train_dataset:
        d = degree(data.edge_index[1], num_nodes=data.num_nodes, dtype=torch.long)
        deg += torch.bincount(d, minlength=deg.numel()) 
    return deg



class PNA_Net(torch.nn.Module):
    def __init__(self, train_dataset):
        super().__init__()

        self.node_emb = Embedding(128, 128)
        self.edge_emb = Embedding(128, 128)

        aggregators = ['mean', 'min', 'max', 'std']
        #aggregators = ['sum']
        scalers = ['identity', 'amplification', 'attenuation']

        self.convs = ModuleList()
        self.batch_norms = ModuleList()
        for _ in range(4):
            conv = PNAConv(in_channels=128, out_channels=128,
                           aggregators=aggregators, scalers=scalers, deg=comp_deg(train_dataset),
                           edge_dim=128, towers=8, pre_layers=1, post_layers=1,
                           divide_input=False)
            self.convs.append(conv)
            self.batch_norms.append(BatchNorm(128))

        self.mlp = Sequential(Linear(128, 64), ReLU(), Linear(64, 32), ReLU(),
                              Linear(32, 1), Sigmoid())

    def forward(self, x, edge_index, edge_attr, batch):
        x = self.node_emb(x.squeeze())
        edge_attr = self.edge_emb(edge_attr)


        for conv, batch_norm in zip(self.convs, self.batch_norms):
            x = F.relu(batch_norm(conv(x, edge_index, edge_attr)))

        x = global_add_pool(x, batch)
        return self.mlp(x), x
#model = Net()
#print(model)