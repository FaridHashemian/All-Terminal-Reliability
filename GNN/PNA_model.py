import torch
from torch.nn import Embedding, Linear, ModuleList, ReLU, Sequential,Dropout, Sigmoid, Tanh, ELU, SiLU
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
    def __init__(self, args, train_dataset):
        super().__init__()

        self.node_emb = Embedding(args.n_embed, args.n_embed)
        self.edge_emb = Embedding(args.e_embed, args.e_embed)
        self.deg = comp_deg(train_dataset)

        
        aggregators = args.aggs
        #aggregators = ['sum']
        scalers = args.scalers

        self.convs = ModuleList()
        self.batch_norms = ModuleList()
        for _ in range(args.n_pna):
            conv = PNAConv(in_channels=args.n_embed, out_channels=args.n_embed,
                           aggregators=aggregators, scalers=scalers, deg=self.deg,
                           edge_dim=args.e_embed, towers=8, pre_layers=1, post_layers=1,
                           divide_input=False)
            self.convs.append(conv)
            self.batch_norms.append(BatchNorm(args.n_embed))

        self.mlp = ModuleList()
        self.mlp_batchnorm = ModuleList()
        self.mlp.append(Linear(args.n_embed, args.hidden))
        self.mlp_batchnorm.append(BatchNorm(args.hidden))
        if args.n_mlp_layer > 0:
            for _ in range(args.n_mlp_layer):
                linear = Linear(args.hidden, args.hidden)

                self.mlp.append(linear)
                self.mlp_batchnorm.append(BatchNorm(args.hidden))

        self.final_layer = Linear(args.hidden, 1)
        #self.mlp = Sequential(Linear(128, 64), ReLU(), Linear(64, 32), ReLU(),
        #                     Linear(32, 1), Sigmoid())
        self.sig = Sigmoid()
        if args.af == 'relu':
            self.af = ReLU()
        elif args.af == 'tanh':
            self.af = Tanh()
        elif args.af == 'elu':
            self.af = ELU()
        elif args.af == 'silu':
            self.af = SiLU()
            
    def forward(self, x, edge_index, edge_attr, batch):
        x = self.node_emb(x.squeeze())
        edge_attr = self.edge_emb(edge_attr)


        for conv, batch_norm in zip(self.convs, self.batch_norms):
            x = self.af(batch_norm(conv(x, edge_index, edge_attr)))

        x = global_add_pool(x, batch)

        for mlp, bn in zip (self.mlp, self.mlp_batchnorm):
            x = self.af(bn(mlp(x)))
        
        x = self.final_layer(x)
        return self.sig(x), x
#model = Net()
#print(model)