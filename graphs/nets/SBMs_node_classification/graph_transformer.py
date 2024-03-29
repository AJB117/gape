import torch
import torch.nn as nn

from layers.pe_layer import PELayer

"""
    Graph Transformer without edge features
    
"""
from layers.graph_transformer_layer import GraphTransformerLayer
from layers.mlp_readout_layer import MLPReadout

class GraphTransformerNet(nn.Module):
    def __init__(self, net_params):
        super().__init__()
        hidden_dim = net_params['hidden_dim']
        n_classes = net_params['n_classes']
        num_heads = net_params['n_heads']
        out_dim = net_params['out_dim']
        in_dim = net_params['in_dim']
        in_feat_dropout = net_params['in_feat_dropout']
        dropout = net_params['dropout']
        n_layers = net_params['L']
        self.readout = net_params['readout']
        self.layer_norm = net_params['layer_norm']
        self.batch_norm = net_params['batch_norm']
        self.residual = net_params['residual']
        self.device = net_params['device']
        self.wl_pos_enc = net_params['wl_pos_enc']
        self.pe_layer = PELayer(net_params)
        self.cat = net_params['cat_gape']
        self.n_classes = n_classes

        if self.cat:
            self.embedding_h = nn.Embedding(in_dim, hidden_dim) # node feat is an integer
        else:
            self.embedding_h = nn.Embedding(in_dim, hidden_dim) # node feat is an integer
        
        self.in_feat_dropout = nn.Dropout(in_feat_dropout)
        
        self.layers = nn.ModuleList([ GraphTransformerLayer(hidden_dim, hidden_dim, num_heads, dropout,
                                                    self.layer_norm, self.batch_norm, self.residual) for _ in range(n_layers-1) ]) 
        self.layers.append(GraphTransformerLayer(hidden_dim, out_dim, num_heads, dropout, self.layer_norm, self.batch_norm, self.residual))
        self.MLP_layer = MLPReadout(out_dim, n_classes)

        if self.cat:
            self.ll = nn.Linear(hidden_dim + net_params['pos_enc_dim'], hidden_dim)
        
    def forward(self, g, h, e, pos_enc=None, h_wl_pos_enc=None):
        h = self.embedding_h(h)
        if self.cat:
            pe = self.pe_layer(g, h, pos_enc)
            h = torch.cat((h, pe), dim=1)
            h = self.ll(h)
            h = self.in_feat_dropout(h)
        else:
            h = self.in_feat_dropout(h)
            pe = self.pe_layer(g, h, pos_enc)
            h = h + pe

        # convnets
        for conv in self.layers:
            h = conv(g, h)

        out = self.MLP_layer(h)
        return out

    def loss(self, pred, label):
        # calculating label weights for weighted loss computation
        V = label.size(0)
        label_count = torch.bincount(label)
        label_count = label_count[label_count.nonzero()].squeeze()
        cluster_sizes = torch.zeros(self.n_classes).long().to(self.device)
        cluster_sizes[torch.unique(label)] = label_count
        weight = (V - cluster_sizes).float() / V
        weight *= (cluster_sizes>0).float()
        
        # weighted cross-entropy for unbalanced classes
        criterion = nn.CrossEntropyLoss(weight=weight)
        loss = criterion(pred, label)

        return loss
