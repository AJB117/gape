import torch
import torch.nn as nn

import dgl
from layers.spectral_attention import SpectralAttention

"""
    Graph Transformer with node spectral attention PE
    
"""
# from layers.graph_transformer_edge_layer import GraphTransformerLayer
from layers.graph_transformer_layer import GraphTransformerLayer
from layers.mlp_readout_layer import MLPReadout

class SAGraphTransformerNet(nn.Module):
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

        # SAN specific
        lpe_layers = net_params['lpe_layers']
        lpe_dim = net_params['lpe_dim']
        lpe_n_heads = net_params['lpe_n_heads']

        self.readout = net_params['readout']
        self.layer_norm = net_params['layer_norm']
        self.batch_norm = net_params['batch_norm']
        self.residual = net_params['residual']
        self.device = net_params['device']
        self.wl_pos_enc = net_params['wl_pos_enc']
        self.spectral_attn = SpectralAttention(lpe_dim, lpe_n_heads, lpe_layers)

        self.n_classes = n_classes

        self.embedding_h = nn.Embedding(in_dim, hidden_dim - lpe_dim)
        
        self.in_feat_dropout = nn.Dropout(in_feat_dropout)
        
        self.layers = nn.ModuleList([ GraphTransformerLayer(hidden_dim, hidden_dim, num_heads, dropout,
                                                    self.layer_norm, self.batch_norm, self.residual) for _ in range(n_layers-1) ]) 
        self.layers.append(GraphTransformerLayer(hidden_dim, out_dim, num_heads, dropout, self.layer_norm, self.batch_norm, self.residual))
        # self.MLP_layer = MLPReadout(out_dim, 1)   # 1 out dim since regression problem        
        self.MLP_layer = MLPReadout(out_dim, n_classes)


    def forward(self, g, h, e, eigvecs, eigvals):
        h = self.embedding_h(h)
        h = self.spectral_attn(h, eigvecs, eigvals)
        h = self.in_feat_dropout(h)

        # convnets
        for conv in self.layers:
            # h, e = conv(g, h, e)
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
