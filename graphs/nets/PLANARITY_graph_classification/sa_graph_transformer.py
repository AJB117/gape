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
        self.edge_feat = net_params['edge_feat']
        self.device = net_params['device']
        self.wl_pos_enc = net_params['wl_pos_enc']
        self.spectral_attn = SpectralAttention(lpe_dim, lpe_n_heads, lpe_layers)

        self.n_classes = n_classes

        self.embedding_h = nn.Linear(in_dim, hidden_dim - lpe_dim)
        
        self.in_feat_dropout = nn.Dropout(in_feat_dropout)
        
        if self.edge_feat:
            from layers.graph_transformer_edge_layer import GraphTransformerLayer
            self.embedding_e = nn.Linear(1, hidden_dim)
        else:
            from layers.graph_transformer_layer import GraphTransformerLayer

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
        if self.edge_feat:
            for conv in self.layers:
                h, e = conv(g, h, e)
        else:
            for conv in self.layers:
                h = conv(g, h)
        g.ndata['h'] = h

        if self.edge_feat:
            e = torch.ones(e.size(0),1).to(self.device)
            e = self.embedding_e(e)   
        
        if self.readout == "sum":
            hg = dgl.sum_nodes(g, 'h')
        elif self.readout == "max":
            hg = dgl.max_nodes(g, 'h')
        elif self.readout == "mean":
            hg = dgl.mean_nodes(g, 'h')
        else:
            hg = dgl.mean_nodes(g, 'h')  # default readout is mean nodes

        out = self.MLP_layer(hg)
        return out

    def loss(self, pred, label):
        criterion = nn.CrossEntropyLoss()
        loss = criterion(pred, label)
        return loss
