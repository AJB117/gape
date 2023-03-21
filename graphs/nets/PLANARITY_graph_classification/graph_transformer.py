import torch
import torch.nn as nn

import dgl
from layers.pe_layer import PELayer

"""
    Graph Transformer with edge features
    
"""
from layers.graph_transformer_layer import GraphTransformerLayer
from layers.mlp_readout_layer import MLPReadout

class GraphTransformerNet(nn.Module):
    def __init__(self, net_params):
        super().__init__()
        in_dim = net_params['in_dim']
        hidden_dim = net_params['hidden_dim']
        n_classes = net_params['n_classes']
        num_heads = net_params['n_heads']
        out_dim = net_params['out_dim']
        in_feat_dropout = net_params['in_feat_dropout']
        dropout = net_params['dropout']
        n_layers = net_params['L']
        self.readout = net_params['readout']
        self.layer_norm = net_params['layer_norm']
        self.batch_norm = net_params['batch_norm']
        self.residual = net_params['residual']
        self.device = net_params['device']
        self.gape_per_layer = net_params['gape_per_layer']
        self.pe_layer = PELayer(net_params)
        self.cat = net_params.get('cat_gape', False)

        if self.cat:
            self.embedding_h = nn.Linear(in_dim, hidden_dim - net_params['pos_enc_dim'])
        else:
            self.embedding_h = nn.Linear(in_dim, hidden_dim)
        
        self.in_feat_dropout = nn.Dropout(in_feat_dropout)
        
        self.layers = nn.ModuleList([ GraphTransformerLayer(hidden_dim, hidden_dim, num_heads, dropout,
                                                    self.layer_norm, self.batch_norm, self.residual) for _ in range(n_layers-1) ]) 
        self.layers.append(GraphTransformerLayer(hidden_dim, out_dim, num_heads, dropout, self.layer_norm, self.batch_norm, self.residual))
        # self.MLP_layer = MLPReadout(out_dim, 1)   # 1 out dim since regression problem        
        self.MLP_layer = MLPReadout(out_dim, n_classes)

        if self.cat:
            self.ll = nn.Linear(hidden_dim - net_params['pos_enc_dim'], hidden_dim)
        

    def forward(self, g, h, e, pos_enc=None, h_wl_pos_enc=None):
        h = self.embedding_h(h)
        if self.cat:
            pe = self.pe_layer(g, h, pos_enc)
            h = torch.cat((h, pe), dim=1)
            h = self.in_feat_dropout(h)
        else:
        # h = self.embedding_h(h)
            h = self.in_feat_dropout(h)
            if self.pe_layer.use_pos_enc:
                pe = self.pe_layer(g, h, pos_enc)
                h = h + pe
            # h = pe

        # convnets
        for conv in self.layers:
            h = conv(g, h)

            if self.gape_per_layer:
                h = h + pe

        g.ndata['h'] = h
        
        if self.readout == "sum":
            hg = dgl.sum_nodes(g, 'h')
        elif self.readout == "max":
            hg = dgl.max_nodes(g, 'h')
        elif self.readout == "mean":
            hg = dgl.mean_nodes(g, 'h')
        else:
            hg = dgl.mean_nodes(g, 'h')  # default readout is mean nodes
            
        # self.g = g
        return self.MLP_layer(hg)
        
    def loss(self, pred, label):
        criterion = nn.CrossEntropyLoss()
        loss_a = criterion(pred, label)
        return loss_a