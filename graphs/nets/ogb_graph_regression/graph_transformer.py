import torch
import torch.nn as nn

import dgl
from layers.pe_layer import PELayer
from ogb.graphproppred.mol_encoder import AtomEncoder
from layers.graph_transformer_layer import GraphTransformerLayer

"""
    Graph Transformer with edge features
    
"""
from layers.mlp_readout_layer import MLPReadout

class GraphTransformerNet(nn.Module):
    def __init__(self, net_params):
        super().__init__()
        num_atom_type = net_params['num_atom_type']
        hidden_dim = net_params['hidden_dim']
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
        self.edge_feat = net_params['edge_feat']
        self.gape_per_layer = net_params['gape_per_layer']
        self.pe_layer = PELayer(net_params)

        self.embedding_h = AtomEncoder(hidden_dim)
        
        self.in_feat_dropout = nn.Dropout(in_feat_dropout)
        

        self.layers = nn.ModuleList([ GraphTransformerLayer(hidden_dim, hidden_dim, num_heads, dropout,
                                                    self.layer_norm, self.batch_norm, self.residual) for _ in range(n_layers-1) ]) 
        self.layers.append(GraphTransformerLayer(hidden_dim, out_dim, num_heads, dropout, self.layer_norm, self.batch_norm, self.residual))
        self.MLP_layer = MLPReadout(out_dim, 1)   # 1 out dim since regression problem        

    def forward(self, g, h, e, pos_enc=None):
        h = self.embedding_h(h)
        h = self.in_feat_dropout(h)

        if self.pe_layer.use_pos_enc:
            pe = self.pe_layer(g, h, pos_enc)
            h = h + pe

        # h = torch.cat([h, pe], dim=1)
        # print(h.shape)
        # h = self.ll(h)
        h = self.in_feat_dropout(h)

        if self.edge_feat:
        # if not self.edge_feat: # edge feature set to 1
            # e = torch.ones(e.size(0),1).to(self.device)
            e = self.embedding_e(e)   
        
        # convnets
        for conv in self.layers:
            if self.edge_feat:
                h, e = conv(g, h, e)
            else:
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
            
        return self.MLP_layer(hg)
        
    def loss(self, pred, label):
        loss = nn.L1Loss()(pred, label)
        return loss
