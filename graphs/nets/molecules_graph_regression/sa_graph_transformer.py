import torch
import torch.nn as nn

import dgl
from layers.pe_layer import PELayer
from layers.spectral_attention import SpectralAttention

"""
    Graph Transformer with edge features
    
"""
from layers.mlp_readout_layer import MLPReadout

class SAGraphTransformerNet(nn.Module):
    def __init__(self, net_params):
        super().__init__()
        num_atom_type = net_params['num_atom_type']
        num_bond_type = net_params['num_bond_type']
        hidden_dim = net_params['hidden_dim']
        num_heads = net_params['n_heads']
        out_dim = net_params['out_dim']
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
        self.edge_feat = net_params['edge_feat']
        self.pe_layer = PELayer(net_params)
        self.spectral_attn = SpectralAttention(lpe_dim, lpe_n_heads, lpe_layers)

        self.embedding_h = nn.Embedding(num_atom_type, hidden_dim - lpe_dim)
        if self.edge_feat:
            self.embedding_e = nn.Embedding(num_bond_type, hidden_dim)
        # else:
        #     self.embedding_e = nn.Linear(1, hidden_dim)
        
        self.in_feat_dropout = nn.Dropout(in_feat_dropout)
        
        if not self.edge_feat:
            from layers.graph_transformer_layer import GraphTransformerLayer
        else:
            from layers.graph_transformer_edge_layer import GraphTransformerLayer

        self.layers = nn.ModuleList([ GraphTransformerLayer(hidden_dim, hidden_dim, num_heads, dropout,
                                                    self.layer_norm, self.batch_norm, self.residual) for _ in range(n_layers-1) ]) 
        self.layers.append(GraphTransformerLayer(hidden_dim, out_dim, num_heads, dropout, self.layer_norm, self.batch_norm, self.residual))
        self.MLP_layer = MLPReadout(out_dim, 1)   # 1 out dim since regression problem        

    def forward(self, g, h, e, eigvecs, eigvals):
        h = self.embedding_h(h)
        h = self.spectral_attn(h, eigvecs, eigvals)
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
