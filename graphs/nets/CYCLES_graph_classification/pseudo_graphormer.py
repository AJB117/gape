
import torch
import torch.nn as nn
import dgl

"""
    Graphormer without spatial edge encoding and VNode
"""
# from layers.graph_transformer_edge_layer import GraphTransformerLayer
from layers.graph_transformer_layer import GraphTransformerLayer
from layers.mlp_readout_layer import MLPReadout

class PseudoGraphormerNet(nn.Module):
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
        # self.edge_feat = net_params['edge_feat']
        self.device = net_params['device']

        in_deg_centrality = net_params['in_deg_centrality']
        out_deg_centrality = net_params['out_deg_centrality']
        spd_len = net_params['spd_len']
        self.n_classes = n_classes

        self.in_degree_encoder = nn.Embedding(in_deg_centrality, hidden_dim, padding_idx=0)
        self.out_degree_encoder = nn.Embedding(
            out_deg_centrality, hidden_dim, padding_idx=0
        )
        self.spatial_pos_encoder = nn.Embedding(spd_len, num_heads, padding_idx=0)

        self.embedding_h = nn.Linear(in_dim, hidden_dim) # node feat is an integer
        
        self.in_feat_dropout = nn.Dropout(in_feat_dropout)
        
        self.layers = nn.ModuleList([ GraphTransformerLayer(hidden_dim, hidden_dim, num_heads, dropout,
                                                    self.layer_norm, self.batch_norm, self.residual) for _ in range(n_layers-1) ]) 
        self.layers.append(GraphTransformerLayer(hidden_dim, out_dim, num_heads, dropout, self.layer_norm, self.batch_norm, self.residual))
        self.MLP_layer = MLPReadout(out_dim, n_classes)

    def forward(self, g, h, e, spatial_pos_bias):
        h = self.embedding_h(h)
        h = self.in_feat_dropout(h)
        h = h + self.in_degree_encoder(g.in_degrees()) + self.out_degree_encoder(g.out_degrees())

        # spatial_pos = g.ndata['spatial_pos_bias']
        spatial_pos_bias = self.spatial_pos_encoder(spatial_pos_bias)
        # g.ndata['spatial_pos_bias'] = spatial_pos_bias.permute(2, 0, 1) # (num_heads, V, V)
        # spatial_pos_bias = spatial_pos_bias.permute(2, 0, 1) # (num_heads, V, V)

        # convnets
        for conv in self.layers:
            # h, e = conv(g, h, e)
            h = conv(g, h, spatial_pos_bias=spatial_pos_bias)

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
        criterion = nn.CrossEntropyLoss()
        loss = criterion(pred, label)
        return loss