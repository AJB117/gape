import torch
import torch.nn as nn
import torch.nn.functional as F

import dgl
import dgl.function as fn
import numpy as np

"""
    Graph Transformer Layer
    
"""

"""
    Util functions
"""
def src_dot_dst(src_field, dst_field, out_field):
    def func(edges):
        return {out_field: (edges.src[src_field] * edges.dst[dst_field])}
    return func

def scaling(field, scale_constant):
    def func(edges):
        return {field: ((edges.data[field]) / scale_constant)}
    return func

def add_bias(field, bias):
    def func(edges):
        eids = edges.edges()
        src, dest = eids[0], eids[1]
        bias_weights = bias[src, dest].unsqueeze(-1)
        return {field: edges.data[field] + bias_weights}
    return func

def add_node_bias(field, bias):
    def func(nodes):
        return {
            field: nodes.data[field] + bias
        }
    return func

def exp(field):
    def func(edges):
        # clamp for softmax numerical stability
        return {'score_soft': torch.exp((edges.data[field].sum(-1, keepdim=True)).clamp(-5, 5))}
    return func


# def src_dot_dst(src_field, dst_field, out_field):
#     def func(edges):
#         return {out_field: (edges.src[src_field] * edges.dst[dst_field]).sum(-1, keepdim=True)}
#     return func

# def scaled_exp(field, scale_constant):
#     def func(edges):
#         # clamp for softmax numerical stability
#         return {field: torch.exp((edges.data[field] / scale_constant).clamp(-5, 5))}

#     return func

# def scaled_exp_with_bias(field, scale_constant, bias):
#     def func(edges):
#         src, dest = edges.edges()[0], edges.edges()[1]
#         bias_weights = bias[src, dest].unsqueeze(-1)

#         # clamp for softmax numerical stability
#         return {field: torch.exp((edges.data[field] / scale_constant).clamp(-5, 5) + bias_weights)}

#     return func

# def add_pe(g, field):
#     def func(nodes):
#         return {field: nodes.data[field] + g.ndata['pos_enc']}
#     return func

"""
    Single Attention Head
"""

class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, in_dim, out_dim, num_heads, use_bias):
        super().__init__()
        
        self.out_dim = out_dim
        self.num_heads = num_heads
        
        if use_bias:
            self.Q = nn.Linear(in_dim, out_dim * num_heads, bias=True)
            self.K = nn.Linear(in_dim, out_dim * num_heads, bias=True)
            self.V = nn.Linear(in_dim, out_dim * num_heads, bias=True)
        else:
            self.Q = nn.Linear(in_dim, out_dim * num_heads, bias=False)
            self.K = nn.Linear(in_dim, out_dim * num_heads, bias=False)
            self.V = nn.Linear(in_dim, out_dim * num_heads, bias=False)
        
    def propagate_attention(self, g: dgl.DGLGraph, spatial_pos_bias=None):
        g.apply_edges(src_dot_dst('K_h', 'Q_h', 'score'))
        g.apply_edges(scaling('score', np.sqrt(self.out_dim)))

        if spatial_pos_bias is not None:
            g.apply_edges(add_bias('score', spatial_pos_bias))

        g.apply_edges(exp('score'))

        eids = g.edges()
        g.send_and_recv(eids, fn.src_mul_edge('V_h', 'score_soft', 'V_h'), fn.sum('V_h', 'wV'))
        g.send_and_recv(eids, fn.copy_edge('score_soft', 'score_soft'), fn.sum('score_soft', 'z'))

    # def propagate_attention(self, g, spatial_pos_bias=None):
    #     # Compute attention score
    #     # g.apply_nodes(add_pe(g, 'K_h'))
    #     g.apply_edges(src_dot_dst('K_h', 'Q_h', 'score')) #, edges)
    #     if spatial_pos_bias is not None:
    #         g.apply_edges(scaled_exp_with_bias('score', np.sqrt(self.out_dim), spatial_pos_bias))
    #     else:
    #         g.apply_edges(scaled_exp('score', np.sqrt(self.out_dim)))

    #     # Send weighted values to target nodes
    #     eids = g.edges()
    #     # g.apply_nodes(add_pe(g, 'V_h'))
    #     g.send_and_recv(eids, fn.src_mul_edge('V_h', 'score', 'V_h'), fn.sum('V_h', 'wV'))
    #     g.send_and_recv(eids, fn.copy_edge('score', 'score'), fn.sum('score', 'z'))
    
    def forward(self, g, h, spatial_pos_bias=None):
        
        Q_h = self.Q(h)
        K_h = self.K(h)
        V_h = self.V(h)

        # Reshaping into [num_nodes, num_heads, feat_dim] to 
        # get projections for multi-head attention
        g.ndata['Q_h'] = Q_h.view(-1, self.num_heads, self.out_dim)
        g.ndata['K_h'] = K_h.view(-1, self.num_heads, self.out_dim)
        g.ndata['V_h'] = V_h.view(-1, self.num_heads, self.out_dim)
        # g.ndata['pos_enc'] = g.ndata['pos_enc'].view(-1, self.num_heads, self.out_dim)
        
        self.propagate_attention(g, spatial_pos_bias)
        head_out = g.ndata['wV']/g.ndata['z']
        head_out[head_out != head_out] = 0
        
        return head_out
    

class GraphTransformerLayer(nn.Module):
    """
        Param: 
    """
    def __init__(self, in_dim, out_dim, num_heads, dropout=0.0, layer_norm=False, batch_norm=True, residual=True, use_bias=False):
        super().__init__()

        self.in_channels = in_dim
        self.out_channels = out_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.residual = residual
        self.layer_norm = layer_norm        
        self.batch_norm = batch_norm
        
        self.attention = MultiHeadAttentionLayer(in_dim, out_dim//num_heads, num_heads, use_bias)
        
        self.O = nn.Linear(out_dim, out_dim)

        if self.layer_norm:
            self.layer_norm1 = nn.LayerNorm(out_dim)
            
        if self.batch_norm:
            self.batch_norm1 = nn.BatchNorm1d(out_dim)

        # FFN
        self.FFN_layer1 = nn.Linear(out_dim, out_dim*2)
        self.FFN_layer2 = nn.Linear(out_dim*2, out_dim)

        if self.layer_norm:
            self.layer_norm2 = nn.LayerNorm(out_dim)
            
        if self.batch_norm:
            self.batch_norm2 = nn.BatchNorm1d(out_dim)
        
    def forward(self, g, h, spatial_pos_bias=None):
        h_in1 = h # for first residual connection
        
        # multi-head attention out
        if self.layer_norm:
            h = self.layer_norm1(h)

        attn_out = self.attention(g, h, spatial_pos_bias=spatial_pos_bias)
        h = attn_out.view(-1, self.out_channels)
        
        h = F.dropout(h, self.dropout, training=self.training)
        
        h = self.O(h)
        
        if self.residual:
            h = h_in1 + h # residual connection
        
        if self.layer_norm:
            h = self.layer_norm1(h)
            
        if self.batch_norm:
            h = self.batch_norm1(h)
        
        h_in2 = h # for second residual connection
        
        # FFN
        h = self.FFN_layer1(h)
        h = F.relu(h)
        h = F.dropout(h, self.dropout, training=self.training)
        h = self.FFN_layer2(h)

        if self.residual:
            h = h_in2 + h # residual connection
        
        if self.layer_norm:
            h = self.layer_norm2(h)
            
        if self.batch_norm:
            h = self.batch_norm2(h)       

        return h
        
    def __repr__(self):
        return '{}(in_channels={}, out_channels={}, heads={}, residual={})'.format(self.__class__.__name__,
                                             self.in_channels,
                                             self.out_channels, self.num_heads, self.residual)