"""
    Utility file to select GraphNN model as
    selected by the user
"""

from nets.ogb_graph_regression.graph_transformer import GraphTransformerNet
from nets.ogb_graph_regression.sa_graph_transformer import SAGraphTransformerNet 
from nets.ogb_graph_regression.pseudo_graphormer import PseudoGraphormerNet

def GraphTransformer(net_params):
    return GraphTransformerNet(net_params)

def SAGraphTransformer(net_params):
    return SAGraphTransformerNet(net_params)

def PseudoGraphormer(net_params):
    return PseudoGraphormerNet(net_params)

def gnn_model(MODEL_NAME, net_params):
    models = {
        'GraphTransformer': GraphTransformer,
        'SAGraphTransformer': SAGraphTransformer,
        'PseudoGraphormer': PseudoGraphormer
    }
        
    return models[MODEL_NAME](net_params)