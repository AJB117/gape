"""
    Utility file to select GraphNN model as
    selected by the user
"""

from nets.CSL_graph_classification.graph_transformer import GraphTransformerNet
from nets.CSL_graph_classification.pseudo_graphormer import PseudoGraphormerNet
from nets.CSL_graph_classification.sa_graph_transformer import SAGraphTransformerNet


def GraphTransformer(net_params):
    return GraphTransformerNet(net_params)

def PseudoGraphormer(net_params):
    return PseudoGraphormerNet(net_params)

def SAGraphTransformer(net_params):
    return SAGraphTransformerNet(net_params)

def gnn_model(MODEL_NAME, net_params):
    models = {
        'GraphTransformer': GraphTransformer,
        'PseudoGraphormer': PseudoGraphormer,
        'SAGraphTransformer': SAGraphTransformer
    }
        
    return models[MODEL_NAME](net_params)