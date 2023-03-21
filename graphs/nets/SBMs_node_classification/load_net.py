"""
    Utility file to select GraphNN model as
    selected by the user
"""

from nets.SBMs_node_classification.graph_transformer import GraphTransformerNet
from nets.SBMs_node_classification.sa_graph_transformer import SAGraphTransformerNet
from nets.SBMs_node_classification.pseudo_graphormer import PseudoGraphormerNet


def GraphTransformer(net_params):
    return GraphTransformerNet(net_params)

def SAN_NodeLPE(net_params):
    return SAGraphTransformerNet(net_params)

def PseudoGraphormer(net_params):
    return PseudoGraphormerNet(net_params)

def gnn_model(MODEL_NAME, net_params):
    models = {
        'GraphTransformer': GraphTransformer,
        'SAGraphTransformer': SAN_NodeLPE,
        'PseudoGraphormer': PseudoGraphormer
    }
        
    return models[MODEL_NAME](net_params)