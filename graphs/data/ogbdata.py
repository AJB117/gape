import torch
import torch.utils.data
import time
import numpy as np
import networkx as nx

import dgl

from scipy import sparse as sp
import numpy as np
import torch.nn.functional as F
from ogb.lsc import PCQM4Mv2Dataset

# *NOTE
# The dataset pickle and index files are in ./zinc_molecules/ dir
# [<split>.pickle and <split>.index; for split 'train', 'val' and 'test']




class OGBDGL(torch.utils.data.Dataset):
    def __init__(self, dataset, split, num_graphs=None):
        self.split = split
        self.num_graphs = num_graphs
        self.data = dataset
        self.num_graphs = len(self.data)

        self.graph_lists = []
        self.graph_labels = []
        self.spatial_pos_lists = []
        self.n_samples = len(self.data)
        self._prepare()

    def _prepare(self):
        print("preparing %d graphs for the %s set..." % (self.num_graphs, self.split.upper()))

        for molecule, label in self.data:
            self.graph_lists.append(molecule)
            self.graph_labels.append(label)
        
    def __len__(self):
        """Return the number of graphs in the dataset."""
        return self.n_samples

    def __getitem__(self, idx):
        """
            Get the idx^th sample.
            Parameters
            ---------
            idx : int
                The sample index.
            Returns
            -------
            (dgl.DGLGraph, int)
                DGLGraph with node feature stored in `feat` field
                And its label.
        """
        try:
            spatial_pos_list = self.spatial_pos_lists[idx]
        except:
            spatial_pos_list = None

        return self.graph_lists[idx], self.graph_labels[idx], spatial_pos_list


    
class OGBDatasetDGL(torch.utils.data.Dataset):
    def __init__(self, name='ogb'):
        t0 = time.time()
        self.name = name

        self.num_atom_type = 14
        self.num_atom_feat = 9

        data_dir='./data/dataset/'

        if self.name == 'ogb':
            self.train = OGBDGL(data_dir, 'train', num_graphs=3045360)
            self.val = OGBDGL(data_dir, 'valid', num_graphs=380670)
            self.test = OGBDGL(data_dir, 'test', num_graphs=377423)

        print("Time taken: {:.4f}s".format(time.time()-t0))
        


def self_loop(g):
    """
        Utility function only, to be used only when necessary as per user self_loop flag
        : Overwriting the function dgl.transform.add_self_loop() to not miss ndata['feat'] and edata['feat']
        
        
        This function is called inside a function in MoleculeDataset class.
    """
    new_g = dgl.DGLGraph()
    new_g.add_nodes(g.number_of_nodes())
    new_g.ndata['feat'] = g.ndata['feat']

    try:
        new_g.ndata['pos_enc'] = g.ndata['pos_enc']
    except:
        pass
    
    src, dst = g.all_edges(order="eid")
    src = dgl.backend.zerocopy_to_numpy(src)
    dst = dgl.backend.zerocopy_to_numpy(dst)
    non_self_edges_idx = src != dst
    nodes = np.arange(g.number_of_nodes())
    new_g.add_edges(src[non_self_edges_idx], dst[non_self_edges_idx])
    new_g.add_edges(nodes, nodes)
    
    # This new edata is not used since this function gets called only for GCN, GAT
    # However, we need this for the generic requirement of ndata and edata
    new_g.edata['feat'] = torch.zeros(new_g.number_of_edges())
    return new_g



def positional_encoding(g, pos_enc_dim):
    """
        Graph positional encoding v/ Laplacian eigenvectors
    """

    # Laplacian
    A = g.adjacency_matrix_scipy(return_edge_ids=False).astype(float)
    N = sp.diags(dgl.backend.asnumpy(g.in_degrees()).clip(1) ** -0.5, dtype=float)
    L = sp.eye(g.number_of_nodes()) - N * A * N

    # Eigenvectors with numpy
    EigVal, EigVec = np.linalg.eig(L.toarray())
    idx = EigVal.argsort() # increasing order
    EigVal, EigVec = EigVal[idx], np.real(EigVec[:,idx])
    g.ndata['pos_enc'] = torch.from_numpy(EigVec[:,1:pos_enc_dim+1]).float() 

    # # Eigenvectors with scipy
    # EigVal, EigVec = sp.linalg.eigs(L, k=pos_enc_dim+1, which='SR')
    # EigVec = EigVec[:, EigVal.argsort()] # increasing order
    # g.ndata['pos_enc'] = torch.from_numpy(np.abs(EigVec[:,1:pos_enc_dim+1])).float() 
    
    n = g.number_of_nodes()
    if n <= pos_enc_dim:
        g.ndata['pos_enc'] = F.pad(g.ndata['pos_enc'], (0, pos_enc_dim - n + 1), value=float('0'))
    
    return g

def make_full_graph(g):
    """
        Converting the given graph to fully connected
        This function just makes full connections
        removes available edge features 
    """

    full_g = dgl.from_networkx(nx.complete_graph(g.number_of_nodes()))
    full_g.ndata['feat'] = g.ndata['feat']
    full_g.edata['feat'] = torch.zeros(full_g.number_of_edges()).long()
    
    try:
        full_g.ndata['pos_enc'] = g.ndata['pos_enc']
    except:
        pass

    try:
        count = 0
        while True:
            full_g.ndata[f'pos_enc_{count}'] = g.ndata[f'pos_enc_{count}']
            count += 1
    except:
        pass

    try:
        full_g.ndata['EigVals'] = g.ndata['EigVals']
        full_g.ndata['EigVecs'] = g.ndata['EigVecs']
    except:
        pass
    
    try:
        full_g.ndata['wl_pos_enc'] = g.ndata['wl_pos_enc']
    except:
        pass    
    
    return full_g


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield (i, i+n)


class OGBDataset(torch.utils.data.Dataset):

    def __init__(self, name, fold=1, logger=None):
        """
            Loading Moleccular datasets
        """
        start = time.time()
        if logger:
            logger.info(f"[I] Loading dataset {name}...")
        else:
            print(f"[I] Loading dataset {name}...")

        dataset = PCQM4Mv2Dataset(root='data/datasetv2')

        self.name = name
        self.num_atom_type = 14
        self.num_atom_feat = 9

        if logger:
            logger.info("Splitting dataset...")
        else:
            print("Splitting dataset...")
            
        if name == 'OGB':
            splits = dataset.get_idx_split()
            train = dataset[splits['train']]
            val = dataset[splits['valid']]
            self.train = OGBDGL(train, 'train')
            self.val = OGBDGL(val, 'valid')

        if logger:
            logger.info("Time taken: {:.4f}s".format(time.time()-start))
        else:
            print("Time taken: {:.4f}s".format(time.time()-start))

        if logger:
            # logger.info(f'train, test, val sizes: {len(self.train)},{len(self.test)},{len(self.val)}')
            logger.info(f'train, val sizes: {len(self.train)},{len(self.val)}')
            logger.info("[I] Finished loading.")
            logger.info("[I] Data load time: {:.4f}s".format(time.time()-start))
        else:
            # print('train, test, val sizes :',len(self.train),len(self.test),len(self.val))
            print('train, val sizes :',len(self.train),len(self.val))
            print("[I] Finished loading.")
            print("[I] Data load time: {:.4f}s".format(time.time()-start))


    # form a mini batch from a given list of samples = [(graph, label) pairs]
    def collate(self, samples):
        # The input samples is a list of pairs (graph, label).
        graphs, labels, spatial_pos_biases = map(list, zip(*samples))
        labels = torch.tensor(labels).unsqueeze(1)
        batched_graph = dgl.batch(graphs)
        # if all(bool(x) for x in spatial_pos_biases):
        if all([x is not None for x in spatial_pos_biases]):
            batched_spatial_pos_biases = torch.block_diag(*spatial_pos_biases)
        else:
            batched_spatial_pos_biases = None

        return batched_graph, labels, batched_spatial_pos_biases
    
    
    def _sym_normalize_adj(self, adj):
        deg = torch.sum(adj, dim = 0)#.squeeze()
        deg_inv = torch.where(deg>0, 1./torch.sqrt(deg), torch.zeros(deg.size()))
        deg_inv = torch.diag(deg_inv)
        return torch.mm(deg_inv, torch.mm(adj, deg_inv))
    
    def _add_self_loops(self):
        
        # function for adding self loops
        # this function will be called only if self_loop flag is True
            
        self.train.graph_lists = [self_loop(g) for g in self.train.graph_lists]
        self.val.graph_lists = [self_loop(g) for g in self.val.graph_lists]
        self.test.graph_lists = [self_loop(g) for g in self.test.graph_lists]

    def _add_positional_encodings(self, pos_enc_dim):
        
        # Graph positional encoding v/ Laplacian eigenvectors
        self.train.graph_lists = [positional_encoding(g, pos_enc_dim) for g in self.train.graph_lists]
        self.val.graph_lists = [positional_encoding(g, pos_enc_dim) for g in self.val.graph_lists]
        if self.name != 'OGB':
            self.test.graph_lists = [positional_encoding(g, pos_enc_dim) for g in self.test.graph_lists]

    def _make_full_graph(self):
        
        # function for converting graphs to full graphs
        # this function will be called only if full_graph flag is True
        self.train.graph_lists = [make_full_graph(g) for g in self.train.graph_lists]
        self.val.graph_lists = [make_full_graph(g) for g in self.val.graph_lists]
        self.test.graph_lists = [make_full_graph(g) for g in self.test.graph_lists]
