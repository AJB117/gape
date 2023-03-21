import os
import torch
import dgl
import scipy
import pickle
import scipy.sparse as sp
import numpy as np
import torch.nn.functional as F
import pyximport

pyximport.install(setup_args={"include_dirs": np.get_include()})
from . import algos

def spd_encoding(g):
    shortest_path_result, _ = algos.floyd_warshall(g.adjacency_matrix().to_dense().numpy().astype(int))
    spatial_pos = torch.from_numpy((shortest_path_result)).long()

    spatial_pos = spatial_pos.type(torch.long)

    return spatial_pos

def add_spd_encodings(dataset):
    dataset.train.spatial_pos_lists = [spd_encoding(g) for g in dataset.train.graph_lists]
    dataset.val.spatial_pos_lists = [spd_encoding(g) for g in dataset.val.graph_lists]
    if dataset.name != 'OGB':
        dataset.test.spatial_pos_lists = [spd_encoding(g) for g in dataset.test.graph_lists]
    return dataset

def simple_spectral_decomp(g):
    A = g.adjacency_matrix_scipy(return_edge_ids=False).astype(float)
    EigVals, EigVecs = np.linalg.eigh(A.toarray())
    setattr(g, 'EigVecs', torch.from_numpy(EigVecs))
    setattr(g, 'EigVals', torch.from_numpy(EigVals))
    return g

def add_simple_spectral_decomp(dataset):
    dataset.train.graph_lists = [simple_spectral_decomp(g) for g in dataset.train.graph_lists]
    dataset.val.graph_lists = [simple_spectral_decomp(g) for g in dataset.val.graph_lists]
    dataset.test.graph_lists = [simple_spectral_decomp(g) for g in dataset.test.graph_lists]
    return dataset


def spectral_decomposition(g, pos_enc_dim):
    # Laplacian
    n = g.number_of_nodes()
    A = g.adjacency_matrix_scipy(return_edge_ids=False).astype(float)
    N = sp.diags(dgl.backend.asnumpy(g.in_degrees()).clip(1) ** -0.5, dtype=float)
    L = sp.eye(g.number_of_nodes()) - N * A * N

    # Eigenvectors with numpy
    EigVals, EigVecs = np.linalg.eigh(L.toarray())
    EigVals, EigVecs = EigVals[: pos_enc_dim], EigVecs[:, :pos_enc_dim]  # Keep up to the maximum desired number of frequencies

    # Normalize and pad EigenVectors
    EigVecs = torch.from_numpy(EigVecs).float()
    EigVecs = F.normalize(EigVecs, p=2, dim=1, eps=1e-12, out=None)
    
    if n<pos_enc_dim:
        g.ndata['EigVecs'] = F.pad(EigVecs, (0, pos_enc_dim-n), value=float('nan'))
    else:
        g.ndata['EigVecs']= EigVecs
        
    
    #Save eigenvales and pad
    EigVals = torch.from_numpy(np.sort(np.abs(np.real(EigVals)))) #Abs value is taken because numpy sometimes computes the first eigenvalue approaching 0 from the negative
    
    if n<pos_enc_dim:
        EigVals = F.pad(EigVals, (0, pos_enc_dim-n), value=float('nan')).unsqueeze(0)
    else:
        EigVals=EigVals.unsqueeze(0)
        
    
    #Save EigVals node features
    g.ndata['EigVals'] = EigVals.repeat(g.number_of_nodes(),1).unsqueeze(2)
    
    return g


def add_spectral_decomposition(dataset, pos_enc_dim):
    dataset.train.graph_lists = [spectral_decomposition(g, pos_enc_dim) for g in dataset.train.graph_lists]
    dataset.val.graph_lists = [spectral_decomposition(g, pos_enc_dim) for g in dataset.val.graph_lists]
    if dataset.name != 'OGB':
        dataset.test.graph_lists = [spectral_decomposition(g, pos_enc_dim) for g in dataset.test.graph_lists]
    return dataset

def random_walk_encoding(g, pos_enc_dim, type='partial', ret_pe=False):
    A = g.adjacency_matrix(scipy_fmt="csr")
    Dinv = sp.diags(dgl.backend.asnumpy(g.in_degrees()).clip(1) ** -1.0, dtype=float) # D^-1
    RW = A * Dinv  
    M = RW

    nb_pos_enc = pos_enc_dim
    PE = [torch.from_numpy(M.diagonal()).float()]
    M_power = M
    for _ in range(nb_pos_enc-1):
        M_power = M_power * M
        if type == 'partial':
            PE.append(torch.from_numpy(M_power.diagonal()).float())
        else:
            PE.append(torch.from_numpy(M_power).float())
    PE = torch.stack(PE,dim=-1)
    if ret_pe:
        return PE

    g.ndata['pos_enc'] = PE  

    return g


def add_rw_pos_encodings(dataset, pos_enc_dim, type='partial', logger=None):
    if logger:
        logger.info("Adding PE to train graphs...")
    dataset.train.graph_lists = [random_walk_encoding(g, pos_enc_dim, type) for g in dataset.train.graph_lists]
    if logger:
        logger.info("Adding PE to val graphs...")
    dataset.val.graph_lists = [random_walk_encoding(g, pos_enc_dim, type) for g in dataset.val.graph_lists]
    if dataset.name != 'OGB':
        if logger:
            logger.info("Adding PE to test graphs...")
        dataset.test.graph_lists = [random_walk_encoding(g, pos_enc_dim, type) for g in dataset.test.graph_lists]
    return dataset


def multiple_automaton_encodings(g: dgl.DGLGraph, transition_matrix, initial_vector, diag=False, matrix='A', idx=0, model=None):
    pe = automaton_encoding(g, transition_matrix, initial_vector, diag, matrix, ret_pe=True, storage=None, idx=idx, model=model)
    key = f'pos_enc_{idx}'
    g.ndata[key] = pe
    return g

def random_orientation(g: dgl.DGLGraph):
    edges = g.edges()
    src_tensor, dst_tensor = edges[0], edges[1]
    for i, (src, dst) in enumerate(zip(src_tensor, dst_tensor)):
        if (dst, src) in g.edges():
            p = np.random.rand()
            if p > 0.5:
                g.remove_edge(i)
    return g

def add_random_orientation(dataset):
    dataset.train.graph_lists = [random_orientation(g) for g in dataset.train.graph_lists]
    dataset.val.graph_lists = [random_orientation(g) for g in dataset.val.graph_lists]
    dataset.test.graph_lists = [random_orientation(g) for g in dataset.test.graph_lists]
    return dataset

def add_multiple_automaton_encodings(dataset, transition_matrices, initial_vectors, diag=False, matrix='A', model=None):
    transition_matrix = transition_matrices[0]
    initial_vector = initial_vectors[0]
    # for i, (_, _) in enumerate(zip(transition_matrices, initial_vectors)):
    for i, _ in enumerate(transition_matrices):
        dataset.train.graph_lists = [multiple_automaton_encodings(g, transition_matrix, initial_vector, diag, matrix, i, model) for g in dataset.train.graph_lists]
        dataset.val.graph_lists = [multiple_automaton_encodings(g, transition_matrix, initial_vector, diag, matrix, i, model) for g in dataset.val.graph_lists]
        dataset.test.graph_lists = [multiple_automaton_encodings(g, transition_matrix, initial_vector, diag, matrix, i, model) for g in dataset.test.graph_lists]

    # dump_encodings(dataset, transition_matrix.shape[0])
    return dataset

def automaton_encoding(g, transition_matrix, initial_vector, diag=False, matrix='A', ret_pe=False, idx=0, model=None):
    transition_matrix = torch.nan_to_num(transition_matrix)
    if diag:
        transition_inv = transition_matrix**-1
    else:
        transition_inv = torch.linalg.inv(transition_matrix).cpu().numpy()

    if matrix == 'A':
        # Adjacency matrix
        mat = g.adjacency_matrix().to_dense().cpu().numpy()
        if idx > 0:
            mat = np.linalg.matrix_power(mat, idx)
        # mat = np.linalg.matrix_power(mat, idx+1)
    elif matrix == 'L':
        # Normalized Laplacian
        n = g.number_of_nodes()
        A = g.adjacency_matrix_scipy(return_edge_ids=False).astype(float)
        D = sp.diags(dgl.backend.asnumpy(g.in_degrees()).clip(1) ** -0.5, dtype=float)
        L = sp.eye(n) - D * A * D
        mat = L.todense()
    elif matrix == 'SL':
        # Normalized unsigned Laplacian
        n = g.number_of_nodes()
        A = g.adjacency_matrix_scipy(return_edge_ids=False).astype(float)
        D = sp.diags(dgl.backend.asnumpy(g.in_degrees()).clip(1) ** -0.5, dtype=float)
        L = sp.eye(n) + D * A * D
        mat = L.todense()

    if model.pe_layer.gape_normalize_mat:
        D = sp.diags(dgl.backend.asnumpy(g.in_degrees()).clip(1) ** -1.0, dtype=float) # D^-1
        mat = mat @ D
    
    if model.pe_layer.ngape_betas:
        gape_beta = float(model.pe_layer.ngape_betas[idx])
    else:
        gape_beta = float(model.pe_layer.gape_beta)

    if gape_beta < 1:
        mat = mat * (1-gape_beta) # emulate pagerank


    if model.pe_layer.gape_weight_id:
        initial_vector = torch.zeros_like(initial_vector)
        import random
        rows, cols = initial_vector.shape
        indices = [n for n in range(rows)]
        for i in range(cols):
            p = random.choice(indices)
            initial_vector[p, i] = 1
    else:
        initial_vector = model.pe_layer.stack_strategy(g)

    if gape_beta < 1:
        initial_vector = initial_vector * gape_beta # emulate pagerank

    initial_vector_torch = initial_vector.clone()

    if diag:
        # mat_product = torch.einsum('ij, i->ij', initial_vector, transition_inv).cpu().numpy()
        mat_product = (torch.diag(transition_matrix) @ initial_vector).cpu().numpy() 
        transition_inv = torch.diag(transition_inv).cpu().numpy()
    else:
        initial_vector = initial_vector.cpu().numpy()
        mat_product = transition_inv @ initial_vector


    pe = scipy.linalg.solve_sylvester(transition_inv, -mat, mat_product)
    pe = torch.from_numpy(pe.T).float()

    if model.pe_layer.gape_tau:
        pe = pe.to(torch.device('cpu'))
        initial_vector_torch = initial_vector_torch.to(torch.device('cpu'))
        pe = torch.mul(pe.T, initial_vector_torch).T

    if ret_pe:
        return pe

    g.ndata['pos_enc'] = pe
    return g

def add_automaton_encodings(dataset, transition_matrix, initial_vector, diag=False, matrix='A', model=None):
    # Graph positional encoding w/ pre-computed automaton encoding
    storage = {
        'before': {
            'mins': [],
            'maxs': [],
            'all': []
        },
        'after': {
            'mins': [],
            'maxs': [],
            'all': []
        }
    }

    dataset.train.graph_lists = [automaton_encoding(g, transition_matrix, initial_vector, diag, matrix, False, model=model) for g in dataset.train.graph_lists]
    dataset.val.graph_lists = [automaton_encoding(g, transition_matrix, initial_vector, diag, matrix, False, model=model) for g in dataset.val.graph_lists]
    if dataset.name != 'OGB':
        dataset.test.graph_lists = [automaton_encoding(g, transition_matrix, initial_vector, diag, matrix, False, model=model) for g in dataset.test.graph_lists]
    return dataset

def add_random_walk_encoding_CSL(splits, pos_enc_dim):
    graphs = []
    for split in splits[0]:
        graphs.append(random_walk_encoding(split, pos_enc_dim))
    new_split = (graphs, splits[1])
    return new_split

def add_spd_encoding_CSL(splits):
    spatial_pos_list = []
    for split in splits[0]:
        spatial_pos_list.append(spd_encoding(split))
    new_split = (splits[0], splits[1], spatial_pos_list)
    return new_split

def add_spectral_decomposition_CSL(splits, pos_enc_dim):
    graphs = []
    for split in splits[0]:
        graphs.append(spectral_decomposition(split, pos_enc_dim))
    new_split = (graphs, splits[1])
    return new_split

def multiple_automaton_encodings_CSL(g, transition_matrix, initial_vector, idx=0, model=None):
    pe = automaton_encoding_CSL(g, transition_matrix, initial_vector, ret_pe=True, model=model)
    key = f'pos_enc_{idx}'
    g.ndata[key] = pe
    return g

def add_multiple_automaton_encodings_CSL(splits, model):
    transition_matrices = model.pe_layer.pos_transitions
    initial_vectors = model.pe_layer.pos_initials
    for i, (transition_matrix, initial_vector) in enumerate(zip(transition_matrices, initial_vectors)):
        graphs = []
        for g in splits[0]:
            graphs.append(multiple_automaton_encodings_CSL(g, transition_matrix, initial_vector, idx=i, model=model))
        new_split = (graphs, splits[1])
    return new_split

def automaton_encoding_CSL(g, transition_matrix, initial_vector, ret_pe=False, idx=0, model=None, matrix_type='A', prev_graph=None):
    transition_inv = transition_matrix.transpose(1, 0).cpu().numpy() # assuming the transition matrix is orthogonal
    matrix = g.adjacency_matrix().to_dense().cpu().numpy()
    mat = matrix

    if model.pe_layer.gape_normalize_mat:
        mat = mat @ sp.diags(dgl.backend.asnumpy(g.in_degrees()).clip(1) ** -1.0, dtype=float) # D^-1

    matrix = mat

    initial_vector = model.pe_layer.stack_strategy(g.number_of_nodes())

    initial_vector = initial_vector.detach().cpu().numpy()

    pe = scipy.linalg.solve_sylvester(transition_inv, -matrix, transition_inv @ initial_vector)
    pe = torch.from_numpy(pe.T).float()

    if ret_pe:
        return pe

    g.ndata['pos_enc'] = pe
    return g

def add_automaton_encodings_CSL(splits, model, matrix_type='A'):
    transition_matrix = model.pe_layer.pos_transitions[0]
    graphs = []
    prev_graph = None
    for i, split in enumerate(splits[0]):
        initial_vector = model.pe_layer.stack_strategy(split.num_nodes())
        # initial_vector = model.pe_layer.pos_initials[0]
        graphs.append(automaton_encoding_CSL(split, transition_matrix, initial_vector, False, i, model, matrix_type=matrix_type, prev_graph=prev_graph))
        prev_graph = split

    new_split = (graphs, splits[1])
    return new_split


def dump_encodings(dataset, pos_enc_dim):
    name = dataset.name
    if not os.path.exists(f'./{name}'):
        os.makedirs(f'./{name}')

    with open(f'./{name}/train_{pos_enc_dim}.pkl', 'wb+') as f:
        pickle.dump(dataset.train.graph_lists, f)

    with open(f'./{name}/val_{pos_enc_dim}.pkl', 'wb+') as f:
        pickle.dump(dataset.val.graph_lists, f)

    with open(f'./{name}/test_{pos_enc_dim}.pkl', 'wb+') as f:
        pickle.dump(dataset.test.graph_lists, f)


def load_encodings(dataset, pos_enc_dim):
    name = dataset.name
    with open(f'./{name}/train_{pos_enc_dim}.pkl', 'rb') as f:
        dataset.train.graph_lists = pickle.load(f)

    with open(f'./{name}/val_{pos_enc_dim}.pkl', 'rb') as f:
        dataset.val.graph_lists = pickle.load(f)

    with open(f'./{name}/test_{pos_enc_dim}.pkl', 'rb') as f:
        dataset.test.graph_lists = pickle.load(f)

    return dataset