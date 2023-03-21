import torch
import torch.nn as nn
import scipy as sp
import numpy as np
import networkx as nx
import dgl

from utils.main_utils import get_logger
from random import choices

def type_of_enc(net_params):
    learned_pos_enc = net_params.get('learned_pos_enc', False)
    pos_enc = net_params.get('pos_enc', False)
    adj_enc = net_params.get('adj_enc', False)
    rand_pos_enc = net_params.get('rand_pos_enc', False)
    partial_rw_pos_enc = net_params.get('partial_rw_pos_enc', False)
    spectral_attn = net_params.get('spectral_attn', False)
    n_gape = net_params.get('n_gape', 1)
    if learned_pos_enc:
        return 'learned_pos_enc'
    elif pos_enc:
        return 'pos_enc'
    elif adj_enc:
        return 'adj_enc'
    elif rand_pos_enc:
        return f'rand_pos_enc, using {str(n_gape)} automata/automaton'
    elif partial_rw_pos_enc:
        return 'partial_rw_pos_enc'
    elif spectral_attn:
        return 'spectral_attn'
    else:
        return 'no_pe'

class PELayer(nn.Module):
    def __init__(self, net_params):
        super().__init__()
        self.device = net_params['device']
        self.pos_enc = net_params.get('pos_enc', False)
        self.learned_pos_enc = net_params.get('learned_pos_enc', False)
        self.rand_pos_enc = net_params.get('rand_pos_enc', False)
        self.rw_pos_enc = net_params.get('rw_pos_enc', False) or net_params.get('partial_rw_pos_enc', False)
        self.adj_enc = net_params['adj_enc']
        self.pos_enc_dim = net_params.get('pos_enc_dim', 0)
        self.dataset = net_params.get('dataset', 'CYCLES')
        self.num_initials = net_params.get('num_initials', 1)
        self.cat = net_params.get('cat_gape', False)
        self.n_gape = net_params.get('n_gape', 1)
        self.gape_softmax_after = net_params.get('gape_softmax_after', False)
        self.gape_softmax_before = net_params.get('gape_softmax_before', False)
        self.gape_individual = net_params.get('gape_individual', False)
        self.matrix_type = net_params['matrix_type']
        self.logger = get_logger(net_params['log_file'])

        self.seed_array = net_params['seed_array']

        self.gape_scale = net_params.get('gape_scale', 1/40)
        self.gape_softmax_init = net_params.get('gape_softmax_init', False)
        self.gape_stack_strat = net_params.get('gape_stack_strat', '2')

        self.gape_stoch = net_params.get('gape_stoch', False)
        self.gape_beta = net_params.get('gape_beta', False)
        self.gape_weight_id = net_params.get('gape_weight_id', False)
        self.ngape_betas = net_params.get('ngape_betas', [])
        self.ngape_agg = net_params.get('ngape_agg', 'sum')

        self.gape_scalar = net_params.get('gape_scalar', False)
        if self.gape_scalar:
            self.scalar = nn.Parameter(torch.empty((1,)))
            nn.init.normal_(self.scalar)

        hidden_dim = net_params['hidden_dim']

        self.logger.info(type_of_enc(net_params))
        if self.pos_enc:
            self.embedding_pos_enc = nn.Linear(self.pos_enc_dim, hidden_dim)
        if self.adj_enc:
            self.embedding_pos_enc = nn.Linear(self.pos_enc_dim, hidden_dim)
        elif self.learned_pos_enc or self.rand_pos_enc:

            # init initial vectors

            self.pos_initials = nn.ParameterList(
                nn.Parameter(torch.empty(self.pos_enc_dim, 1, device=self.device), requires_grad=not self.rand_pos_enc)
                for _ in range(self.num_initials)
            )
            for pos_initial in self.pos_initials:
                nn.init.normal_(pos_initial)

            # init transition weights
            shape = (self.pos_enc_dim, self.pos_enc_dim)
            transitions = [torch.empty(*shape, requires_grad=not self.rand_pos_enc) for _ in range(self.n_gape)]

            for transition in transitions:
                torch.nn.init.orthogonal_(transition)

            # divide transition weights by norm or scalar
            modified_transitions = []
            for transition in transitions:
                mod_transition = transition
                if self.gape_scalar is not None and self.gape_scale != '0':
                    mod_transition = mod_transition * float(self.gape_scale)

                if self.gape_stoch:
                    mod_transition = torch.softmax(transition, dim=0)
                modified_transitions.append(mod_transition)

            self.pos_transitions = nn.ParameterList(
                nn.Parameter(mod_transition, requires_grad=not self.rand_pos_enc) for mod_transition in modified_transitions
            )

            if self.n_gape > 1:
                shape = (self.pos_enc_dim, self.pos_enc_dim)

                transition_matrices = []
                for transition in transitions:
                    if self.gape_scalar is not None and self.gape_scale != '0':
                        mod_transition = mod_transition * float(self.gape_scale)
                    # option for normalizing weights
                    if self.gape_stoch:
                        mod_transition = torch.softmax(transition, dim=0)

                    transition_matrices.append(mod_transition)

                self.pos_transitions = nn.ParameterList(nn.Parameter(transition, requires_grad=not self.rand_pos_enc) for transition in transition_matrices)

            # init linear layers for reshaping to hidden dim
            if self.gape_individual:
                self.embedding_pos_encs = nn.ModuleList(nn.Linear(self.pos_enc_dim, hidden_dim) for _ in range(self.n_gape))
            else:
                self.embedding_pos_encs = nn.ModuleList(nn.Linear(self.pos_enc_dim*int(self.n_gape), hidden_dim) for _ in range(1))

            if self.n_gape > 1:
                self.gape_pool_vec = nn.Parameter(torch.Tensor(self.n_gape, 1), requires_grad=True)
                nn.init.normal_(self.gape_pool_vec)


        if self.rw_pos_enc:
            self.embedding_pos_enc = nn.Linear(self.pos_enc_dim, hidden_dim) 

        self.use_pos_enc = self.pos_enc or self.learned_pos_enc or self.rand_pos_enc or self.adj_enc or self.rw_pos_enc
        if self.use_pos_enc:
            self.logger.info(f"Using {self.pos_enc_dim} dimension positional encoding")

        self.logger.info(f"Using matrix: {self.matrix_type}")


    def stack_strategy(self, num_nodes):
        num_pos_initials = len(self.pos_initials)
        try:
            num_nodes = num_nodes.number_of_nodes()
        except: pass

        if self.gape_stack_strat == "1":
            out = torch.cat([tensor for tensor in self.pos_initials[:num_nodes]], dim=1)     # pick top n, num_initials > n
            if self.gape_softmax_init:
                out = out.softmax(dim=1)

            return out

        indices = choices([i for i in range(num_pos_initials)], k=num_nodes)    # random n out of k

        out = torch.cat([self.pos_initials[i] for i in indices], dim=1)
        if self.gape_softmax_init:
            out = out.softmax(1)

        return out


    def forward(self, g, h, pos_enc=None, graph_lens=None):
        pe = pos_enc
        if not self.use_pos_enc:
            return h

        if self.rw_pos_enc:
            pe = self.embedding_pos_enc(pos_enc)
            return pe

        if self.pos_enc or self.adj_enc or self.rw_pos_enc:
            pe = self.embedding_pos_enc(pos_enc)
        elif self.learned_pos_enc:

            mat = self.type_of_matrix(g, self.matrix_type)
            vec_init = self.stack_strategy(g.num_nodes())
            vec_init = vec_init.transpose(1, 0).flatten()
            kron_prod = torch.kron(mat.reshape(mat.shape[1], mat.shape[0]), 0.02*self.pos_transitions[0].transpose(0, 1).contiguous()).to(self.device)

            B = torch.eye(kron_prod.shape[1]).to(self.device) - kron_prod
            encs = torch.linalg.solve(B, vec_init)

            stacked_encs = torch.stack(encs.split(self.pos_enc_dim), dim=1)
            stacked_encs = stacked_encs.transpose(1, 0)
            pe = self.embedding_pos_encs[0](stacked_encs)

            return pe

        elif self.rand_pos_enc:
            if self.gape_scalar:
                device = torch.device("cpu")
                vec_init = self.stack_strategy(g.num_nodes())
                mat = self.type_of_matrix(g, self.matrix_type)
                transition_inv = torch.inverse(self.pos_transitions[0]).to(device)

                # AX + XB = Q
                transition_inv = transition_inv.numpy()
                mat = mat.detach().cpu().numpy()
                vec_init = vec_init.cpu().numpy()
                pe = sp.linalg.solve_sylvester(transition_inv, -mat, transition_inv @ vec_init)
                pe = torch.from_numpy(pe.T).to(self.device)
            else:
                pe = pos_enc
            if self.n_gape > 1:
                pos_encs = [g.ndata[f'pos_enc_{i}'] for i in range(self.n_gape)]
                
                if self.gape_individual:
                    pos_encs = [self.embedding_pos_encs[i](pos_encs[i]) for i in range(self.n_gape)]

                if self.gape_softmax_before:
                    normalized_pos_encs = []
                    for pos_enc in pos_encs:
                        normalized_pos_enc = torch.softmax(pos_enc, dim=1)
                        normalized_pos_encs.append(normalized_pos_enc)
                    pos_encs = normalized_pos_encs

                pos_encs = torch.cat(pos_encs, dim=1)

                pe = pos_encs

                if self.gape_softmax_after:
                    pe = torch.softmax(pe, dim=1)

                if not self.gape_individual:
                    pe = self.embedding_pos_encs[0](pe)

            else:
                if not self.cat:
                    pe = self.embedding_pos_encs[0](pos_enc)

                if self.gape_softmax_after:
                    pe = torch.softmax(pe, dim=1)

            if self.gape_scalar:
                pe = self.scalar * pe
            return pe

        else:
            if self.dataset == "ZINC":
                pe = h
            elif self.dataset == "Cora":
                return h
            pe = self.embedding_h(h)

        return pe

    def get_normalized_laplacian(self, g):
        A = g.adjacency_matrix_scipy(return_edge_ids=False).astype(float)
        N = sp.sparse.diags(dgl.backend.asnumpy(g.in_degrees()).clip(1) ** -0.5, dtype=float)
        L = sp.sparse.eye(g.number_of_nodes()) - N * A * N
        return L

    def type_of_matrix(self, g, matrix_type):
        matrix = g.adjacency_matrix().to_dense().to(self.device)
        if matrix_type == 'A':
            matrix = g.adjacency_matrix().to_dense().to(self.device)
        elif matrix_type == 'NL':
            laplacian = self.get_normalized_laplacian(g)
            matrix = torch.from_numpy(laplacian.A).float().to(self.device) 
        elif matrix_type == "L":
            graph = g.cpu().to_networkx().to_undirected()
            matrix = torch.from_numpy(nx.laplacian_matrix(graph).A).to(self.device).type(torch.float32)
        elif matrix_type == "E":
            laplacian = self.get_normalized_laplacian(g)
            EigVal, EigVec = np.linalg.eig(laplacian.toarray())
            idx = EigVal.argsort() # increasing order
            EigVal, EigVec = EigVal[idx], np.real(EigVec[:,idx])
            matrix = torch.from_numpy(EigVec).float().to(self.device)

        return matrix

