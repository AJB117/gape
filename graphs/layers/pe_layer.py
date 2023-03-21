import torch
import torch.nn as nn
import scipy as sp
import numpy as np
import networkx as nx
import dgl
import scipy

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
        self.pow_of_mat = net_params.get('pow_of_mat', 1)
        self.num_initials = net_params.get('num_initials', 1)
        self.cat = net_params.get('cat_gape', False)
        self.n_gape = net_params.get('n_gape', 1)
        self.gape_pooling = net_params.get('gape_pooling', 'mean')
        self.gape_softmax_after = net_params.get('gape_softmax_after', False)
        self.gape_softmax_before = net_params.get('gape_softmax_before', False)
        self.gape_individual = net_params.get('gape_individual', False)
        self.clamp = net_params.get('gape_clamp', False)
        self.diag = net_params.get('diag', False)
        self.matrix_type = net_params['matrix_type']
        self.logger = get_logger(net_params['log_file'])

        self.experiment_1 = net_params.get('experiment_1', False)
        self.gape_normalization = net_params.get('gape_normalization', 'none')
        self.gape_squash = net_params.get('gape_squash', 'none')
        self.seed_array = net_params['seed_array']

        self.power_method = net_params.get('power_method', False)
        self.power_method_iters = net_params.get('power_iters', 50)

        self.rand_sketchy_pos_enc = net_params.get('rand_sketchy_pos_enc', False)

        self.gape_norm = net_params.get('gape_norm', False)
        self.gape_div = net_params.get('gape_div', False)
        self.gape_scale = net_params.get('gape_scale', 1/40)
        self.gape_weight_gen = net_params.get('gape_weight_gen', False)
        self.gape_softmax_init = net_params.get('gape_softmax_init', False)
        self.gape_uniform_init = net_params.get('gape_uniform_init', False)
        self.gape_stack_strat = net_params.get('gape_stack_strat', '2')

        self.gape_normalize_mat = net_params.get('gape_normalize_mat', False)

        self.gape_symmetric = net_params.get('gape_symmetric', False)
        self.gape_stoch = net_params.get('gape_stoch', False)
        self.gape_tau = net_params.get('gape_tau', False)
        self.gape_tau_mat = net_params.get('gape_tau_mat', False)
        self.gape_beta = net_params.get('gape_beta', False)
        self.gape_weight_id = net_params.get('gape_weight_id', False)
        self.gape_break_batch = net_params.get('gape_break_batch', False)
        self.ngape_betas = net_params.get('ngape_betas', [])
        self.gape_cond_lbl = net_params.get('gape_cond_lbl', False)
        self.ngape_agg = net_params.get('ngape_agg', 'sum')

        self.eigen_bartels_stewart = net_params.get('eigen_bartels_stewart', False)
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
        elif self.learned_pos_enc or self.rand_pos_enc or self.rand_sketchy_pos_enc:

            if self.eigen_bartels_stewart and self.gape_tau_mat:
                self.stop_vec = nn.Parameter(torch.empty(self.pos_enc_dim, device=self.device), requires_grad=self.learned_pos_enc)
                nn.init.normal_(self.stop_vec)

            # init initial vectors

            self.pos_initials = nn.ParameterList(
                nn.Parameter(torch.empty(self.pos_enc_dim, 1, device=self.device), requires_grad=not self.rand_pos_enc and not self.rand_sketchy_pos_enc)
                for _ in range(self.num_initials)
            )
            for pos_initial in self.pos_initials:
                nn.init.normal_(pos_initial)

            # init transition weights
            if self.diag:
                shape = (self.pos_enc_dim,)
            elif self.gape_cond_lbl and self.num_initials > 1:
                shape = (self.pos_enc_dim, self.pos_enc_dim, self.num_initials)
            else:
                shape = (self.pos_enc_dim, self.pos_enc_dim)

            transitions = [torch.empty(*shape, requires_grad=not self.rand_pos_enc and not self.rand_sketchy_pos_enc) for _ in range(self.n_gape)]

            for transition in transitions:
                if self.diag:
                    torch.nn.init.normal_(transition)
                else:
                    torch.nn.init.orthogonal_(transition)

            # divide transition weights by norm or scalar
            modified_transitions = []
            for transition in transitions:
                mod_transition = transition
                if self.gape_norm:
                    mod_transition = transition / torch.linalg.norm(transition)
                elif self.gape_scalar is not None and self.gape_scale != '0':
                    mod_transition = mod_transition * float(self.gape_scale)

                if self.gape_stoch:
                    mod_transition = torch.softmax(transition, dim=0)
                modified_transitions.append(mod_transition)

            # store matrices or vectors depending whether diag
            # if not self.diag:
            self.pos_transitions = nn.ParameterList(
                nn.Parameter(mod_transition, requires_grad=not self.rand_pos_enc and not self.rand_sketchy_pos_enc) for mod_transition in modified_transitions
            )

            if self.n_gape > 1:
                shape = (self.pos_enc_dim,) if net_params['diag'] else (self.pos_enc_dim, self.pos_enc_dim)

                transition_matrices = []
                for transition in transitions:
                    if self.gape_norm:
                        mod_transition = transition / torch.linalg.norm(transition)
                    elif self.gape_scalar is not None and self.gape_scale != '0':
                        mod_transition = mod_transition * float(self.gape_scale)
                    # option for normalizing weights
                    if self.gape_stoch:
                        mod_transition = torch.softmax(transition, dim=0)

                    transition_matrices.append(mod_transition)

                self.pos_transitions = nn.ParameterList(nn.Parameter(transition, requires_grad=not self.rand_pos_enc and not self.rand_sketchy_pos_enc) for transition in transition_matrices)

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

        self.use_pos_enc = self.pos_enc or self.learned_pos_enc or self.rand_pos_enc or self.adj_enc or self.rw_pos_enc or self.rand_sketchy_pos_enc
        if self.use_pos_enc:
            self.logger.info(f"Using {self.pos_enc_dim} dimension positional encoding")

        self.logger.info(f"Using matrix: {self.matrix_type}")
        self.logger.info(f"Matrix power: {self.pow_of_mat}")
        if self.power_method:
            self.logger.info(f"Using power method with {self.power_method_iters} iterations")
        
        self.transition_mul_mat = nn.Parameter(torch.Tensor(self.pos_enc_dim, self.pos_enc_dim), requires_grad=True)
        nn.init.normal_(self.transition_mul_mat)
        self.pos_adder = nn.Parameter(torch.Tensor(self.pos_enc_dim, 1), requires_grad=True)
        nn.init.normal_(self.pos_adder)

    def random_walk_encoding(self, g, pos_enc_dim):
        """
        Graph positional encoding w/ random walk
        """

        # Geometric diffusion features with Random Walk
        A = g.adjacency_matrix(scipy_fmt="csr")
        Dinv = sp.sparse.diags(dgl.backend.asnumpy(g.in_degrees()).clip(1) ** -1.0, dtype=float) # D^-1
        RW = A * Dinv  
        M = RW

        nb_pos_enc = pos_enc_dim
        PE = [torch.from_numpy(M.diagonal()).float()]
        M_power = M
        for _ in range(nb_pos_enc-1):
            M_power = M_power * M
            PE.append(torch.from_numpy(M_power.diagonal()).float())

        PE = torch.stack(PE,dim=-1)

        return PE

    def stack_strategy(self, num_nodes):
        num_pos_initials = len(self.pos_initials)
        try:
            num_nodes = num_nodes.number_of_nodes()
        except: pass

        if self.gape_stack_strat == "1":
            out = torch.cat([tensor for tensor in self.pos_initials[:num_nodes]], dim=1)     # pick top n, num_initials > n
            if self.gape_softmax_init:
                out = out.softmax(dim=1)
            elif self.gape_uniform_init:
                out = torch.full(out.shape, 1/self.pos_enc_dim)

            return out
        # if self.out is None:
        #     options = [i for i in range(num_pos_initials)]
        #     indices = choices(options, k=num_nodes)
        # else:
        # options, indices = self.out[0], self.out[1]
        indices = choices([i for i in range(num_pos_initials)], k=num_nodes)    # random n out of k

        # if not num_nodes in self.out:
        #     out = torch.cat([torch.clone(self.pos_initials[i]) for i in indices], dim=1)
        #     self.out[num_nodes] = out
            # return out
        # self.out = [options, indices]
        out = torch.cat([self.pos_initials[i] for i in indices], dim=1)
        if self.gape_softmax_init:
            out = out.softmax(1)
        elif self.gape_uniform_init:
            out = torch.full(out.shape, 1/self.pos_enc_dim)

        return out
        # return self.out[num_nodes]
        """
            Given more than one initial weight vector, define the stack strategy.

            If n = number of nodes and k = number of weight vectors,
                by default, we repeat each initial weight vector n//k times
                and stack them together with final n-(n//k) weight vectors.
        """
        try:
            num_nodes = num_nodes.number_of_nodes()
        except: pass
        num_pos_initials = len(self.pos_initials)
        if num_pos_initials == 1:
            out = torch.cat([torch.clone(self.pos_initials[0]) for _ in range(num_nodes)], dim=1)
            return out
        remainder = num_nodes % num_pos_initials
        capacity = num_nodes - remainder

        out = torch.cat([torch.clone(self.pos_initials[i]) for i in range(num_pos_initials)], dim=1)
        out = torch.repeat_interleave(out, capacity//num_pos_initials, dim=1)
        if remainder != 0:
            remaining_stack = torch.cat([self.pos_initials[-1] for _ in range(remainder)], dim=1)
            out = torch.cat([out, remaining_stack], dim=1)
        return out

    def kronecker(self, mat1, mat2):
        return torch.einsum('ab,cd->acbd', mat1, mat2).reshape(mat1.shape[0] * mat2.shape[0], mat1.shape[1] * mat2.shape[1])


    def sylvester(self, A, B, C):
        m = B.shape[-1]
        n = A.shape[-1]
        try:
            R, U = torch.linalg.eigh(A)
            S, V = torch.linalg.eigh(B)
            U = U.type(torch.complex64)
            V = V.type(torch.complex64)
        except:
            R, U = torch.linalg.eig(A)
            S, V = torch.linalg.eig(B)

        F = torch.linalg.solve(U, (C + 0j) @ V)
        W = R[..., :, None] - S[..., None, :]
        Y = F / W
        X = U[...,:n,:n] @ Y[...,:n,:m] @ torch.linalg.inv(V)[...,:m,:m]
        return X

    def spectral_radius(self, matrix):
        return torch.abs(torch.real(matrix)).max()

    def break_batch(self, g, graph_lens):
        adj = g.adjacency_matrix_scipy().todense().astype(float)
        prev, until = 0, 0

        for graph_len in graph_lens:
            until += graph_len
            graph_adj = adj[prev:until, prev:until]
            prev = until

            src, dst = np.nonzero(graph_adj)
            new_graph = dgl.graph((src, dst))
            yield new_graph

    def compute_learned_gape(self, g):
        mat = self.type_of_matrix(g, self.matrix_type).to(self.device)
        if self.gape_normalize_mat:
            A = g.adjacency_matrix_scipy(return_edge_ids=False).astype(float)
            D = sp.sparse.diags(dgl.backend.asnumpy(g.in_degrees()).clip(1) ** -1.0, dtype=float)
            # mat = torch.from_numpy((A * D).todense()).to(self.device).type(torch.float)
            mat = torch.from_numpy((D * A).todense()).to(self.device).type(torch.float)
            # N = sp.sparse.diags(dgl.backend.asnumpy(g.in_degrees()).clip(1) ** -0.5, dtype=float)
            # mat = torch.from_numpy((N * A * N).todense()).to(self.device).type(torch.float)
            # mat = torch.from_numpy(A.todense()).to(self.device).type(torch.float)
            # mat = mat * 1.1

        if self.gape_tau_mat:
            stop_vec = torch.softmax(self.stop_vec, dim=0)
            stop_diag = torch.eye(self.pos_enc_dim, device=self.device) - torch.diag(stop_vec)

        if not self.ngape_betas:
            betas = [float(self.gape_beta)]
        else:
            betas = [float(beta) for beta in self.ngape_betas]

        pes = []

        for beta in betas:
            if beta < 1:
                mat = mat * (1-beta)

            vec_init = self.stack_strategy(g.number_of_nodes()).to(self.device)

            if self.gape_weight_id:
                vec_init = torch.zeros_like(vec_init)
                vec_init.fill_diagonal_(1)

            if self.gape_tau_mat:
                vec_init = torch.diag(stop_vec) @ vec_init

            if beta < 1:
                vec_init = vec_init * beta # emulate pagerank

            transition = self.pos_transitions[0]
            transition = transition.transpose(0, 1)
            if self.gape_stoch:
                transition = torch.softmax(transition, dim=0)

            if self.gape_symmetric:
                triu = torch.triu(transition)
                values = triu[triu != 0]
                i, j = torch.triu_indices(self.pos_enc_dim, self.pos_enc_dim)
                triu[i, j] = values
                triu.T[i, j] = values
                transition = triu

            if self.gape_tau_mat:
                A = torch.linalg.inv(transition.transpose(0, 1) @ stop_diag)
                B = -mat
                C = A.clone() @ vec_init
            else:
                transition_inverse = torch.linalg.inv(transition).to(self.device)

                B = -mat
                if self.gape_tau_mat:
                    A = torch.linalg.inv(transition @ stop_diag) # insert stopping probabilities
                    C = transition_inverse @ (vec_init @ torch.diag(stop_vec))
                else:
                    A = transition_inverse
                    C = transition_inverse @ vec_init

            # A = torch.linalg.inv(transition.transpose(0, 1) @ stop_diag)
            # B = -mat
            # C = A.clone() @ vec_init

            A = torch.nan_to_num(A)
            B = torch.nan_to_num(B)
            C = torch.nan_to_num(C)
            pe = self.sylvester(A, B, C)

            if self.gape_tau:
                pe = torch.mul(pe, vec_init)

            pe = pe.transpose(1, 0).type(torch.float32)
            pe = torch.real(pe)
            pes.append(pe)

        if len(pes) > 1:
            pe = sum(pes)
        else:
            pe = pes[0]

        pe = self.embedding_pos_encs[0](pe)

        if self.gape_softmax_after:
            pe = torch.softmax(pe, dim=1)

        if self.clamp:
            pe = torch.tanh(pe)

        return pe


    def learned_forward(self, g, graph_lens=None):
        if self.gape_break_batch:
            pes = []
            for graph in self.break_batch(g, graph_lens):
                pe = self.compute_learned_gape(graph)
                pes.append(pe)
            return pes

        return self.compute_learned_gape(g)


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

            if self.eigen_bartels_stewart:
                pes = self.learned_forward(g, graph_lens=graph_lens)

                if self.gape_break_batch:
                    return torch.cat(pes, dim=0)

                return pes

            mat = self.type_of_matrix(g, self.matrix_type)
            vec_init = self.stack_strategy(g.num_nodes())
            vec_init = vec_init.transpose(1, 0).flatten()
            kron_prod = torch.kron(mat.reshape(mat.shape[1], mat.shape[0]), 0.02*self.pos_transitions[0].transpose(0, 1).contiguous()).to(self.device)

            if self.power_method:
                encs = vec_init
                lam = 0.25
                for _ in range(self.power_method_iters):
                    encs = ((lam*kron_prod) @ encs) + vec_init
                    # norm = torch.norm(encs, p=2, dim=0)
                    # encs = encs / norm
                    # encs = encs.clamp(min=-1, max=1)
            else:
                B = torch.eye(kron_prod.shape[1]).to(self.device) - kron_prod
                encs = torch.linalg.solve(B, vec_init)

            stacked_encs = torch.stack(encs.split(self.pos_enc_dim), dim=1)
            stacked_encs = stacked_encs.transpose(1, 0)
            pe = self.embedding_pos_encs[0](stacked_encs)

            if self.clamp:
                # pe = pe.clamp(-3, 3)
                pe = torch.tanh(pe)

            return pe
        elif self.rand_sketchy_pos_enc:
            mat = self.type_of_matrix(g, self.matrix_type)
            initial_vector = torch.cat([self.pos_initials[0] for _ in range(mat.shape[0])], dim=1)

            initial_adder = torch.cat([self.pos_adder for _ in range(mat.shape[0])], dim=1)
            initial_vector = (initial_vector + initial_adder).detach().cpu().numpy()

            self.pos_transitions[0] = self.pos_transitions[0] + self.transition_mul_mat
            transition_inv = torch.inverse(self.pos_transitions[0]).detach().cpu().numpy()
            mat_product = torch.inverse(self.pos_transitions[0]).detach().cpu() @ initial_vector
            mat_product = mat_product.cpu().numpy()

            mat = mat.cpu().numpy()
            pe = scipy.linalg.solve_sylvester(transition_inv, -mat, mat_product)
            pe = torch.from_numpy(pe.T).float().to(self.device)
            return self.embedding_pos_encs[0](pe)

        elif self.rand_pos_enc:
            if self.pow_of_mat > 1 or self.gape_scalar:
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

            if self.clamp:
                pe = torch.tanh(pe)

            pre_modified = pe

            if self.gape_squash == 'softplus':
                pe = torch.nn.functional.softplus(pe)
            if self.gape_squash == 'exp':
                pe = torch.exp(pe)
            if self.gape_squash == 'square':
                pe = torch.mul(pe, pe)
            if self.gape_squash == 'tanh':
                pe = torch.tanh(pe)

            if self.gape_normalization == 'max':
                pe = pe / pe.max()
            if self.gape_normalization == 'softmax_0':
                pe = torch.softmax(pe, dim=0)
            if self.gape_normalization == 'softmax_1':
                pe = torch.softmax(pe, dim=1)

            if self.experiment_1:
                try:
                    pes = torch.load(f'./data/{self.dataset}_{self.gape_squash}_{self.gape_normalization}_{self.seed_array[0]}.pt')
                    pes.append((pe, pre_modified, g))
                except:
                    pes = [(pe, pre_modified, g)]
                torch.save(pes, f'./data/{self.dataset}_{self.gape_squash}_{self.gape_normalization}_{self.seed_array[0]}.pt')

            if self.gape_scalar:
                pe = self.scalar * pe
            return pe

        else:
            if self.dataset == "ZINC":
                pe = h
            elif self.dataset == "Cora":
                return h
            pe = self.embedding_h(h)

        # pe = torch.dropout(pe, p=0.05)
        return pe

    def get_normalized_laplacian(self, g):
        A = g.adjacency_matrix_scipy(return_edge_ids=False).astype(float)
        N = sp.sparse.diags(dgl.backend.asnumpy(g.in_degrees()).clip(1) ** -0.5, dtype=float)
        L = sp.sparse.eye(g.number_of_nodes()) - N * A * N
        return L

    def type_of_matrix(self, g, matrix_type):
        """
        Takes a DGL graph and returns the type of matrix to use for the layer.
            'A': adjacency matrix (default),
            'L': Laplacian matrix,
            'NL': normalized Laplacian matrix,
            'E': eigenvector matrix,
        """
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

        if self.pow_of_mat > 1:
            matrices = [torch.matrix_power(matrix, i) for i in range(self.pow_of_mat)]

            # multiply each matrix by the corresponding mat_pow
            for i in range(len(matrices)):
                matrices[i] = matrices[i] * self.mat_pows[i]
            matrix = torch.sum(torch.stack(matrices, dim=0), dim=0)

        return matrix

