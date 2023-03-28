import os, os.path
import logging
from datetime import timedelta
import subprocess

import numpy
import torch
import torch.nn as nn
import torch.nn.functional as F

import nmt.all_constants as ac
import networkx as nx
import scipy.sparse as sp
import scipy.linalg
import dgl
import layers


def get_logger(logfile=None):
    _logfile = logfile if logfile else './DEBUG.log'
    """Global logger for every logging"""
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        '%(asctime)s:%(filename)s:%(lineno)s - %(funcName)20s(): %(message)s')

    if not logger.handlers:
        debug_handler = logging.FileHandler(_logfile)
        debug_handler.setFormatter(formatter)
        debug_handler.setLevel(logging.DEBUG)
        logger.addHandler(debug_handler)

    return logger


def shuffle_file(input_file):
    shuffled_file = input_file + '.shuf'
    scriptdir = os.path.dirname(os.path.abspath(__file__))
    commands = 'bash {}/../scripts/shuffle_file.sh {} {}'.format(scriptdir, input_file, shuffled_file)
    subprocess.check_call(commands, shell=True)
    subprocess.check_call('mv {} {}'.format(shuffled_file, input_file), shell=True)


def get_validation_frequency(train_length_file, val_frequency, batch_size):
    with open(train_length_file) as f:
        line = f.readline().strip()
        num_train_toks = int(line)

    return int(num_train_toks * val_frequency / batch_size)


def format_seconds(seconds):
    return str(timedelta(seconds=seconds))


def get_vocab_masks(config, src_vocab_size, trg_vocab_size):
    masks = []
    for vocab_size, lang in [(src_vocab_size, config['src_lang']), (trg_vocab_size, config['trg_lang'])]:
        if config['tie_mode'] == ac.ALL_TIED:
            mask = numpy.load(os.path.join(config['data_dir'], 'joint_vocab_mask.{}.npy'.format(lang)))
        else:
            mask = numpy.ones([vocab_size], numpy.float32)

        mask[ac.PAD_ID] = 0.
        mask[ac.BOS_ID] = 0.
        masks.append(torch.from_numpy(mask).type(torch.uint8))

    return masks


def get_vocab_sizes(config):
    def _get_vocab_size(vocab_file):
        vocab_size = 0
        with open(vocab_file) as f:
            for line in f:
                if line.strip():
                    vocab_size += 1
        return vocab_size

    src_vocab_file = os.path.join(config['data_dir'], 'vocab-{}.{}'.format(config['src_vocab_size'], config['src_lang']))
    trg_vocab_file = os.path.join(config['data_dir'], 'vocab-{}.{}'.format(config['trg_vocab_size'], config['trg_lang']))

    return _get_vocab_size(src_vocab_file), _get_vocab_size(trg_vocab_file)

def get_sine_encoding(dim, sentence_length):
    div_term = numpy.power(10000.0, - (numpy.arange(dim) // 2).astype(numpy.float32) * 2.0 / dim)
    div_term = div_term.reshape(1, -1)
    pos = numpy.arange(sentence_length, dtype=numpy.float32).reshape(-1, 1)
    encoded_vec = numpy.matmul(pos, div_term)
    encoded_vec[:, 0::2] = numpy.sin(encoded_vec[:, 0::2])
    encoded_vec[:, 1::2] = numpy.cos(encoded_vec[:, 1::2])

    dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
    return torch.from_numpy(encoded_vec.reshape([sentence_length, dim])).type(dtype)

def get_cycle_graph_lapes(dim, sentence_length):
    """Compute laplacian eigenvectors for large cycle graph (unnormalized Laplacian)"""
    big_length = int(1e6)
    cos_encs = []
    sine_encs = []
    for i in range((big_length//2)+1):
        cos_enc = numpy.cos((2*numpy.pi*i/big_length)*numpy.arange(dim)).reshape(-1,1)
        if i != 0 and i != big_length//2:
            sin_enc = numpy.sin((2*numpy.pi*i/big_length)*numpy.arange(dim)).reshape(-1,1)
            sine_encs.append(sin_enc)
        cos_encs.append(cos_enc)
    cos_encs = numpy.concatenate(cos_encs, axis=1)
    sine_encs = numpy.concatenate(sine_encs, axis=1)
    cos_prefix = cos_encs[:, :(big_length//2)-1]
    evec_prefix = numpy.vstack((cos_prefix, sine_encs)).reshape(dim, big_length-2, order='F')
    evecs = numpy.concatenate((evec_prefix, cos_encs[:, (big_length)//2-1:big_length]), axis=1)
    dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
    return torch.from_numpy(evecs[:dim, :sentence_length].T).type(dtype)

def get_lape_encoding(dim, sentence_length, graph_size=None):
    p_n = nx.cycle_graph(graph_size if graph_size else sentence_length)
    laplacian = nx.normalized_laplacian_matrix(p_n)
    evals, evecs = numpy.linalg.eig(laplacian.A)
    idx = evals.argsort()
    evals, evecs = evals[idx], numpy.real(evecs[:, idx])
    dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
    if graph_size is not None:
        evecs = evecs[:sentence_length,1:dim+1]
        numpy.random.shuffle(evecs)
        pos_enc = torch.from_numpy(evecs).type(dtype)
    else:
        pos_enc = torch.from_numpy(evecs[:, 1:dim+1]).type(dtype)

def get_rw_pos(dim, sentence_length):
    g = dgl.from_networkx(nx.path_graph(sentence_length))
    A = g.adjacency_matrix(scipy_fmt="csr")
    Dinv = sp.diags(dgl.backend.asnumpy(g.in_degrees()).clip(1) ** -1.0, dtype=float) # D^-1
    RW = A * Dinv  
    M = RW

    PE = [torch.from_numpy(M.diagonal()).float()]
    M_power = M
    for _ in range(dim-1):
        M_power = M_power * M
        PE.append(torch.from_numpy(M_power.diagonal()).float())

    PE = torch.stack(PE, dim=-1)
    dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

    pos_enc = PE.type(dtype)

    return pos_enc

def normalize(x, scale=True):
    mean = x.mean(-1, keepdim=True)
    std = x.std(-1, keepdim=True) + 1e-6
    if scale:
        std = std * x.size()[-1] ** 0.5
    return (x - mean) / std


def gnmt_length_model(alpha):
    def f(time_step, prob):
        return prob / ((5.0 + time_step + 1.0) ** alpha / 6.0 ** alpha)
    return f

def get_laplacian_eigs(sentence_length):
    # g = dgl.from_networkx(nx.cycle_graph(sentence_length))
    g = dgl.from_networkx(nx.path_graph(sentence_length))
    n = g.number_of_nodes()
    deg = g.in_degrees()
    A = g.adjacency_matrix_scipy(return_edge_ids=False).astype(float)
    N = sp.diags(dgl.backend.asnumpy(deg).clip(1) ** -0.5, dtype=float)
    L = sp.eye(n) - N * A * N
    evals, evecs = numpy.linalg.eigh(L.toarray())
    return evals, evecs

class LAPE:
    def __init__(self, config):
        super(LAPE, self).__init__()
        self.evals, self.evecs = get_laplacian_eigs(config['max_pos_length'])

    def get_lape_encoding(self, dim, sentence_length, graph_size=None):
        # g = nx.cycle_graph(graph_size if graph_size else sentence_length)
        # laplacian = nx.normalized_laplacian_matrix(g)
        # evals, evecs = numpy.linalg.eig(laplacian.A)
        evals, evecs = self.evals, self.evecs
        idx = evals.argsort()
        evals, evecs = evals[idx], numpy.real(evecs[:, idx])
        dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
        if graph_size is not None:
            evecs = evecs[:sentence_length,1:dim+1]
            numpy.random.shuffle(evecs)
            pos_enc = torch.from_numpy(evecs).type(dtype)
        else:
            pos_enc = torch.from_numpy(evecs[:, 1:dim+1]).type(dtype)
        return pos_enc

class SpectralAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        embed_dim = config['spectral_embed_dim']
        lpe_n_heads = config['lpe_n_heads']
        lpe_n_layers = config['lpe_n_layers']
        lpe_ff_dim = config['lpe_ff_dim']
        self.spectral_dim = config['spectral_dim'] # num frequencies
        self.max_pos_length = config['max_pos_length']
        eigvals, eigvecs = get_laplacian_eigs(self.max_pos_length)
        self.eigvals = eigvals
        self.eigvecs = eigvecs

        # encoder_layer = nn.TransformerEncoderLayer(embed_dim, lpe_n_heads)
        # self.lpe_attn = nn.TransformerEncoder(encoder_layer, lpe_n_layers)
        self.lpe_attn = layers.encoders.Encoder(lpe_n_layers, lpe_n_heads, embed_dim, lpe_ff_dim)
        self.linear = nn.Linear(2, embed_dim)

    def forward(self, sentence_length):
        dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

        # eigvals, eigvecs = get_laplacian_eigs(sentence_length)
        # eigvals, eigvecs = get_laplacian_eigs(self.max_pos_length)
        eigvals = self.eigvals[: self.spectral_dim]
        eigvecs = self.eigvecs[:, :self.spectral_dim]

        eigvecs = torch.from_numpy(eigvecs).float().type(dtype)
        eigvecs = F.normalize(eigvecs, p=2, dim=1, eps=1e-12, out=None)
        eigvecs = eigvecs.unsqueeze(2)

        eigvals = torch.from_numpy(numpy.sort(numpy.abs(numpy.real(eigvals)))).type(dtype) #Abs value is taken because numpy sometimes computes the first eigenvalue approaching 0 from the negative
        eigvals = eigvals.unsqueeze(0)
        eigvals = eigvals.repeat(self.max_pos_length, 1).unsqueeze(2)

        lpe = torch.cat((eigvecs, eigvals), dim=2) # (Num nodes) x (Num Eigenvectors) x 2
        empty_mask = torch.isnan(lpe) # (Num nodes) x (Num Eigenvectors) x 2

        lpe[empty_mask] = 0 # (Num nodes) x (Num Eigenvectors) x 2
        lpe = torch.transpose(lpe, 0 ,1) # (Num Eigenvectors) x (Num nodes) x 2
        lpe = self.linear(lpe) # (Num Eigenvectors) x (Num nodes) x PE_dim

        # lpe = self.lpe_attn(lpe, empty_mask[:,:,0])
        lpe = self.lpe_attn(lpe, None)

        #remove masked sequences
        lpe[torch.transpose(empty_mask, 0 ,1)[:,:,0]] = float('nan')

        #Sum pooling
        nan = torch.isnan(lpe)
        lpe[nan] = 0
        lpe = torch.sum(lpe, 0, keepdim=False)
        # lpe = torch.nansum(lpe, 0, keepdim=False)

        return lpe

class AutomatonPELayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_states = config['num_states']
        self.directed = config['directed']
        embed_dim = config['embed_dim']

        self.embedding_pos_enc = nn.Linear(self.num_states, embed_dim)
        self.pos_initial = nn.Parameter(torch.Tensor(self.num_states, 1), requires_grad=True)
        self.pos_transition = nn.Parameter(torch.Tensor(self.num_states, self.num_states), requires_grad=True)

        nn.init.normal_(self.pos_initial)
        nn.init.orthogonal_(self.pos_transition)

    def get_pe(self, max_len):
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        ones = torch.ones(max_len-1)
        if self.directed:
            adj = torch.diag(ones, 1)
        else:
            adj = torch.diag(ones, -1) + torch.diag(ones, 1)
        transition_inv = torch.inverse(self.pos_transition).detach().cpu().numpy()
        adj = adj.cpu().numpy()

        z = torch.zeros(self.num_states, max_len-1, requires_grad=False, device=torch.device('cpu'))
        vec_init = torch.cat((self.pos_initial, z), dim=1)
        vec_init = vec_init.detach().cpu().numpy()

        pe = scipy.linalg.solve_sylvester(transition_inv, -adj, transition_inv @ vec_init)
        pe = torch.from_numpy(pe.T).to(device)

        return pe

    def forward(self, sentence_len):
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        ones = torch.ones(sentence_len-1)
        if self.directed:
            adj = torch.diag(ones, 1)
        else:
            adj = torch.diag(ones, -1) + torch.diag(ones, 1)

        transition_inv = torch.inverse(self.pos_transition).detach().cpu().numpy()
        adj = adj.cpu().numpy()

        z = torch.zeros(self.num_states, sentence_len-1, requires_grad=False, device=device)
        vec_init = torch.cat((self.pos_initial, z), dim=1)
        vec_init = vec_init.detach().cpu().numpy()

        pe = scipy.linalg.solve_sylvester(transition_inv, -adj, transition_inv @ vec_init)
        pe = torch.from_numpy(pe.T).to(device)
        pe = self.embedding_pos_enc(pe)

        return pe
