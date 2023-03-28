import torch
from torch import nn
from torch.nn import Parameter
import torch.nn.functional as F
from layers import Encoder, Decoder
import nmt.all_constants as ac
import nmt.utils as ut
import networkx as nx
import numpy as np


class Model(nn.Module):
    """Model"""
    def __init__(self, config):
        super(Model, self).__init__()
        self.config = config

        self.init_embeddings()
        self.init_model()

    def init_embeddings(self):
        embed_dim = self.config['embed_dim']
        tie_mode = self.config['tie_mode']
        fix_norm = self.config['fix_norm']
        max_pos_length = self.config['max_pos_length']
        learned_pos = self.config['learned_pos']
        sine_pos = self.config['sine_pos']
        lape_pos = self.config['lape_pos']
        spectral_attn = self.config['spectral_attn']
        rw_pos = self.config['rw_pos']
        spd_centrality = self.config['spd_centrality']

        self.automaton = self.config['automaton']
        self.lape_pos = lape_pos
        self.spectral_attn = spectral_attn
        self.rw_pos = rw_pos
        self.spd_centrality = spd_centrality
        self.max_pos_length = max_pos_length

        self.graph_size = self.config.get('graph_size', None)

        if self.rw_pos:
            rw_pos_dim = self.config['rw_pos_dim']
            self.rw_pos_emb = nn.Linear(rw_pos_dim, embed_dim)

        if self.spd_centrality:
            path_embed_dim = self.config['path_embed_dim']
            centrality_embed_dim = self.config['centrality_embed_dim']
            self.path_embed = nn.Embedding(path_embed_dim, 1)
            self.centrality_embed = nn.Embedding(centrality_embed_dim, embed_dim)
            self.path_graph = nx.path_graph(self.config['path_embed_dim'])

            try:
                print('loading shortest paths from disk...')
                self.spatial_pos = torch.load('./spatial_pos.pt').to(torch.device('cuda'))
            except:
                print('computing shortest paths and saving to disk...')
                self.spatial_pos = {}
                self.shortest_paths = nx.floyd_warshall(self.path_graph)
                self.spatial_pos = [[-1]*len(self.shortest_paths) for _ in range(len(self.shortest_paths))]
                for src, trg_dict in self.shortest_paths.items():
                    for trg, distance in trg_dict.items():
                        self.spatial_pos[src][trg] = distance
                        self.spatial_pos[trg][src] = distance
                self.spatial_pos = torch.from_numpy(np.array(self.spatial_pos)).type(torch.long).to(torch.device('cuda'))
                torch.save(self.spatial_pos, "./spatial_pos.pt")

        self.big_graph_ul = self.config.get('big_graph_ul', None)

        # get positonal embedding
        # if not learned_pos:
        #     self.pos_embedding = ut.get_positional_encoding(embed_dim, max_pos_length)
        # else:
        #     self.pos_embedding = Parameter(torch.Tensor(max_pos_length, embed_dim))
        #     nn.init.normal_(self.pos_embedding, mean=0, std=embed_dim ** -0.5)
        if learned_pos:
            ut.get_logger().info('using learned positional embedding')
            self.pos_embedding = Parameter(torch.Tensor(max_pos_length, embed_dim))
            nn.init.normal_(self.pos_embedding, mean=0, std=embed_dim ** -0.5)
        elif sine_pos:
            ut.get_logger().info('using sine positional embedding')
            self.pos_embedding = ut.get_sine_encoding(embed_dim, max_pos_length)
        elif lape_pos:
            ut.get_logger().info('Using lape positional embedding')
            # self.pos_embedding = ut.get_lape_encoding(embed_dim, max_pos_length, self.graph_size)
            self.LAPE = ut.LAPE(self.config)
            self.pos_embedding = self.LAPE.get_lape_encoding(dim=self.config['lape_dim'], sentence_length=self.config['max_pos_length'])
        elif spectral_attn:
            ut.get_logger().info('Using spectral positional embedding')
            self.spectral_embedding = ut.SpectralAttention(self.config)
        elif self.big_graph_ul:
            ut.get_logger().info('using big cycle graph positional embedding (unnormalized Laplacian)')
            self.pos_embedding = ut.get_cycle_graph_lapes(embed_dim, max_pos_length)
        elif self.rw_pos:
            ut.get_logger().info('Using random walk positional embedding')
            self.pos_embedding = ut.get_rw_pos(rw_pos_dim, max_pos_length)
        elif self.spd_centrality:
            ut.get_logger().info('Using shortest-path distance + node centrality embedding')
        elif self.automaton:
            ut.get_logger().info(f"using automaton encoding, directed: {self.config['directed']}, num_states: {self.config['num_states']}")
            self.pos_embedding = ut.AutomatonPELayer(self.config).get_pe(self.config['max_pos_length'])

        # get word embeddings
        src_vocab_size, trg_vocab_size = ut.get_vocab_sizes(self.config)
        self.src_vocab_mask, self.trg_vocab_mask = ut.get_vocab_masks(self.config, src_vocab_size, trg_vocab_size)
        if tie_mode == ac.ALL_TIED:
            src_vocab_size = trg_vocab_size = self.trg_vocab_mask.shape[0]

        self.out_bias = Parameter(torch.Tensor(trg_vocab_size))
        nn.init.constant_(self.out_bias, 0.)

        self.src_embedding = nn.Embedding(src_vocab_size, embed_dim)
        self.trg_embedding = nn.Embedding(trg_vocab_size, embed_dim)
        self.out_embedding = self.trg_embedding.weight
        self.embed_scale = embed_dim ** 0.5

        if spectral_attn:
            self.diet_linear = nn.Linear(embed_dim, embed_dim - self.config['spectral_embed_dim'])

        if lape_pos:
            self.lape_emb = nn.Linear(self.config['lape_dim'], embed_dim)

        if tie_mode == ac.ALL_TIED:
            self.src_embedding.weight = self.trg_embedding.weight

        if not fix_norm:
            nn.init.normal_(self.src_embedding.weight, mean=0, std=embed_dim ** -0.5)
            nn.init.normal_(self.trg_embedding.weight, mean=0, std=embed_dim ** -0.5)
        else:
            d = 0.01 # pure magic
            nn.init.uniform_(self.src_embedding.weight, a=-d, b=d)
            nn.init.uniform_(self.trg_embedding.weight, a=-d, b=d)

    def init_model(self):
        num_enc_layers = self.config['num_enc_layers']
        num_enc_heads = self.config['num_enc_heads']
        num_dec_layers = self.config['num_dec_layers']
        num_dec_heads = self.config['num_dec_heads']

        embed_dim = self.config['embed_dim']
        ff_dim = self.config['ff_dim']
        dropout = self.config['dropout']
        norm_in = self.config['norm_in']

        # get encoder, decoder
        self.encoder = Encoder(num_enc_layers, num_enc_heads, embed_dim, ff_dim, dropout=dropout, norm_in=norm_in)
        self.decoder = Decoder(num_dec_layers, num_dec_heads, embed_dim, ff_dim, dropout=dropout, norm_in=norm_in)

        # leave layer norm alone
        init_func = nn.init.xavier_normal_ if self.config['weight_init_type'] == ac.XAVIER_NORMAL else nn.init.xavier_uniform_
        for m in [self.encoder.self_atts, self.encoder.pos_ffs, self.decoder.self_atts, self.decoder.pos_ffs, self.decoder.enc_dec_atts]:
            for p in m.parameters():
                if p.dim() > 1:
                    init_func(p)
                else:
                    nn.init.constant_(p, 0.)

    def get_input(self, toks, is_src=True):
        embeds = self.src_embedding if is_src else self.trg_embedding
        word_embeds = embeds(toks) # [bsz, max_len, embed_dim]

        if self.spd_centrality:
            word_embeds[:, 0, :] = word_embeds[:, 0, :] + self.centrality_embed(torch.LongTensor([0]).to(torch.device('cuda'))).to(torch.device('cuda'))

            word_embeds[:, -1, :] = word_embeds[:, -1, :] + self.centrality_embed(torch.LongTensor([0]).to(torch.device('cuda'))).to(torch.device('cuda'))

            word_embeds[:, 1:-1, :] = word_embeds[:, 0:-2, :] + self.centrality_embed(torch.LongTensor([1]).to(torch.device('cuda'))).to(torch.device('cuda'))

            return word_embeds 

        if self.config['fix_norm']:
            word_embeds = ut.normalize(word_embeds, scale=False)
        else:
            word_embeds = word_embeds * self.embed_scale

        if self.rw_pos:
            self.pos_embedding = ut.get_rw_pos(self.config['rw_pos_dim'], self.config['max_pos_length'])
            self.pos_embedding = self.rw_pos_emb(self.pos_embedding)

        if self.lape_pos:
            sign_flip = torch.rand(self.pos_embedding.shape[1]).to(torch.device('cuda'))
            sign_flip[sign_flip >= 0.5] = 1.0
            sign_flip[sign_flip < 0.5] = -1.0
            self.pos_embedding *= sign_flip.unsqueeze(0)

        if self.spectral_attn:
            self.pos_embedding = self.spectral_embedding(toks.size()[1])
            pos_embeds = self.pos_embedding[:toks.size()[-1], :].unsqueeze(0).repeat(toks.size()[0], 1, 1)
            word_embeds = self.diet_linear(word_embeds)
            return torch.cat((word_embeds, pos_embeds), dim=-1)

        pos_embeds = self.pos_embedding[:toks.size()[-1], :].unsqueeze(0) # [1, max_len, embed_dim]
        return word_embeds + pos_embeds

    def forward(self, src_toks, trg_toks, targets):
        encoder_mask = (src_toks == ac.PAD_ID).unsqueeze(1).unsqueeze(2) # [bsz, 1, 1, max_src_len]
        decoder_mask = torch.triu(torch.ones((trg_toks.size()[-1], trg_toks.size()[-1])), diagonal=1).type(trg_toks.type()) == 1
        decoder_mask = decoder_mask.unsqueeze(0).unsqueeze(1)

        if self.lape_pos:
            self.pos_embedding = self.LAPE.get_lape_encoding(self.config['lape_dim'], self.config['max_pos_length'])
            self.pos_embedding = self.lape_emb(self.pos_embedding)

        encoder_inputs = self.get_input(src_toks, is_src=True)

        spatial_pos = None
        if self.spd_centrality:
            # shortest_paths = nx.floyd_warshall(self.cycle_graph)
            # spatial_pos = [[-1]*len(shortest_paths) for _ in range(len(shortest_paths))]
            # for src, trg_dict in shortest_paths.items():
            #     for trg, distance in trg_dict.items():
            #         spatial_pos[src][trg] = distance
            #         spatial_pos[trg][src] = distance

            # spatial_pos = torch.from_numpy(np.array(spatial_pos))
            # spatial_pos = spatial_pos.type(torch.long)
            spatial_pos = self.path_embed(self.spatial_pos).permute(2, 1, 0)

        encoder_outputs = self.encoder(encoder_inputs, encoder_mask, spatial_pos)

        decoder_inputs = self.get_input(trg_toks, is_src=False)
        decoder_outputs = self.decoder(decoder_inputs, decoder_mask, encoder_outputs, encoder_mask)

        logits = self.logit_fn(decoder_outputs)
        neglprobs = F.log_softmax(logits, -1)
        neglprobs = neglprobs * self.trg_vocab_mask.type(neglprobs.type()).reshape(1, -1)
        targets = targets.reshape(-1, 1)
        non_pad_mask = targets != ac.PAD_ID
        nll_loss = -neglprobs.gather(dim=-1, index=targets)[non_pad_mask]
        smooth_loss = -neglprobs.sum(dim=-1, keepdim=True)[non_pad_mask]

        nll_loss = nll_loss.sum()
        smooth_loss = smooth_loss.sum()
        label_smoothing = self.config['label_smoothing']

        if label_smoothing > 0:
            loss = (1.0 - label_smoothing) * nll_loss + label_smoothing * smooth_loss / self.trg_vocab_mask.type(smooth_loss.type()).sum()
        else:
            loss = nll_loss

        return {
            'loss': loss,
            'nll_loss': nll_loss
        }

    def logit_fn(self, decoder_output):
        softmax_weight = self.out_embedding if not self.config['fix_norm'] else ut.normalize(self.out_embedding, scale=True)
        logits = F.linear(decoder_output, softmax_weight, bias=self.out_bias)
        logits = logits.reshape(-1, logits.size()[-1])
        logits[:, ~self.trg_vocab_mask] = -1e9
        return logits

    def beam_decode(self, src_toks):
        """Translate a minibatch of sentences. 

        Arguments: src_toks[i,j] is the jth word of sentence i.

        Return: See encoders.Decoder.beam_decode
        """
        encoder_mask = (src_toks == ac.PAD_ID).unsqueeze(1).unsqueeze(2) # [bsz, 1, 1, max_src_len]
        encoder_inputs = self.get_input(src_toks, is_src=True)
        encoder_outputs = self.encoder(encoder_inputs, encoder_mask)
        max_lengths = torch.sum(src_toks != ac.PAD_ID, dim=-1).type(src_toks.type()) + 50

        def get_trg_inp(ids, time_step):
            ids = ids.type(src_toks.type())
            word_embeds = self.trg_embedding(ids)
            if self.config['fix_norm']:
                word_embeds = ut.normalize(word_embeds, scale=False)
            else:
                word_embeds = word_embeds * self.embed_scale

            if self.spd_centrality:
                word_embeds[:, 0, :] = word_embeds[:, 0, :] + self.centrality_embed(torch.LongTensor([0]).to(torch.device('cuda'))).to(torch.device('cuda'))

                word_embeds[:, -1, :] = word_embeds[:, -1, :] + self.centrality_embed(torch.LongTensor([0]).to(torch.device('cuda'))).to(torch.device('cuda'))

                word_embeds[:, 1:-1, :] = word_embeds[:, 0:-2, :] + self.centrality_embed(torch.LongTensor([1]).to(torch.device('cuda'))).to(torch.device('cuda'))

                return word_embeds 

            if self.lape_pos:
                self.pos_embedding = self.LAPE.get_lape_encoding(self.config['lape_dim'], self.config['max_pos_length'])
                self.pos_embedding = self.lape_emb(self.pos_embedding)

            if self.spectral_attn:
                self.pos_embedding = self.spectral_embedding(time_step+1)
                pos_embeds = self.pos_embedding[time_step, :].unsqueeze(0).repeat(word_embeds.shape[0], word_embeds.shape[1], 1) # bsz x beam_size x embed_dim
                word_embeds = self.diet_linear(word_embeds)
                ret = torch.cat((word_embeds, pos_embeds), dim=-1)
                return ret
            
            if self.rw_pos:
                self.pos_embedding = ut.get_rw_pos(self.config['rw_pos_dim'], self.config['max_pos_length'])
                self.pos_embedding = self.rw_pos_emb(self.pos_embedding)


            pos_embeds = self.pos_embedding[time_step, :].reshape(1, 1, -1)
            return word_embeds + pos_embeds

        def logprob(decoder_output):
            return F.log_softmax(self.logit_fn(decoder_output), dim=-1)

        if self.config['length_model'] == 'gnmt':
            length_model = ut.gnmt_length_model(self.config['length_alpha'])
        elif self.config['length_model'] == 'linear':
            length_model = lambda t, p: p + self.config['length_alpha'] * t
        elif self.config['length_model'] == 'none':
            length_model = lambda t, p: p
        else:
            raise ValueError("invalid length_model '{}'".format(self.config['length_model']))

        return self.decoder.beam_decode(encoder_outputs, encoder_mask, get_trg_inp, logprob, length_model, ac.BOS_ID, ac.EOS_ID, max_lengths, beam_size=self.config['beam_size'])
