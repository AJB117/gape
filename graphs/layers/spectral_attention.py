import torch
import torch.nn as nn

class SpectralAttention(nn.Module):
    def __init__(self, lpe_dim, lpe_n_heads, lpe_n_layers):
        super().__init__()

        encoder_layer = nn.TransformerEncoderLayer(lpe_dim, lpe_n_heads)
        self.lpe_attn = nn.TransformerEncoder(encoder_layer, lpe_n_layers)
        self.linear = nn.Linear(2, lpe_dim)

    def forward(self, h, eigvecs, eigvals):

        lpe = torch.cat((eigvecs.unsqueeze(2), eigvals), dim=2).float() # (Num nodes) x (Num Eigenvectors) x 2
        empty_mask = torch.isnan(lpe) # (Num nodes) x (Num Eigenvectors) x 2

        lpe[empty_mask] = 0 # (Num nodes) x (Num Eigenvectors) x 2
        lpe = torch.transpose(lpe, 0 ,1).float() # (Num Eigenvectors) x (Num nodes) x 2
        lpe = self.linear(lpe) # (Num Eigenvectors) x (Num nodes) x PE_dim

        lpe = self.lpe_attn(src=lpe, src_key_padding_mask=empty_mask[:,:,0])

        #remove masked sequences
        lpe[torch.transpose(empty_mask, 0 ,1)[:,:,0]] = float('nan')

        #Sum pooling
        lpe = torch.nansum(lpe, 0, keepdim=False)

        h = torch.cat((h, lpe), dim=1)
        return h