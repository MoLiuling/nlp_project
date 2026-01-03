# models/attention.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class Attention(nn.Module):
    def __init__(self, hidden_dim, attn_type='additive'):
        super(Attention, self).__init__()
        self.attn_type = attn_type
        self.hidden_dim = hidden_dim

        if attn_type == 'additive':
            self.Wa = nn.Linear(hidden_dim, hidden_dim, bias=False)
            self.Ua = nn.Linear(hidden_dim, hidden_dim, bias=False)
            self.Va = nn.Linear(hidden_dim, 1, bias=False)
        elif attn_type in ['dot', 'multiplicative']:
            # dot: raw dot product; multiplicative: scaled dot (like Transformer)
            pass
        else:
            raise ValueError(f"Unsupported attention type: {attn_type}")

    def forward(self, decoder_hidden, encoder_outputs, mask=None):
        """
        decoder_hidden: [batch_size, hidden_dim] (last hidden state of decoder)
        encoder_outputs: [batch_size, src_len, hidden_dim]
        returns: context [batch_size, hidden_dim], attn_weights [batch_size, src_len]
        """
        batch_size, src_len, _ = encoder_outputs.size()

        if self.attn_type == 'additive':
            # Additive attention (Bahdanau)
            dec_expanded = decoder_hidden.unsqueeze(1)  # [B, 1, H]
            energy = torch.tanh(self.Wa(dec_expanded) + self.Ua(encoder_outputs))  # [B, src_len, H]
            scores = self.Va(energy).squeeze(2)  # [B, src_len]

        elif self.attn_type == 'dot':
            # Dot-product attention
            scores = torch.bmm(
                decoder_hidden.unsqueeze(1),  # [B, 1, H]
                encoder_outputs.transpose(1, 2)  # [B, H, src_len]
            ).squeeze(1)  # [B, src_len]

        elif self.attn_type == 'multiplicative':
            # Scaled dot-product
            scores = torch.bmm(
                decoder_hidden.unsqueeze(1),
                encoder_outputs.transpose(1, 2)
            ).squeeze(1) / (self.hidden_dim ** 0.5)

        else:
            raise NotImplementedError

        if mask is not None:
            scores.masked_fill_(mask == 0, -1e9)

        attn_weights = F.softmax(scores, dim=1)  # [B, src_len]
        context = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs)  # [B, 1, H]
        context = context.squeeze(1)  # [B, H]

        return context, attn_weights