# models/transformer.py
import torch
import torch.nn as nn
import math
import torch.nn.functional as F

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000, learned=False):
        super(PositionalEncoding, self).__init__()
        self.learned = learned
        if learned:
            self.pe = nn.Embedding(max_len, d_model)
        else:
            pe = torch.zeros(max_len, d_model)  # [max_len, d_model]
            position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # [max_len, 1]
            div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            pe = pe.unsqueeze(0)  # [1, max_len, d_model]
            self.register_buffer('pe', pe)

    def forward(self, x):
        # x: [batch_size, seq_len, d_model]
        if self.learned:
            seq_len = x.size(1)
            positions = torch.arange(seq_len, device=x.device).unsqueeze(0)  # [1, seq_len]
            return x + self.pe(positions)
        else:
            return x + self.pe[:, :x.size(1), :]


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_k = d_model // num_heads
        self.num_heads = num_heads
        self.dropout = nn.Dropout(dropout)

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)

        Q = self.q_proj(query)  # [B, L, d_model]
        K = self.k_proj(key)
        V = self.v_proj(value)

        # Split into heads
        Q = Q.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)  # [B, H, L, d_k]
        K = K.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        # Scaled Dot-Product Attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)  # [B, H, Lq, Lk]

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        context = torch.matmul(attn_weights, V)  # [B, H, Lq, d_k]
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.d_k)

        output = self.out_proj(context)
        return output, attn_weights


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()

    def forward(self, x):
        return self.linear2(self.dropout(self.activation(self.linear1(x))))


class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ffn = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, src_mask):
        # Self-attention
        attn_out, _ = self.self_attn(src, src, src, mask=src_mask)
        src = self.norm1(src + self.dropout(attn_out))
        # FFN
        ffn_out = self.ffn(src)
        src = self.norm2(src + self.dropout(ffn_out))
        return src


class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(DecoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.enc_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ffn = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, tgt, memory, tgt_mask, memory_mask):
        # Self-attention (with causal mask)
        self_attn_out, _ = self.self_attn(tgt, tgt, tgt, mask=tgt_mask)
        tgt = self.norm1(tgt + self.dropout(self_attn_out))
        # Encoder-decoder attention
        enc_attn_out, _ = self.enc_attn(tgt, memory, memory, mask=memory_mask)
        tgt = self.norm2(tgt + self.dropout(enc_attn_out))
        # FFN
        ffn_out = self.ffn(tgt)
        tgt = self.norm3(tgt + self.dropout(ffn_out))
        return tgt


class TransformerEncoder(nn.Module):
    def __init__(self, vocab_size, d_model, num_layers, num_heads, d_ff, dropout=0.1, max_len=5000, pos_enc_type='sinusoidal'):
        super(TransformerEncoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.scale = math.sqrt(d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len, learned=(pos_enc_type == 'learned'))
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)
        ])
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, src_mask):
        x = self.embedding(src) * self.scale  # [B, L, d_model]
        x = self.pos_encoding(x)
        x = self.dropout(x)
        for layer in self.layers:
            x = layer(x, src_mask)
        return x


class TransformerDecoder(nn.Module):
    def __init__(self, vocab_size, d_model, num_layers, num_heads, d_ff, dropout=0.1, max_len=5000, pos_enc_type='sinusoidal'):
        super(TransformerDecoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.scale = math.sqrt(d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len, learned=(pos_enc_type == 'learned'))
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)
        ])
        self.dropout = nn.Dropout(dropout)
        self.fc_out = nn.Linear(d_model, vocab_size)

    def forward(self, tgt, memory, tgt_mask, memory_mask):
        x = self.embedding(tgt) * self.scale
        x = self.pos_encoding(x)
        x = self.dropout(x)
        for layer in self.layers:
            x = layer(x, memory, tgt_mask, memory_mask)
        logits = self.fc_out(x)
        return logits


class Seq2SeqTransformer(nn.Module):
    def __init__(self, encoder, decoder, src_pad_idx, tgt_pad_idx, device):
        super(Seq2SeqTransformer, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_pad_idx = src_pad_idx
        self.tgt_pad_idx = tgt_pad_idx
        self.device = device

    def make_src_mask(self, src):
        # src: [B, L]
        return (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)  # [B, 1, 1, L]

    def make_tgt_mask(self, tgt):
        # Causal mask + padding mask
        batch_size, tgt_len = tgt.shape
        tgt_pad_mask = (tgt != self.tgt_pad_idx).unsqueeze(1).unsqueeze(2)  # [B, 1, 1, L]
        tgt_sub_mask = torch.tril(torch.ones((tgt_len, tgt_len), device=self.device)).bool()  # [L, L]
        tgt_mask = tgt_pad_mask & tgt_sub_mask  # [B, 1, L, L]
        return tgt_mask

    def forward(self, src, tgt):
        src_mask = self.make_src_mask(src)
        tgt_mask = self.make_tgt_mask(tgt)

        enc_output = self.encoder(src, src_mask)
        output = self.decoder(tgt, enc_output, tgt_mask, src_mask)
        return output