# models/rnn.py
import torch
import torch.nn as nn
from .attention import Attention

class RNNEncoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers=2, dropout=0.3):
        super(RNNEncoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.rnn = nn.LSTM(embed_dim, hidden_dim, num_layers=num_layers,
                           batch_first=True, dropout=dropout, bidirectional=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        # src: [batch_size, src_len]
        embedded = self.dropout(self.embedding(src))  # [B, L, E]
        outputs, (hidden, cell) = self.rnn(embedded)  # outputs: [B, L, H]
        return outputs, hidden, cell

class RNNDecoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers=2,
                 dropout=0.3, attn_type='additive'):
        super(RNNDecoder, self).__init__()
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.attention = Attention(hidden_dim, attn_type=attn_type)
        self.rnn = nn.LSTM(embed_dim + hidden_dim, hidden_dim, num_layers=num_layers,
                           batch_first=True, dropout=dropout)
        self.fc_out = nn.Linear(hidden_dim * 2, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, tgt, encoder_outputs, hidden, cell):
        """
        tgt: [batch_size, 1] (one token at a time during training/inference)
        encoder_outputs: [batch_size, src_len, hidden_dim]
        hidden, cell: from encoder, each [num_layers, batch_size, hidden_dim]
        """
        tgt = tgt.unsqueeze(1)  # [B, 1]
        embedded = self.dropout(self.embedding(tgt))  # [B, 1, E]

        # Get context vector via attention
        dec_hidden_last = hidden[-1]  # [B, H]
        context, attn_weights = self.attention(dec_hidden_last, encoder_outputs)

        # Concatenate embedding and context
        rnn_input = torch.cat([embedded, context.unsqueeze(1)], dim=2)  # [B, 1, E+H]

        output, (hidden, cell) = self.rnn(rnn_input, (hidden, cell))
        output = output.squeeze(1)  # [B, H]
        context = context  # [B, H]

        pred = self.fc_out(torch.cat([output, context], dim=1))  # [B, vocab_size]
        return pred, hidden, cell, attn_weights

class Seq2SeqRNN(nn.Module):
    def __init__(self, encoder, decoder, device):
        super(Seq2SeqRNN, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, src, tgt, teacher_forcing_ratio=0.5):
        """
        src: [batch_size, src_len]
        tgt: [batch_size, tgt_len]
        """
        batch_size, tgt_len = tgt.shape
        tgt_vocab_size = self.decoder.vocab_size

        # Encode
        encoder_outputs, hidden, cell = self.encoder(src)

        # First input to decoder is <sos>
        dec_input = tgt[:, 0]  # [B]
        outputs = torch.zeros(batch_size, tgt_len, tgt_vocab_size).to(self.device)

        for t in range(1, tgt_len):
            pred, hidden, cell, _ = self.decoder(dec_input, encoder_outputs, hidden, cell)
            outputs[:, t, :] = pred

            # Teacher forcing
            use_teacher_forcing = torch.rand(1).item() < teacher_forcing_ratio
            dec_input = tgt[:, t] if use_teacher_forcing else pred.argmax(1)

        return outputs