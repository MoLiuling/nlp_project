# models/__init__.py
from .rnn import RNNEncoder, RNNDecoder, Seq2SeqRNN
from .transformer import (
    TransformerEncoder,
    TransformerDecoder,
    Seq2SeqTransformer,
    PositionalEncoding,
    MultiHeadAttention
)

def build_model(config, device):
    model_type = config.get('model_type', 'rnn')
    
    if model_type == 'rnn':
        from .rnn import RNNEncoder, RNNDecoder, Seq2SeqRNN
        encoder = RNNEncoder(
            vocab_size=config['src_vocab_size'],
            embed_dim=config['embed_dim'],
            hidden_dim=config['hidden_dim'],
            num_layers=config.get('num_layers', 2),
            dropout=config.get('dropout', 0.3)
        )
        decoder = RNNDecoder(
            vocab_size=config['tgt_vocab_size'],
            embed_dim=config['embed_dim'],
            hidden_dim=config['hidden_dim'],
            num_layers=config.get('num_layers', 2),
            dropout=config.get('dropout', 0.3),
            attn_type=config.get('attn_type', 'additive')
        )
        return Seq2SeqRNN(encoder, decoder, device)

    elif model_type == 'transformer':
        from .transformer import TransformerEncoder, TransformerDecoder, Seq2SeqTransformer
        encoder = TransformerEncoder(
            vocab_size=config['src_vocab_size'],
            d_model=config['d_model'],
            num_layers=config.get('num_layers', 4),
            num_heads=config['num_heads'],
            d_ff=config.get('d_ff', 2048),
            dropout=config.get('dropout', 0.1),
            pos_enc_type=config.get('pos_enc_type', 'sinusoidal')
        )
        decoder = TransformerDecoder(
            vocab_size=config['tgt_vocab_size'],
            d_model=config['d_model'],
            num_layers=config.get('num_layers', 4),
            num_heads=config['num_heads'],
            d_ff=config.get('d_ff', 2048),
            dropout=config.get('dropout', 0.1),
            pos_enc_type=config.get('pos_enc_type', 'sinusoidal')
        )
        return Seq2SeqTransformer(
            encoder, decoder,
            src_pad_idx=config['src_pad_idx'],
            tgt_pad_idx=config['tgt_pad_idx'],
            device=device
        )
    else:
        raise ValueError(f"Unknown model_type: {model_type}")