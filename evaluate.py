# encoding='utf-8'
import os
import json
import argparse
import torch
import pickle
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

# ====== 1. 加载自研模型（RNN/Transformer）======
def load_custom_model(model_dir, model_type, variant=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if model_type == 'rnn':
        from models.rnn import RNNEncoder, RNNDecoder, Seq2SeqRNN

        # === 加载词汇表：优先从 model_dir，回退到 data/ ===
        def find_vocab(vocab_name):
            path_in_model = os.path.join(model_dir, vocab_name)
            path_in_data = os.path.join('data', f'prepared_{vocab_name}')
            if os.path.exists(path_in_model):
                return path_in_model
            elif os.path.exists(path_in_data):
                return path_in_data
            else:
                raise FileNotFoundError(f"Vocabulary file not found: neither {path_in_model} nor {path_in_data}")

        vocab_zh_path = find_vocab('vocab_zh.pkl')
        vocab_en_path = find_vocab('vocab_en.pkl')

        with open(vocab_zh_path, 'rb') as f:
            src_vocab = pickle.load(f)
        with open(vocab_en_path, 'rb') as f:
            tgt_vocab = pickle.load(f)

        # === 加载配置 ===
        config_path = os.path.join(model_dir, 'config.json')
        if not os.path.exists(config_path):
            config_path = os.path.join('data', 'prepared_config.json')
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)

        # 获取词汇大小
        src_vocab_size = len(src_vocab)
        tgt_vocab_size = len(tgt_vocab)

        # 获取 <pad> 索引（安全回退）
        src_pad_idx = src_vocab.get('<pad>', src_vocab.get('<unk>', 0))
        tgt_pad_idx = tgt_vocab.get('<pad>', tgt_vocab.get('<unk>', 0))

        # 从 config 或默认值中提取参数
        embed_dim = config.get('embed_dim', 256)
        hidden_dim = config.get('hidden_dim', 512)
        num_layers = config.get('num_layers', 2)
        dropout = config.get('dropout', 0.3)
        attn_type = variant if variant is not None else config.get('attn_type', 'multiplicative')

        # 创建 Encoder 和 Decoder
        encoder = RNNEncoder(
            vocab_size=src_vocab_size,
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout
        )
        decoder = RNNDecoder(
            vocab_size=tgt_vocab_size,
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            attn_type=attn_type
        )

        # 创建完整模型
        model = Seq2SeqRNN(encoder, decoder, device)

        # === 加载权重 ===
        if variant is None:
            model_file = 'best_model.pt'
        else:
            model_file = f'model_{variant}.pth'

        model_path = os.path.join(model_dir, model_file)
        if not os.path.exists(model_path):
            model_path = os.path.join(model_dir, 'model.pth')
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model weight file not found. Tried: {os.path.join(model_dir, model_file)} and {os.path.join(model_dir, 'model.pth')}")

        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()

        return model, src_vocab, tgt_vocab

    elif model_type == 'transformer':
        from models.transformer import (
            TransformerEncoder,
            TransformerDecoder,
            Seq2SeqTransformer
        )

        # === 加载词汇表（固定从 data/ 加载 prepared_*.pkl）===
        src_vocab_path = os.path.join('data', 'prepared_vocab_zh.pkl')
        tgt_vocab_path = os.path.join('data', 'prepared_vocab_en.pkl')

        with open(src_vocab_path, 'rb') as f:
            src_vocab = pickle.load(f)
        with open(tgt_vocab_path, 'rb') as f:
            tgt_vocab = pickle.load(f)

        # === 加载 config.json（优先从 model_dir）===
        config_path = os.path.join(model_dir, 'config.json')
        if not os.path.exists(config_path):
            config_path = os.path.join('data', 'prepared_config.json')
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)

        # 获取参数
        src_vocab_size = len(src_vocab)
        tgt_vocab_size = len(tgt_vocab)
        src_pad_idx = src_vocab.get('<pad>', 0)
        tgt_pad_idx = tgt_vocab.get('<pad>', 0)

        d_model = config.get('d_model', 512)
        num_heads = config.get('num_heads', 8)
        num_encoder_layers = config.get('num_encoder_layers', config.get('num_layers', 6))
        num_decoder_layers = config.get('num_decoder_layers', config.get('num_layers', 6))
        d_ff = config.get('d_ff', 2048)
        dropout = config.get('dropout', 0.1)
        max_len = config.get('max_len', 5000)
        pos_enc_type = config.get('pos_enc_type', 'sinusoidal')

        # 创建 Encoder 和 Decoder
        encoder = TransformerEncoder(
            vocab_size=src_vocab_size,
            d_model=d_model,
            num_layers=num_encoder_layers,
            num_heads=num_heads,
            d_ff=d_ff,
            dropout=dropout,
            max_len=max_len,
            pos_enc_type=pos_enc_type
        )

        decoder = TransformerDecoder(
            vocab_size=tgt_vocab_size,
            d_model=d_model,
            num_layers=num_decoder_layers,
            num_heads=num_heads,
            d_ff=d_ff,
            dropout=dropout,
            max_len=max_len,
            pos_enc_type=pos_enc_type
        )

        # 创建完整模型
        model = Seq2SeqTransformer(
            encoder=encoder,
            decoder=decoder,
            src_pad_idx=src_pad_idx,
            tgt_pad_idx=tgt_pad_idx,
            device=device
        )

        # === 加载权重 ===
        model_path = os.path.join(model_dir, 'best_model.pt')
        if not os.path.exists(model_path):
            model_path = os.path.join(model_dir, 'model.pth')
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()

        return model, src_vocab, tgt_vocab

    else:
        raise ValueError(f"Unsupported model_type: {model_type}")


# ====== 2. RNN 模型推理函数 ======
def translate_custom(model, src_vocab, tgt_vocab, zh_text, device):
    tokens = ['<sos>'] + list(zh_text) + ['<eos>']
    src_ids = [src_vocab.get(t, src_vocab.get('<unk>', 0)) for t in tokens]
    src_tensor = torch.LongTensor(src_ids).unsqueeze(0).to(device)

    with torch.no_grad():
        encoder_outputs, hidden, cell = model.encoder(src_tensor)
        dec_input = torch.tensor([tgt_vocab['<sos>']], device=device)
        decoded_ids = []
        max_len = 100

        for _ in range(max_len):
            pred, hidden, cell, _ = model.decoder(dec_input, encoder_outputs, hidden, cell)
            next_token = pred.argmax(1).item()
            if next_token == tgt_vocab.get('<eos>', 1):
                break
            decoded_ids.append(next_token)
            dec_input = torch.tensor([next_token], device=device)

    idx2word = {idx: word for word, idx in tgt_vocab.items()}
    en_words = [idx2word.get(tid, '<unk>') for tid in decoded_ids]
    return " ".join(en_words)


# ====== 3. Transformer 模型推理函数 ======
def translate_transformer(model, src_vocab, tgt_vocab, zh_text, device):
    tokens = ['<sos>'] + list(zh_text) + ['<eos>']
    src_ids = [src_vocab.get(t, src_vocab.get('<unk>', 0)) for t in tokens]
    src_tensor = torch.LongTensor(src_ids).unsqueeze(0).to(device)  # [1, L]

    decoded_ids = [tgt_vocab['<sos>']]
    max_len = 100

    with torch.no_grad():
        for _ in range(max_len):
            tgt_tensor = torch.LongTensor(decoded_ids).unsqueeze(0).to(device)  # [1, T]
            output = model(src_tensor, tgt_tensor)  # [1, T, vocab_size]
            next_token = output[0, -1, :].argmax().item()
            if next_token == tgt_vocab.get('<eos>', 1):
                break
            decoded_ids.append(next_token)

    idx2word = {idx: word for word, idx in tgt_vocab.items()}
    en_words = [idx2word.get(tid, '<unk>') for tid in decoded_ids[1:]]  # skip <sos>
    return " ".join(en_words)


# ====== 4. T5 模型（保持不变）======
def load_t5_model(model_dir):
    from transformers import T5ForConditionalGeneration, T5Tokenizer
    tokenizer = T5Tokenizer.from_pretrained(model_dir)
    model = T5ForConditionalGeneration.from_pretrained(model_dir)
    return model, tokenizer


def translate_t5(model, tokenizer, zh_text, device):
    input_text = f"translate Chinese to English: {zh_text}"
    input_ids = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True, max_length=128).input_ids.to(device)
    outputs = model.generate(
        input_ids,
        max_length=128,
        num_beams=4,
        early_stopping=True
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


# ====== 5. 主评估函数 ======
def evaluate(model_type, model_path, data_path, device, variant=None):
    valid_data = []
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line.strip())
            valid_data.append((item['zh'], item['en']))

    if model_type == 't5':
        model, tokenizer = load_t5_model(model_path)
        src_vocab = tgt_vocab = None
    else:  # 'rnn' or 'transformer'
        model, src_vocab, tgt_vocab = load_custom_model(model_path, model_type, variant)
        tokenizer = None

    model.to(device)
    predictions = []
    references = []

    for i, (zh, en_ref) in enumerate(valid_data):
        try:
            if model_type == 't5':
                pred = translate_t5(model, tokenizer, zh, device)
            elif model_type == 'rnn':
                pred = translate_custom(model, src_vocab, tgt_vocab, zh, device)
            elif model_type == 'transformer':
                pred = translate_transformer(model, src_vocab, tgt_vocab, zh, device)
            else:
                raise ValueError(f"Unknown model_type: {model_type}")
        except Exception as e:
            print(f"⚠️ Error at sample {i}: {e}")
            pred = ""

        predictions.append(pred.split())
        references.append(en_ref.split())

    smooth = SmoothingFunction().method4
    bleu_scores = [
        sentence_bleu([ref], pred, smoothing_function=smooth)
        for pred, ref in zip(predictions, references)
    ]
    avg_bleu = sum(bleu_scores) / len(bleu_scores) if bleu_scores else 0.0

    print(f"✅ Average BLEU: {avg_bleu:.4f}")
    return avg_bleu, predictions, references, valid_data


# ====== 6. 命令行接口 ======
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', type=str, required=True, choices=['rnn', 't5', 'transformer'])
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--data_path', type=str, default='data/valid.jsonl')
    parser.add_argument('--output_result', type=str, default=None,
                        help='Save evaluation result to JSON file (e.g., results_rnn.json)')
    parser.add_argument('--variant', type=str, default=None,
                        help='For RNN: additive, dot, multiplicative (optional)')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    avg_bleu, predictions, references, valid_data = evaluate(
        args.model_type, args.model_path, args.data_path, device, args.variant
    )

    if args.output_result:
        output_dir = os.path.dirname(args.output_result)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        examples = []
        for i in range(min(5, len(valid_data))):
            examples.append({
                "src": valid_data[i][0],
                "ref": " ".join(references[i]),
                "pred": " ".join(predictions[i])
            })

        result = {
            "model_type": args.model_type,
            "variant": args.variant,
            "bleu_score": round(avg_bleu, 4),
            "examples": examples
        }

        with open(args.output_result, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"✅ Saved evaluation result to {args.output_result}")