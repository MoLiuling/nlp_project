import torch
import json
import pickle
import sys
import os
from pathlib import Path

# 动态导入 transformers（T5 所需）
try:
    from transformers import T5ForConditionalGeneration, T5Tokenizer
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False

# 添加当前目录到 Python 路径
sys.path.append(str(Path(__file__).parent))

# 导入模型
from models import RNNEncoder, RNNDecoder, Seq2SeqRNN
from models.transformer import TransformerEncoder, TransformerDecoder, Seq2SeqTransformer

# ===== 全局配置 =====
SUPPORTED_MODELS = ['rnn_additive', 'rnn_dot', 'rnn_multiplicative', 't5', 'transformer']
BASE_DIR = Path(".")

# RNN 配置
RNN_CONFIG = {
    "enc_emb_dim": 256,
    "dec_emb_dim": 256,
    "enc_hid_dim": 512,
    "dec_hid_dim": 512,
    "n_layers": 2,
    "dropout": 0.5,
    "max_len": 50
}

# Transformer 配置路径
TRANSFORMER_CKPT_DIR = BASE_DIR / "outputs/transformer"

# ✅ T5 模型路径
T5_MODEL_NAME_OR_PATH = "t5-small"


class Vocab:
    def __init__(self, stoi_dict=None):
        self.stoi = stoi_dict or {}
        self.itos = {v: k for k, v in self.stoi.items()}

    @classmethod
    def from_file(cls, path):
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Vocabulary file not found: {path}")
        with open(path, "rb") as f:
            stoi_dict = pickle.load(f)
        return cls(stoi_dict)

    def __len__(self):
        return len(self.stoi)


def tokenize_zh(text):
    return list(text.strip())


class ModelManager:
    def __init__(self, model_name: str, device: torch.device):
        self.model_name = model_name
        self.device = device
        self.model = None
        self.src_vocab = None
        self.tgt_vocab = None
        self.tokenizer = None

        if model_name.startswith("rnn_"):
            self.load_rnn_model()
        elif model_name == "t5":
            self.load_t5_model()
        elif model_name == "transformer":
            self.load_transformer_model()
        else:
            raise ValueError(f"Unsupported model: {model_name}")

    def load_rnn_model(self):
        attn_type_map = {
            "rnn_additive": "additive",
            "rnn_dot": "dot",
            "rnn_multiplicative": "multiplicative"
        }
        attn_type = attn_type_map[self.model_name]
        config = RNN_CONFIG

        src_vocab_path = BASE_DIR / "data" / "prepared_vocab_zh.pkl"
        tgt_vocab_path = BASE_DIR / "data" / "prepared_vocab_en.pkl"

        self.src_vocab = Vocab.from_file(src_vocab_path)
        self.tgt_vocab = Vocab.from_file(tgt_vocab_path)

        enc = RNNEncoder(
            vocab_size=len(self.src_vocab),
            embed_dim=config["enc_emb_dim"],
            hidden_dim=config["enc_hid_dim"],
            num_layers=config["n_layers"],
            dropout=config["dropout"]
        )
        dec = RNNDecoder(
            vocab_size=len(self.tgt_vocab),
            embed_dim=config["dec_emb_dim"],
            hidden_dim=config["dec_hid_dim"],
            num_layers=config["n_layers"],
            dropout=config["dropout"],
            attn_type=attn_type
        )

        self.model = Seq2SeqRNN(enc, dec, self.device).to(self.device)
        ckpt_path = BASE_DIR / f"outputs/rnn/model_{attn_type}.pth"
        self.model.load_state_dict(torch.load(ckpt_path, map_location=self.device))
        self.model.eval()

    def load_t5_model(self):
        if not HAS_TRANSFORMERS:
            raise ImportError("Please install transformers")
        
        # 使用官方 tokenizer（避免 spiece.model 缺失）
        self.tokenizer = T5Tokenizer.from_pretrained("t5-small", legacy=False)
        
        # ✅ 构建本地模型路径：使用 .as_posix() 确保正斜杠
        model_dir = BASE_DIR / "t5" / "checkpoint-313"
        if not model_dir.exists():
            raise FileNotFoundError(
                f"T5 checkpoint directory not found: {model_dir.resolve()}\n"
                "Please ensure you have trained the T5 model and saved it to this path."
            )
        
        model_path = model_dir.as_posix()  # 转为 'a/b/c' 格式，兼容 transformers
        self.model = T5ForConditionalGeneration.from_pretrained(model_path).to(self.device)
        self.model.eval()

    def load_transformer_model(self):
        # Load config.json
        config_path = TRANSFORMER_CKPT_DIR / "config.json"
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)

        # Load vocab
        src_vocab_path = BASE_DIR / "data" / "prepared_vocab_zh.pkl"
        tgt_vocab_path = BASE_DIR / "data" / "prepared_vocab_en.pkl"

        self.src_vocab = Vocab.from_file(src_vocab_path)
        self.tgt_vocab = Vocab.from_file(tgt_vocab_path)

        # Extract config values
        d_model = config.get("d_model", 512)
        num_heads = config.get("num_heads", 8)
        num_layers = config.get("num_layers", 4)
        d_ff = config.get("d_ff", 2048)
        dropout = config.get("dropout", 0.1)
        pos_enc_type = config.get("pos_enc_type", "sinusoidal")

        src_pad_idx = self.src_vocab.stoi.get('<pad>', 0)
        tgt_pad_idx = self.tgt_vocab.stoi.get('<pad>', 0)

        enc = TransformerEncoder(
            vocab_size=len(self.src_vocab),
            d_model=d_model,
            num_layers=num_layers,
            num_heads=num_heads,
            d_ff=d_ff,
            dropout=dropout,
            pos_enc_type=pos_enc_type,
            max_len=config.get("max_len", 50)
        )
        dec = TransformerDecoder(
            vocab_size=len(self.tgt_vocab),
            d_model=d_model,
            num_layers=num_layers,
            num_heads=num_heads,
            d_ff=d_ff,
            dropout=dropout,
            pos_enc_type=pos_enc_type,
            max_len=config.get("max_len", 50)
        )

        self.model = Seq2SeqTransformer(
            encoder=enc,
            decoder=dec,
            src_pad_idx=src_pad_idx,
            tgt_pad_idx=tgt_pad_idx,
            device=self.device
        ).to(self.device)

        model_path = TRANSFORMER_CKPT_DIR / "model.pth"
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()

    def translate(self, sentence: str, decode_mode="greedy", beam_size=5, max_len=50):
        if self.model_name == "t5":
            return self._translate_t5(sentence, decode_mode, beam_size, max_len)
        elif self.model_name == "transformer":
            if decode_mode == "greedy":
                return self._translate_transformer(sentence, max_len)
            else:
                return self._translate_transformer_beam(sentence, beam_size, max_len)
        else:
            if decode_mode == "greedy":
                return self._translate_rnn(sentence, max_len)
            else:
                return self._translate_rnn_beam(sentence, beam_size, max_len)

    def _translate_t5(self, sentence: str, decode_mode: str, beam_size: int, max_len: int):
        input_text = f"translate Chinese to English: {sentence}"
        inputs = self.tokenizer(input_text, return_tensors="pt", padding=True, truncation=True, max_length=max_len).to(self.device)

        if decode_mode == "greedy":
            outputs = self.model.generate(
                **inputs,
                max_length=max_len,
                num_beams=1,
                do_sample=False,
                early_stopping=True
            )
        else:
            outputs = self.model.generate(
                **inputs,
                max_length=max_len,
                num_beams=beam_size,
                early_stopping=True,
                length_penalty=0.6
            )

        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    def _translate_rnn(self, sentence: str, max_len=50):
        tokens = tokenize_zh(sentence)
        unk_id = self.src_vocab.stoi.get('<unk>', 1)
        sos_id = self.src_vocab.stoi.get('<sos>', 2)
        eos_id = self.src_vocab.stoi.get('<eos>', 3)

        src_ids = [self.src_vocab.stoi.get(tok, unk_id) for tok in tokens]
        src_ids = [sos_id] + src_ids + [eos_id]
        src_tensor = torch.LongTensor(src_ids).unsqueeze(0).to(self.device)

        encoder_outputs, hidden, cell = self.model.encoder(src_tensor)
        ys = [sos_id]
        for _ in range(max_len):
            input_token = torch.tensor([ys[-1]], device=self.device)
            output, hidden, cell, _ = self.model.decoder(input_token, encoder_outputs, hidden, cell)
            pred = output.argmax(1).item()
            if pred == eos_id:
                break
            ys.append(pred)

        translated = [self.tgt_vocab.itos.get(idx, '<unk>') for idx in ys[1:]]
        return ' '.join(translated)

    def _translate_rnn_beam(self, sentence: str, beam_size: int, max_len: int):
        tokens = tokenize_zh(sentence)
        unk_id = self.src_vocab.stoi.get('<unk>', 1)
        sos_id = self.src_vocab.stoi.get('<sos>', 2)
        eos_id = self.src_vocab.stoi.get('<eos>', 3)

        src_ids = [self.src_vocab.stoi.get(tok, unk_id) for tok in tokens]
        src_ids = [sos_id] + src_ids + [eos_id]
        src_tensor = torch.LongTensor(src_ids).unsqueeze(0).to(self.device)

        encoder_outputs, hidden, cell = self.model.encoder(src_tensor)
        beams = [(0.0, [sos_id], hidden, cell)]  # (score, seq, h, c)

        for step in range(max_len):
            candidates = []
            for score, seq, h, c in beams:
                if seq[-1] == eos_id:
                    candidates.append((score, seq, h, c))
                    continue
                input_token = torch.tensor([[seq[-1]]], device=self.device)
                output, new_h, new_c, _ = self.model.decoder(input_token, encoder_outputs, h, c)
                log_probs = torch.log_softmax(output.squeeze(0), dim=-1)
                topk_vals, topk_ids = log_probs.topk(beam_size)
                for val, idx in zip(topk_vals, topk_ids):
                    new_seq = seq + [idx.item()]
                    new_score = score + val.item()
                    candidates.append((new_score, new_seq, new_h, new_c))
            beams = sorted(candidates, key=lambda x: x[0], reverse=True)[:beam_size]

        best_seq = beams[0][1][1:]
        if best_seq and best_seq[-1] == eos_id:
            best_seq = best_seq[:-1]
        translated = [self.tgt_vocab.itos.get(idx, '<unk>') for idx in best_seq]
        return ' '.join(translated)

    def _translate_transformer(self, sentence: str, max_len=50):
        tokens = tokenize_zh(sentence)
        unk_id = self.src_vocab.stoi.get('<unk>', 1)
        sos_id = self.src_vocab.stoi.get('<sos>', 2)
        eos_id = self.src_vocab.stoi.get('<eos>', 3)
        pad_id = self.src_vocab.stoi.get('<pad>', 0)

        src_ids = [self.src_vocab.stoi.get(tok, unk_id) for tok in tokens]
        src_ids = [sos_id] + src_ids + [eos_id]
        src_tensor = torch.LongTensor(src_ids).unsqueeze(0).to(self.device)

        src_mask = self.model.make_src_mask(src_tensor)
        with torch.no_grad():
            memory = self.model.encoder(src_tensor, src_mask)

        ys = [sos_id]
        for _ in range(max_len - 1):
            tgt_tensor = torch.LongTensor(ys).unsqueeze(0).to(self.device)
            tgt_mask = self.model.make_tgt_mask(tgt_tensor)
            with torch.no_grad():
                output = self.model.decoder(tgt_tensor, memory, tgt_mask, src_mask)
                pred = output[:, -1, :].argmax(1).item()
            if pred == eos_id:
                break
            ys.append(pred)

        translated = [self.tgt_vocab.itos.get(idx, '<unk>') for idx in ys[1:]]
        return ' '.join(translated)

    def _translate_transformer_beam(self, sentence: str, beam_size: int, max_len: int):
        tokens = tokenize_zh(sentence)
        unk_id = self.src_vocab.stoi.get('<unk>', 1)
        sos_id = self.src_vocab.stoi.get('<sos>', 2)
        eos_id = self.src_vocab.stoi.get('<eos>', 3)

        src_ids = [self.src_vocab.stoi.get(tok, unk_id) for tok in tokens]
        src_ids = [sos_id] + src_ids + [eos_id]
        src_tensor = torch.LongTensor(src_ids).unsqueeze(0).to(self.device)

        src_mask = self.model.make_src_mask(src_tensor)
        with torch.no_grad():
            memory = self.model.encoder(src_tensor, src_mask)

        beams = [(0.0, [sos_id])]  # (score, seq)

        for step in range(max_len):
            candidates = []
            for score, seq in beams:
                if seq[-1] == eos_id:
                    candidates.append((score, seq))
                    continue
                tgt_tensor = torch.LongTensor(seq).unsqueeze(0).to(self.device)
                tgt_mask = self.model.make_tgt_mask(tgt_tensor)
                with torch.no_grad():
                    output = self.model.decoder(tgt_tensor, memory, tgt_mask, src_mask)
                    logits = output[:, -1, :]  # [1, vocab]
                    log_probs = torch.log_softmax(logits, dim=-1)
                    topk_vals, topk_ids = log_probs.topk(beam_size)
                    for val, idx in zip(topk_vals, topk_ids):
                        new_seq = seq + [idx.item()]
                        new_score = score + val.item()
                        candidates.append((new_score, new_seq))
            beams = sorted(candidates, key=lambda x: x[0], reverse=True)[:beam_size]

        best_seq = beams[0][1][1:]
        if best_seq and best_seq[-1] == eos_id:
            best_seq = best_seq[:-1]
        translated = [self.tgt_vocab.itos.get(idx, '<unk>') for idx in best_seq]
        return ' '.join(translated)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Translate Chinese to English")
    parser.add_argument("--model", type=str, choices=SUPPORTED_MODELS, default="rnn_dot",
                        help="Model to use for inference")
    parser.add_argument("--decode", type=str, choices=["greedy", "beam"], default="greedy",
                        help="Decoding strategy: greedy or beam search")
    parser.add_argument("--beam-size", type=int, default=5, help="Beam size for beam search")
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(f"Loading model: {args.model}")
    print(f"Decoding mode: {args.decode} (beam_size={args.beam_size})")

    try:
        manager = ModelManager(args.model, device)
        print("✅ Model loaded successfully.")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        sys.exit(1)

    print("\n🔤 Chinese-to-English Translation")
    print(f"Model: {args.model} | Device: {device}")
    print("Commands:")
    print("  - Type a Chinese sentence to translate")
    print("  - Type 'quit' to exit\n")

    while True:
        try:
            user_input = input("🇨🇳 Input: ").strip()
            if not user_input:
                continue
            if user_input.lower() in ["quit", "exit", "q"]:
                print("👋 Goodbye!")
                break

            translation = manager.translate(user_input, decode_mode=args.decode, beam_size=args.beam_size)
            print(f"🇺🇸 Output: {translation}\n")

        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            import traceback
            print(f"❌ Error during translation: {e}")
            traceback.print_exc()


if __name__ == "__main__":
    main()