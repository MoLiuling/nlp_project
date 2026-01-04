# inference.py
import torch
import pickle
from models import Encoder, Decoder, Seq2Seq, Attention
from utils import Vocab, tokenize_zh
import sys

# ===== 配置参数（需与 train.py 一致）=====
ENC_EMB_DIM = 256
DEC_EMB_DIM = 256
ENC_HID_DIM = 512
DEC_HID_DIM = 512
N_LAYERS = 2
DROPOUT = 0.5
RNN_TYPE = 'gru'
ATTN_TYPE = 'additive'  # 或你最终选的 best attention type
MODEL_PATH = "best_model.pt"
SRC_VOCAB_PATH = "src_vocab.pkl"
TGT_VOCAB_PATH = "tgt_vocab.pkl"
MAX_LEN = 50

def load_model_and_vocab(device):
    # Load vocabularies
    with open(SRC_VOCAB_PATH, 'rb') as f:
        src_vocab = pickle.load(f)
    with open(TGT_VOCAB_PATH, 'rb') as f:
        tgt_vocab = pickle.load(f)

    # Reconstruct model architecture
    attn = Attention(ENC_HID_DIM, DEC_HID_DIM, attn_type=ATTN_TYPE)
    enc = Encoder(len(src_vocab), ENC_EMB_DIM, ENC_HID_DIM, DEC_HID_DIM, N_LAYERS, DROPOUT, RNN_TYPE)
    dec = Decoder(len(tgt_vocab), DEC_EMB_DIM, ENC_HID_DIM, DEC_HID_DIM, attn, N_LAYERS, DROPOUT, RNN_TYPE)
    model = Seq2Seq(enc, dec, device).to(device)

    # Load trained weights
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    print("✅ Model and vocabularies loaded successfully.")
    return model, src_vocab, tgt_vocab

def translate_sentence(model, src_vocab, tgt_vocab, sentence, device, max_len=MAX_LEN):
    model.eval()
    with torch.no_grad():
        # Tokenize and encode source
        tokens = tokenize_zh(sentence.strip())
        src_ids = [src_vocab.stoi.get(tok, src_vocab.stoi['<unk>']) for tok in tokens]
        src_ids = [src_vocab.stoi['<sos>']] + src_ids + [src_vocab.stoi['<eos>']]
        src_tensor = torch.LongTensor(src_ids).unsqueeze(0).to(device)  # [1, src_len]

        # Encode
        encoder_outputs, hidden = model.encoder(src_tensor)
        src_mask = (src_tensor != 0)

        # Decode greedily
        ys = [tgt_vocab.stoi['<sos>']]
        for _ in range(max_len):
            input_token = torch.tensor([ys[-1]], device=device)  # [1]
            output, hidden = model.decoder(input_token, hidden, encoder_outputs, src_mask)
            pred = output.argmax(1).item()
            if pred == tgt_vocab.stoi['<eos>']:
                break
            ys.append(pred)

        # Convert to words
        translated = [tgt_vocab.itos.get(idx, '<unk>') for idx in ys[1:]]
        return ' '.join(translated)

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    try:
        model, src_vocab, tgt_vocab = load_model_and_vocab(device)
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        print("Make sure you have run train.py first to generate:")
        print("  - best_model.pt")
        print("  - src_vocab.pkl")
        print("  - tgt_vocab.pkl")
        sys.exit(1)

    print("\n🔤 Chinese-to-English Neural Machine Translation")
    print("Type a Chinese sentence to translate (or 'quit' to exit):")

    while True:
        try:
            user_input = input("\n🇨🇳 Input: ").strip()
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break
            if not user_input:
                continue

            translation = translate_sentence(model, src_vocab, tgt_vocab, user_input, device)
            print(f"🇺🇸 Output: {translation}")

        except KeyboardInterrupt:
            print("\n👋 Interrupted by user. Goodbye!")
            break

if __name__ == '__main__':
    main()