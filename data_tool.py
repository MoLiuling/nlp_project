import os
import json
import re
import pickle
import argparse
from collections import Counter
from typing import List, Dict, Tuple

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import jieba

# ====================== 全局常量 ======================
# Special Tokens
PAD, SOS, EOS, UNK = 0, 1, 2, 3
SPECIAL_TOKENS = ['<pad>', '<sos>', '<eos>', '<unk>']

# Default paths (all under data/)
DEFAULT_DATA_DIR = 'data'
DEFAULT_OUTPUT_PREFIX = 'prepared_'


# ====================== 数据预处理核心函数 ======================
def clean_text(text: str, is_chinese: bool = True) -> str:
    """Text cleaning"""
    text = re.sub(r'\s+', ' ', text).strip()
    if is_chinese:
        # Keep Chinese, English, numbers and common punctuation to avoid over-cleaning
        text = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9.,!?;:\'"()\-\s]', '', text)
    else:
        # English keeps ASCII visible characters
        text = re.sub(r'[^\x20-\x7E]', '', text)
    return text


def tokenize_chinese(text: str) -> List[str]:
    return list(jieba.cut(text))


def tokenize_english(text: str) -> List[str]:
    # Enhanced English tokenization: separate punctuation from words
    text = re.sub(r'([.,!?;:\'"()\-])', r' \1 ', text)
    return [t for t in text.lower().split() if t]


class Vocabulary:
    def __init__(self):
        self.word2idx = {t: i for i, t in enumerate(SPECIAL_TOKENS)}
        self.idx2word = {i: t for i, t in enumerate(SPECIAL_TOKENS)}

    def build(self, texts: List[List[str]], max_size: int = 5000, min_freq: int = 1):
        """
        Build vocabulary
        1. Count all word frequencies
        2. Filter out words below min_freq
        3. Sort by frequency and take top max_size
        """
        counter = Counter(w for tokens in texts for w in tokens)

        # 1. Filter & sort (by frequency descending)
        valid_words = sorted(
            [w for w, f in counter.items() if f >= min_freq],
            key=lambda w: counter[w],
            reverse=True
        )

        # 2. Truncate (reserve space for special tokens)
        vocab_slots = max_size - len(SPECIAL_TOKENS)
        valid_words = valid_words[:vocab_slots]

        # 3. Fill
        for word in valid_words:
            if word not in self.word2idx:
                idx = len(self.word2idx)
                self.word2idx[word] = idx
                self.idx2word[idx] = word
                
        print(f"Vocab built: {len(self)} words (Original unique: {len(counter)})")
        return self

    def encode(self, tokens: List[str], add_sos=False, add_eos=False) -> List[int]:
        ids = [self.word2idx.get(t, UNK) for t in tokens]
        if add_sos: ids = [SOS] + ids
        if add_eos: ids = ids + [EOS]
        return ids

    def decode(self, ids: List[int]) -> List[str]:
        # Skip special symbols when decoding
        return [self.idx2word.get(i, '<unk>') for i in ids if i not in (PAD, SOS, EOS)]

    def __len__(self): return len(self.word2idx)

    def save(self, path):
        with open(path, 'wb') as f: pickle.dump(self.word2idx, f)

    @classmethod
    def load(cls, path):
        v = cls()
        with open(path, 'rb') as f: v.word2idx = pickle.load(f)
        v.idx2word = {i: w for w, i in v.word2idx.items()}
        return v


class NMTDataset(Dataset):
    def __init__(self, src_data, tgt_data):
        self.src, self.tgt = src_data, tgt_data

    def __len__(self): return len(self.src)

    def __getitem__(self, i):
        return torch.tensor(self.src[i]), torch.tensor(self.tgt[i])


def collate_fn(batch):
    src, tgt = zip(*batch)
    src_lens = [len(s) for s in src]
    tgt_lens = [len(t) for t in tgt]

    # Pad src and tgt
    src_pad = torch.zeros(len(batch), max(src_lens), dtype=torch.long)
    tgt_pad = torch.zeros(len(batch), max(tgt_lens), dtype=torch.long)

    for i, (s, t) in enumerate(zip(src, tgt)):
        src_pad[i, :len(s)] = s
        tgt_pad[i, :len(t)] = t

    return {
        'src': src_pad,
        'tgt': tgt_pad,
        'src_lengths': torch.tensor(src_lens)
    }


def get_dataloader(data_path: str, batch_size: int = 32, shuffle: bool = True):
    """Get DataLoader"""
    data = torch.load(data_path)
    return DataLoader(NMTDataset(data['src'], data['tgt']),
                      batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)


def load_jsonl(file_path: str) -> List[Dict]:
    """Load JSONL file"""
    data = []
    with open(file_path, encoding='utf-8') as f:
        for l in f:
            if l.strip():
                try:
                    data.append(json.loads(l))
                except:
                    pass
    return data


def tokenize_data(data: List[Dict]) -> Tuple[List[List[str]], List[List[str]]]:
    """Tokenize data"""
    zh_texts = [tokenize_chinese(clean_text(d['zh'], True)) for d in data]
    en_texts = [tokenize_english(clean_text(d['en'], False)) for d in data]
    return zh_texts, en_texts


def encode_data(zh_texts: List[List[str]], en_texts: List[List[str]],
                vocab_zh: Vocabulary, vocab_en: Vocabulary,
                max_length: int) -> Tuple[List, List]:
    """Encode and filter data"""
    src_enc, tgt_enc = [], []
    for zh, en in zip(zh_texts, en_texts):
        if len(zh) == 0 or len(en) == 0:
            continue
        src_ids = vocab_zh.encode(zh)
        tgt_ids = vocab_en.encode(en, add_sos=True, add_eos=True)
        if len(src_ids) <= max_length and len(tgt_ids) <= max_length:
            src_enc.append(src_ids)
            tgt_enc.append(tgt_ids)
    return src_enc, tgt_enc


def load_pretrained_embeddings(embed_path: str, vocab: Vocabulary, embed_dim: int = 300):
    print(f"Loading pretrained embeddings from {embed_path}...")
    # Use Xavier initialization / normal distribution by default instead of all zeros
    embeddings = torch.randn(len(vocab), embed_dim) * 0.1 
    found = 0

    with open(embed_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            parts = line.rstrip().split(' ')
            if len(parts) <= 2: continue
            word = parts[0]
            vec = parts[1:]
            if word in vocab.word2idx and len(vec) == embed_dim:
                idx = vocab.word2idx[word]
                embeddings[idx] = torch.tensor([float(v) for v in vec])
                found += 1

    # Only PAD must be 0, SOS/EOS/UNK should keep random initialization (with gradients)
    embeddings[PAD] = 0

    print(f"Found {found}/{len(vocab)} words.")
    return embeddings


def train_word2vec(texts: List[List[str]], vocab: Vocabulary, embed_dim: int = 300,
                   window: int = 5, min_count: int = 1, epochs: int = 10):
    from gensim.models import Word2Vec
    print(f"Training Word2Vec...")
    # Note: min_count here must be consistent or smaller than vocab's min_freq
    model = Word2Vec(sentences=texts, vector_size=embed_dim, window=window,
                     min_count=min_count, workers=4, epochs=epochs)

    embeddings = torch.randn(len(vocab), embed_dim) * 0.1 # Random initialization for uncovered words
    found = 0
    for word, idx in vocab.word2idx.items():
        if word in model.wv:
            embeddings[idx] = torch.tensor(model.wv[word])
            found += 1

    embeddings[PAD] = 0 # Only PAD is set to 0
    print(f"Word2Vec coverage: {found}/{len(vocab)}")
    return embeddings

# ====================== 预处理主函数 ======================
def prepare_data(data_dir: str = DEFAULT_DATA_DIR,
                 data_size: str = '10k',
                 vocab_size: int = 5000,
                 max_length: int = 128,
                 min_freq: int = 1,
                 embed_path_zh: str = None,
                 embed_path_en: str = None,
                 embed_dim: int = 300,
                 train_embeddings: bool = False):
    """
    Process dataset, generate prepared_train.pt, prepared_valid.pt, etc.
    Vocabulary is built only from training data
    Test set directly uses data/test.jsonl (no preprocessing needed)

    Args:
        data_dir: Root directory containing raw data (e.g., 'data/')
        data_size: '10k' or '100k'
    """
    # Ensure output dir exists
    os.makedirs(data_dir, exist_ok=True)

    # Determine file paths (directly under data/)
    train_file = os.path.join(data_dir, f'train_{data_size}.jsonl')
    valid_file = os.path.join(data_dir, 'valid.jsonl')

    # 1. Load data
    print(f"Loading training data from {train_file}...")
    train_data = load_jsonl(train_file)
    print(f"Loading validation data from {valid_file}...")
    valid_data = load_jsonl(valid_file)

    print(f"Data sizes: train={len(train_data)}, valid={len(valid_data)}")

    # 2. Tokenize
    print("Tokenizing...")
    train_zh, train_en = tokenize_data(train_data)
    valid_zh, valid_en = tokenize_data(valid_data)

    # 3. Build vocabularies from training data only
    print("Building vocabularies (from training data only)...")
    vocab_zh = Vocabulary().build(train_zh, vocab_size, min_freq)
    vocab_en = Vocabulary().build(train_en, vocab_size, min_freq)

    # Save vocabularies with prepared_ prefix
    vocab_zh.save(os.path.join(data_dir, f'{DEFAULT_OUTPUT_PREFIX}vocab_zh.pkl'))
    vocab_en.save(os.path.join(data_dir, f'{DEFAULT_OUTPUT_PREFIX}vocab_en.pkl'))

    # 4. Word embedding processing
    has_embeddings = False

    def handle_embedding(embed_path, vocab, texts, save_name):
        emb = None
        if embed_path and os.path.exists(embed_path):
            emb = load_pretrained_embeddings(embed_path, vocab, embed_dim)
        elif train_embeddings:
            emb = train_word2vec(texts, vocab, embed_dim)

        if emb is not None:
            torch.save(emb, os.path.join(data_dir, save_name))
            return True
        return False

    has_zh = handle_embedding(embed_path_zh, vocab_zh, train_zh, f'{DEFAULT_OUTPUT_PREFIX}embeddings_zh.pt')
    has_en = handle_embedding(embed_path_en, vocab_en, train_en, f'{DEFAULT_OUTPUT_PREFIX}embeddings_en.pt')
    has_embeddings = has_zh or has_en

    # 5. Encode datasets
    print("Encoding datasets...")

    train_src, train_tgt = encode_data(train_zh, train_en, vocab_zh, vocab_en, max_length)
    print(f"  Train: {len(train_src)} samples")

    valid_src, valid_tgt = encode_data(valid_zh, valid_en, vocab_zh, vocab_en, max_length)
    print(f"  Valid: {len(valid_src)} samples")

    # 6. Save datasets with prepared_ prefix
    torch.save({'src': train_src, 'tgt': train_tgt}, os.path.join(data_dir, f'{DEFAULT_OUTPUT_PREFIX}train.pt'))
    torch.save({'src': valid_src, 'tgt': valid_tgt}, os.path.join(data_dir, f'{DEFAULT_OUTPUT_PREFIX}valid.pt'))
    torch.save({'src': train_src, 'tgt': train_tgt}, os.path.join(data_dir, f'{DEFAULT_OUTPUT_PREFIX}data.pt'))

    print(f"Data saved to {data_dir}/")
    print(f"Note: Use data/test.jsonl for evaluation (no preprocessing needed)")

    # Save configuration
    config = {
        'src_vocab_size': len(vocab_zh),
        'tgt_vocab_size': len(vocab_en),
        'pad_idx': PAD,
        'sos_idx': SOS,
        'eos_idx': EOS,
        'unk_idx': UNK,
        'max_length': max_length,
        'embed_dim': embed_dim if has_embeddings else None,
        'data_size': data_size,
        'train_samples': len(train_src),
        'valid_samples': len(valid_src)
    }
    json.dump(config, open(os.path.join(data_dir, f'{DEFAULT_OUTPUT_PREFIX}config.json'), 'w'), indent=2)
    print("Config saved.")


# ====================== 数据查看主函数 ======================
def view_data(data_dir: str, num_samples: int = 5):
    """View preprocessed data samples"""
    # Load vocabularies
    vocab_zh = Vocabulary.load(os.path.join(data_dir, 'prepared_vocab_zh.pkl'))
    vocab_en = Vocabulary.load(os.path.join(data_dir, 'prepared_vocab_en.pkl'))
    
    # Load train data
    train_data = torch.load(os.path.join(data_dir, 'prepared_train.pt'))
    
    # Show samples
    print(f"\n=== Preprocessed Data Samples (first {num_samples}) ===")
    print(f"Source (Chinese) -> Target (English)")
    print("-" * 80)
    
    for i in range(min(num_samples, len(train_data['src']))):
        src_ids = train_data['src'][i]
        tgt_ids = train_data['tgt'][i]
        
        # Decode to text
        src_text = ' '.join(vocab_zh.decode(src_ids))
        tgt_text = ' '.join(vocab_en.decode(tgt_ids))
        
        print(f"\nSample {i+1}:")
        print(f"  ZH: {src_text}")
        print(f"  EN: {tgt_text}")
    
    # Show basic stats
    print("\n=== Basic Dataset Statistics ===")
    config_path = os.path.join(data_dir, 'prepared_config.json')
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = json.load(f)
        print(f"Source vocab size: {config['src_vocab_size']}")
        print(f"Target vocab size: {config['tgt_vocab_size']}")
        print(f"Max sequence length: {config['max_length']}")
        print(f"Train samples: {config['train_samples']}")
        print(f"Valid samples: {config['valid_samples']}")
    else:
        print("Config file not found, cannot show statistics")


# ====================== 主入口函数 ======================
def main():
    parser = argparse.ArgumentParser(description='NLP Midterm Project - Data Processing Tool')
    subparsers = parser.add_subparsers(dest='command', required=True, 
                                      help='Command to execute: prepare or view')
    
    # Prepare data subparser
    prepare_parser = subparsers.add_parser('prepare', help='Prepare preprocessed data')
    prepare_parser.add_argument('--data_dir', default=DEFAULT_DATA_DIR, help='Data directory (default: data)')
    prepare_parser.add_argument('--data', default='10k', choices=['10k', '100k'],
                                help='Dataset size: 10k or 100k')
    prepare_parser.add_argument('--vocab_size', type=int, default=5000)
    prepare_parser.add_argument('--max_length', type=int, default=64)
    prepare_parser.add_argument('--min_freq', type=int, default=1)
    prepare_parser.add_argument('--embed_path_zh', default=None)
    prepare_parser.add_argument('--embed_path_en', default=None)
    prepare_parser.add_argument('--embed_dim', type=int, default=300)
    prepare_parser.add_argument('--train_embeddings', action='store_true')
    
    # View data subparser
    view_parser = subparsers.add_parser('view', help='View preprocessed data samples')
    view_parser.add_argument('--data_dir', default='data', help='Data directory')
    view_parser.add_argument('--num_samples', type=int, default=5, help='Number of samples to display')
    
    args = parser.parse_args()
    
    if args.command == 'prepare':
        prepare_data(
            data_dir=args.data_dir,
            data_size=args.data,
            vocab_size=args.vocab_size,
            max_length=args.max_length,
            min_freq=args.min_freq,
            embed_path_zh=args.embed_path_zh,
            embed_path_en=args.embed_path_en,
            embed_dim=args.embed_dim,
            train_embeddings=args.train_embeddings
        )
    
    elif args.command == 'view':
        if not os.path.exists(args.data_dir):
            print(f"Error: Data directory does not exist: {args.data_dir}")
            print("Please run data preprocessing first:")
            print("  python data_tool.py prepare --data 10k")
            return
        view_data(args.data_dir, args.num_samples)

if __name__ == '__main__':
    main()