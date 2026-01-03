# train_transformer.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import pickle
import time
import json
import os
from data_loader import NMTDataset
from models import build_model

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ----------------------------
# 1. 加载词汇表（与 RNN 相同）
# ----------------------------
with open('data/prepared_vocab_zh.pkl', 'rb') as f:
    vocab_zh = pickle.load(f)
with open('data/prepared_vocab_en.pkl', 'rb') as f:
    vocab_en = pickle.load(f)

# 获取词表大小和 pad index
src_vocab_size = len(vocab_zh)
tgt_vocab_size = len(vocab_en)
src_pad_idx = vocab_zh['<pad>']
tgt_pad_idx = vocab_en['<pad>']

print(f"✅ Chinese vocab size: {src_vocab_size}, pad_idx={src_pad_idx}")
print(f"✅ English vocab size: {tgt_vocab_size}, pad_idx={tgt_pad_idx}")

# ----------------------------
# 2. 配置（注意：Transformer 参数名不同）
# ----------------------------
config = {
    'model_type': 'transformer',
    'src_vocab_size': src_vocab_size,   # ✅ 新增
    'tgt_vocab_size': tgt_vocab_size,   # ✅ 新增
    'd_model': 512,
    'num_heads': 8,
    'num_layers': 4,
    'd_ff': 2048,
    'dropout': 0.1,
    'pos_enc_type': 'sinusoidal',
    'batch_size': 32,
    'epochs': 10,
    'lr': 0.0002,
    'src_pad_idx': src_pad_idx,
    'tgt_pad_idx': tgt_pad_idx,
}

# ----------------------------
# 3. 模型 & 优化器
# ----------------------------
model = build_model(config, device).to(device)
optimizer = optim.Adam(model.parameters(), lr=config['lr'], betas=(0.9, 0.98), eps=1e-9)
criterion = nn.CrossEntropyLoss(ignore_index=tgt_pad_idx)

# ----------------------------
# 4. 数据加载
# ----------------------------
def collate_fn(batch, src_pad_idx=0, tgt_pad_idx=0):
    from torch.nn.utils.rnn import pad_sequence
    src_seqs, tgt_seqs = zip(*batch)
    src_padded = pad_sequence(src_seqs, batch_first=True, padding_value=src_pad_idx)
    tgt_padded = pad_sequence(tgt_seqs, batch_first=True, padding_value=tgt_pad_idx)
    return src_padded, tgt_padded

train_dataset = NMTDataset('data/prepared_train.pt', src_pad_idx, tgt_pad_idx)
train_loader = DataLoader(
    train_dataset,
    batch_size=config['batch_size'],
    shuffle=True,
    collate_fn=lambda b: collate_fn(b, src_pad_idx, tgt_pad_idx)
)

# ----------------------------
# 5. 训练函数
# ----------------------------
def train_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for src, tgt in dataloader:
        src = src.to(device)
        tgt = tgt.to(device)

        optimizer.zero_grad()
        output = model(src, tgt[:, :-1])  # 输入 tgt[:-1]，预测 tgt[1:]

        output = output.reshape(-1, output.shape[-1])
        target = tgt[:, 1:].reshape(-1)

        loss = criterion(output, target)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
    return total_loss / len(dataloader)

# ----------------------------
# 6. 主循环（带日志保存）
# ----------------------------
if __name__ == "__main__":
    print("🚀 Start training Transformer-based Zh→En NMT...")
    
    output_dir = "outputs/transformer"
    os.makedirs(output_dir, exist_ok=True)
    train_losses = []

    for epoch in range(config['epochs']):
        start = time.time()
        loss = train_epoch(model, train_loader, optimizer, criterion, device)
        print(f"Epoch {epoch+1}/{config['epochs']} | Loss: {loss:.4f} | Time: {time.time()-start:.2f}s")
        train_losses.append(loss)

    # 保存模型
    model_path = os.path.join(output_dir, "model.pth")
    torch.save(model.state_dict(), model_path)
    print(f"✅ Model saved to {model_path}")

    # 保存训练日志（含 loss 曲线）
    log_path = os.path.join(output_dir, "train_log.json")
    log_data = {
        'model_type': 'transformer',
        'epochs': config['epochs'],
        'train_losses': train_losses,
        'final_loss': train_losses[-1],
        'config': config
    }
    with open(log_path, 'w') as f:
        json.dump(log_data, f, indent=4)
    print(f"✅ Training log saved to {log_path}")