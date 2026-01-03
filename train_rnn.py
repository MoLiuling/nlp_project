# train_rnn.py
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
# 1. 加载词汇表
# ----------------------------
with open('data/prepared_vocab_zh.pkl', 'rb') as f:
    vocab_zh = pickle.load(f)  # word -> index

with open('data/prepared_vocab_en.pkl', 'rb') as f:
    vocab_en = pickle.load(f)

src_vocab_size = len(vocab_zh)
tgt_vocab_size = len(vocab_en)
src_pad_idx = vocab_zh['<pad>']  # 应该是 0
tgt_pad_idx = vocab_en['<pad>']  # 应该是 0

print(f"✅ Chinese vocab size: {src_vocab_size}, pad_idx: {src_pad_idx}")
print(f"✅ English vocab size: {tgt_vocab_size}, pad_idx: {tgt_pad_idx}")

# ----------------------------
# 2. 配置
# ----------------------------
config = {
    'model_type': 'rnn',
    'src_vocab_size': src_vocab_size,
    'tgt_vocab_size': tgt_vocab_size,
    'embed_dim': 256,
    'hidden_dim': 512,
    'num_layers': 2,
    'attn_type': 'additive',
    'dropout': 0.3,
    'batch_size': 32,
    'epochs': 10,
    'lr': 0.001,
    'teacher_forcing_ratio': 0.5,
    'src_pad_idx': src_pad_idx,
    'tgt_pad_idx': tgt_pad_idx,
}

# ----------------------------
# 3. 模型 & 优化器
# ----------------------------
model = build_model(config, device).to(device)
optimizer = optim.Adam(model.parameters(), lr=config['lr'])
criterion = nn.CrossEntropyLoss(ignore_index=tgt_pad_idx)  # 忽略 <pad> (index=0)

# ----------------------------
# 4. 数据加载
# ----------------------------
train_dataset = NMTDataset('data/prepared_train.pt', src_pad_idx, tgt_pad_idx)
train_loader = DataLoader(
    train_dataset,
    batch_size=config['batch_size'],
    shuffle=True,
    collate_fn=lambda b: collate_fn(b, src_pad_idx, tgt_pad_idx)
)

# 修改 collate_fn 支持自定义 pad_idx
def collate_fn(batch, src_pad_idx=0, tgt_pad_idx=0):
    src_seqs, tgt_seqs = zip(*batch)
    src_padded = torch.nn.utils.rnn.pad_sequence(src_seqs, batch_first=True, padding_value=src_pad_idx)
    tgt_padded = torch.nn.utils.rnn.pad_sequence(tgt_seqs, batch_first=True, padding_value=tgt_pad_idx)
    return src_padded, tgt_padded
# ----------------------------
# 5. 训练函数
# ----------------------------
def train_epoch(model, dataloader, optimizer, criterion, tf_ratio, device):
    model.train()
    total_loss = 0
    for src, tgt in dataloader:
        src = src.to(device)
        tgt = tgt.to(device)

        optimizer.zero_grad()
        output = model(src, tgt, teacher_forcing_ratio=tf_ratio)  # [B, L, vocab]

        # 预测 tgt[1:]，即跳过 <sos>
        output = output[:, 1:].reshape(-1, output.shape[-1])
        tgt = tgt[:, 1:].reshape(-1)

        loss = criterion(output, tgt)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
    return total_loss / len(dataloader)

# 6. 主循环
# ----------------------------
if __name__ == "__main__":
    print("🚀 Start training RNN-based Zh→En NMT...")
    
    # ➕ 新增：定义输出目录
    output_dir = "outputs/rnn"
    os.makedirs(output_dir, exist_ok=True)  # 自动创建文件夹（如果不存在）
    
    train_losses = []

    for epoch in range(config['epochs']):
        start = time.time()
        loss = train_epoch(model, train_loader, optimizer, criterion, config['teacher_forcing_ratio'], device)
        print(f"Epoch {epoch+1}/{config['epochs']} | Loss: {loss:.4f} | Time: {time.time()-start:.2f}s")
        train_losses.append(loss)

    # ➕ 保存模型到 outputs/rnn/
    model_path = os.path.join(output_dir, "model.pth")
    torch.save(model.state_dict(), model_path)
    print(f"✅ Model saved to {model_path}")

    # ➕ 保存训练日志到 outputs/rnn/
    log_path = os.path.join(output_dir, "train_log.json")
    log_data = {
        'model_type': 'rnn',
        'epochs': config['epochs'],
        'train_losses': train_losses,
        'final_loss': train_losses[-1],
        'config': config  # 可选：保存配置以便复现实验
    }
    with open(log_path, 'w') as f:
        json.dump(log_data, f, indent=4)
    print(f"✅ Training log saved to {log_path}")