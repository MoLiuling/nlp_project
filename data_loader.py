# data_loader.py
import torch
from torch.utils.data import Dataset

class NMTDataset(Dataset):
    def __init__(self, data_path, src_pad_idx=0, tgt_pad_idx=0):
        self.data = torch.load(data_path)
        self.src = self.data['src'] 
        self.tgt = self.data['tgt']

 # 用于调试：限制数据集大小为前 100 条
        # self.src = self.src[:100]
        # self.tgt = self.tgt[:100]

        self.src_pad_idx = src_pad_idx
        self.tgt_pad_idx = tgt_pad_idx

    def __len__(self):
        return len(self.src)

    def __getitem__(self, idx):
        # 确保返回的是 LongTensor
        src_tensor = torch.LongTensor(self.src[idx])
        tgt_tensor = torch.LongTensor(self.tgt[idx])
        return src_tensor, tgt_tensor