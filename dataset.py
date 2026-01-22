import numpy as np
import torch
from torch.utils.data import Dataset

labels = ['H','B','E','G','I','P','T','S','.']
label2id = {l:i for i,l in enumerate(labels)}

class SSData(Dataset):
    def __init__(self, df, emb_dir="embeddings"):
        self.df = df
        self.emb_dir = emb_dir

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        pid, aa, pos_str = row.id.split("_")
        pos = int(pos_str)
        emb = np.load(f"{self.emb_dir}/{pid}.npy")
        emb_index = pos + 1

        if not (1 <= emb_index < len(emb) - 1):
            x = np.zeros(480, dtype=np.float32)
        else:
            x = emb[emb_index]

        y = label2id[row.secondary_structure] if "secondary_structure" in row else -1
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y)
