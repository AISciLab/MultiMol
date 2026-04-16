import torch
import numpy as np
import os
from torch.utils.data import Dataset
from .config import *


class DrugDataset(Dataset):
    def __init__(self, c_idx, d_idx, labels, c_feat, d1_feat, d2_feat):
        self.c_idx = c_idx
        self.d_idx = d_idx
        self.labels = torch.FloatTensor(labels)
        self.c_feat = c_feat
        self.d1_feat = d1_feat
        self.d2_feat = d2_feat

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return (self.c_feat[self.c_idx[idx]],
                self.d1_feat[self.d_idx[idx]],
                self.d2_feat[self.d_idx[idx]],
                self.labels[idx])


def load_processed_data():
    required = [CELL_FEAT_PATH, DRUG_BERT_PATH, DRUG_UNI_PATH, CELL_IDX_PATH, DRUG_IDX_PATH, LABEL_PATH]
    for f in required:
        if not os.path.exists(f):
            raise FileNotFoundError(f"Missing: {f}")

    c_feat = torch.FloatTensor(np.load(CELL_FEAT_PATH))
    d1_feat = torch.FloatTensor(np.load(DRUG_BERT_PATH))
    d2_feat = torch.FloatTensor(np.load(DRUG_UNI_PATH))
    c_idx = np.load(CELL_IDX_PATH)
    d_idx = np.load(DRUG_IDX_PATH)
    labels = np.load(LABEL_PATH)
    return c_idx, d_idx, labels, c_feat, d1_feat, d2_feat