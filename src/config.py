import torch
import os

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SEED = 42
BATCH_SIZE = 256
EPOCHS = 300
LR = 1e-4
HIDDEN_DIM = 256

DATA_DIR = 'data'
ALL_DATA_DIR = os.path.join(DATA_DIR, 'all')
CHECKPOINT_DIR = 'checkpoints'

CELL_FEAT_PATH = os.path.join(DATA_DIR, 'cell_features.npy')
DRUG_BERT_PATH = os.path.join(DATA_DIR, 'drug_feat_chemberta.npy')
DRUG_UNI_PATH = os.path.join(DATA_DIR, 'drug_feat_unimol.npy')
SMILES_PATH = os.path.join(DATA_DIR, 'drug_smiles.txt')

CELL_IDX_PATH = os.path.join(ALL_DATA_DIR, 'cell_indices.npy')
DRUG_IDX_PATH = os.path.join(ALL_DATA_DIR, 'drug_indices.npy')
LABEL_PATH = os.path.join(ALL_DATA_DIR, 'labels.npy')