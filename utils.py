import os
import sys
from unittest.mock import MagicMock
try:
    from rdkit.Chem.Draw import rdMolDraw2D
except ImportError:
    mock = MagicMock()
    sys.modules["rdkit.Chem.Draw.rdMolDraw2D"] = mock
    sys.modules["rdkit.Chem.Draw"] = mock
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import numpy as np
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer
from tqdm import tqdm
import warnings
from rdkit import Chem
warnings.filterwarnings('ignore')
from unimol_tools import UniMolRepr
class DrugFeatureExtractor:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f">>> Device set to: {self.device}")

    def get_chemberta_features(self, smiles_list, model_name="seyonec/ChemBERTa-zinc-base-v1"):
        print(">>> Extracting ChemBERTa features...")
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForMaskedLM.from_pretrained(model_name)
            model.to(self.device)
            model.eval()
        except Exception as e:
            print(f"Error loading ChemBERTa model: {e}")
            return np.zeros((len(smiles_list), 768))

        features = []
        batch_size = 32

        for i in tqdm(range(0, len(smiles_list), batch_size), desc="ChemBERTa"):
            batch_smiles = smiles_list[i:i + batch_size]
            try:
                inputs = tokenizer(batch_smiles, padding=True, truncation=True,
                                   return_tensors="pt", max_length=512)
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

                with torch.no_grad():
                    outputs = model(**inputs, output_hidden_states=True)
                    hidden_states = outputs.hidden_states[-1]
                    embedding = torch.mean(hidden_states, dim=1)
                    features.append(embedding.cpu().numpy())
            except Exception as e:
                print(f"Error in batch {i}: {e}")
                features.append(np.zeros((len(batch_smiles), 768)))

        return np.vstack(features)

    def get_unimol_features(self, smiles_list):
        print(">>> Extracting UniMol features...")

        rdkit_processed_smiles = []
        for s in smiles_list:
            try:
                mol = Chem.MolFromSmiles(s)
                if mol:
                    rdkit_processed_smiles.append(Chem.MolToSmiles(mol, isomericSmiles=True))
                else:
                    rdkit_processed_smiles.append(s)
            except:
                rdkit_processed_smiles.append(s)

        try:
            clf = UniMolRepr(
                data_type='molecule',
                remove_hs=False,
                use_gpu=True,
                model_name='unimolv2',
                model_size='84m'
            )
            result = clf.get_repr(rdkit_processed_smiles, return_atomic_reprs=False)
        except Exception as e:
            print(f"UniMol Init/Run Error: {e}")
            return np.zeros((len(smiles_list), 512))

        cls_repr_list = []
        if isinstance(result, dict):
            key = 'cls_repr' if 'cls_repr' in result else list(result.keys())[0]
            cls_repr_list = result[key]
        elif isinstance(result, list):
            cls_repr_list = result

        cleaned_list = []
        valid_dim = 512

        for item in cls_repr_list:
            if item is not None:
                valid_dim = len(item)
                break

        for item in cls_repr_list:
            if item is None:
                cleaned_list.append(np.zeros(valid_dim))
            else:
                cleaned_list.append(item)

        return np.array(cleaned_list)


def main():
    if not os.path.exists('data/drug_smiles.txt'):
        print("Error: data/drug_smiles.txt not found. Run preprocess_data.py first.")
        return

    with open('data/drug_smiles.txt', 'r') as f:
        smiles_list = [line.strip() for line in f if line.strip()]

    print(f"Total SMILES to process: {len(smiles_list)}")

    extractor = DrugFeatureExtractor()

    feat_chemberta = extractor.get_chemberta_features(smiles_list)
    print(f"ChemBERTa Shape: {feat_chemberta.shape}")

    feat_unimol = extractor.get_unimol_features(smiles_list)
    print(f"UniMol Shape: {feat_unimol.shape}")

    if len(feat_chemberta) != len(smiles_list) or len(feat_unimol) != len(smiles_list):
        print("Warning: Feature length mismatch!")

    np.save('data/drug_feat_chemberta.npy', feat_chemberta.astype(np.float32))
    np.save('data/drug_feat_unimol.npy', feat_unimol.astype(np.float32))
    print(">>> Features saved successfully!")


if __name__ == '__main__':
    main()