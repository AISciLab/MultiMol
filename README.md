# MultiMol
MultiMol is a bidirectional attention network for multi-level representation learning for drug response prediction. MultiMol designs a dual-stream architecture to extract molecular representation. Specifically, a 1D sequence-aware molecular language model captures semantic information, while a 3D conformation-aware pre-training model encodes spatial geometric characteristics. Importantly, we propose a bidirectional cross-attention network that  integrates multi-modal representations into a complementary space. Furthermore, a bilinear attention network is developed to capture the bidirectional interaction between cell lines and drugs, thereby enhancing the performance of sensitivity prediction. 
<img width="12950" height="5092" alt="model_pictures" src="https://github.com/user-attachments/assets/24a83769-4044-40aa-9cda-8f3a960b3532" />



## Requirements

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Data Preparation

Ensure the raw data files `CTRP_expres.csv`, `CTRPDrug.csv`, and `CTRPResponse.csv` are placed in the `data/` directory. The data processing consists of two main steps:

**1. Preprocessing Cell Lines and Labels**
Run the following script to process cell line gene expression profiles, generate mapping indices, drug SMILES list, and response labels:

```bash
python preprocess.py
```

This will generate `cell_features.npy` and `drug_smiles.txt` in the `data/` directory, `cell_indices.npy`, `drug_indices.npy`, and `labels.npy` in the `data/all/` directory.

**2. Drug Feature Extraction**
Run the following script to generate multimodal drug embeddings using the `drug_smiles.txt` generated in the previous step:

```bash
python utils.py
```

This will generate `drug_feat_chemberta.npy` and `drug_feat_unimol.npy` in the `data/` directory.

## Usage

Run the experiments using `main.py`. The script automatically runs 5 random seeds and reports the Mean ± Std for metrics including R2, RMSE, Pearson, Spearman, MSE, and MEDAE

### Run Experiments

```bash
python main.py 
```
