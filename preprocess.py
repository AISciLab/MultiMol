import pandas as pd
import numpy as np
import os
import warnings
import sys

warnings.filterwarnings('ignore')

def create_directory():
    os.makedirs('data/all', exist_ok=True)

def safe_load_drug_data():
    try:
        df = pd.read_csv('data/CTRPDrug.csv')
        df.columns = df.columns.str.strip()
        df['master_cpd_id'] = pd.to_numeric(df['master_cpd_id'], errors='coerce')
        df = df.dropna(subset=['master_cpd_id', 'SMILES'])
        df['master_cpd_id'] = df['master_cpd_id'].astype(int)
        drug_id_map = dict(zip(df['master_cpd_id'], df['SMILES']))
        return df, drug_id_map
    except Exception as e:
        sys.exit(f"Error loading drug data: {e}")

def safe_load_expression_data():
    try:
        expr_df = pd.read_csv('data/CTRP_expres.csv', index_col=0)
        new_index = pd.to_numeric(expr_df.index, errors='coerce')
        valid_mask = np.isfinite(new_index)
        cell_expr_df = expr_df[valid_mask]

        if len(cell_expr_df) == 0:
            sys.exit("Error: No valid numeric cell line IDs found.")

        cell_expr_df.index = new_index[valid_mask].astype(int)
        return cell_expr_df
    except Exception as e:
        sys.exit(f"Error loading expression data: {e}")

def load_and_process_data():
    print(">>> Loading raw files...")
    drug_df, drug_id_map = safe_load_drug_data()
    cell_expr_df = safe_load_expression_data()

    print(">>> Processing Response Data...")
    response_df = pd.read_csv('data/CTRPResponse.csv')
    response_df.columns = response_df.columns.str.strip()

    for col in ['master_cpd_id', 'master_ccl_id', 'area_under_curve']:
        response_df[col] = pd.to_numeric(response_df[col], errors='coerce')

    response_df = response_df.dropna(subset=['master_cpd_id', 'master_ccl_id', 'area_under_curve'])
    response_df['master_cpd_id'] = response_df['master_cpd_id'].astype(int)
    response_df['master_ccl_id'] = response_df['master_ccl_id'].astype(int)

    valid_drugs = set(drug_df['master_cpd_id'])
    valid_cells = set(cell_expr_df.index)

    mask = response_df['master_cpd_id'].isin(valid_drugs) & \
           response_df['master_ccl_id'].isin(valid_cells)
    data = response_df[mask].copy()

    if len(data) == 0:
        sys.exit("Error: No overlapping data found.")

    print(f"    Total Samples available: {len(data)}")

    unique_drug_ids = data['master_cpd_id'].unique()
    unique_cell_ids = data['master_ccl_id'].unique()

    drug_to_idx = {uid: i for i, uid in enumerate(unique_drug_ids)}
    cell_to_idx = {uid: i for i, uid in enumerate(unique_cell_ids)}

    drug_smiles_list = [drug_id_map[uid] for uid in unique_drug_ids]
    cell_features_matrix = cell_expr_df.loc[unique_cell_ids].values

    data['c_idx'] = data['master_ccl_id'].map(cell_to_idx)
    data['d_idx'] = data['master_cpd_id'].map(drug_to_idx)
    samples = data[['c_idx', 'd_idx', 'area_under_curve']].values

    print(">>> Saving processed files...")
    np.save('data/cell_features.npy', cell_features_matrix.astype(np.float32))

    with open('data/drug_smiles.txt', 'w') as f:
        for s in drug_smiles_list:
            f.write(s + '\n')

    np.save('data/all/cell_indices.npy', samples[:, 0].astype(int))
    np.save('data/all/drug_indices.npy', samples[:, 1].astype(int))
    np.save('data/all/labels.npy', samples[:, 2].astype(float))
    print(">>> Done.")

if __name__ == '__main__':
    create_directory()
    load_and_process_data()