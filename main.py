import numpy as np
import pandas as pd
import torch
import copy
import os
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from sklearn.metrics import mean_squared_error, r2_score, median_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from scipy.stats import pearsonr, spearmanr

from src.config import *
from src.models import DrugResponseModel
from src.dataset import DrugDataset, load_processed_data

torch.manual_seed(SEED)
np.random.seed(SEED)


def calculate_metrics(y_true, y_pred):
    return {
        'R2': r2_score(y_true, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
        'Pearson': pearsonr(y_true, y_pred)[0],
        'Spearman': spearmanr(y_true, y_pred)[0],
        'MSE': mean_squared_error(y_true, y_pred),
        'MEDAE': median_absolute_error(y_true, y_pred)
    }


def train_epoch(model, loader, criterion, optimizer):
    model.train()
    total_loss = 0
    for batch in loader:
        c, d1, d2, label = [x.to(DEVICE) for x in batch]
        optimizer.zero_grad()
        output = model(c, d1, d2).squeeze()
        loss = criterion(output, label)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def evaluate(model, loader, label_scaler):
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for batch in loader:
            c, d1, d2, label = [x.to(DEVICE) for x in batch]
            output = model(c, d1, d2).squeeze()
            if output.ndim == 0: output = output.unsqueeze(0)
            preds.extend(output.cpu().numpy().flatten())
            trues.extend(label.cpu().numpy().flatten())

    p = label_scaler.inverse_transform(np.array(preds).reshape(-1, 1)).flatten()
    t = label_scaler.inverse_transform(np.array(trues).reshape(-1, 1)).flatten()
    return t, p


def main():
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    c_idx, d_idx, y, c_feat, d1_feat, d2_feat = load_processed_data()

    label_scaler = StandardScaler()
    y_norm = label_scaler.fit_transform(y.reshape(-1, 1)).flatten()
    c_feat_raw = c_feat.numpy()

    df_all = pd.DataFrame({'d_idx': d_idx, 'label': y})
    drug_stats = df_all.groupby('d_idx')['label'].std()
    mask = np.isin(d_idx, drug_stats[drug_stats > drug_stats.mean()].index.values)

    c_idx_f, d_idx_f, y_f = c_idx[mask], d_idx[mask], y_norm[mask]
    indices = np.arange(len(y_f))
    all_results = {k: [] for k in ['R2', 'RMSE', 'Pearson', 'Spearman', 'MSE', 'MEDAE']}

    for run in range(5):
        print(f"\n{'=' * 20} Run {run + 1} / 5 {'=' * 20}")

        train_i, temp_i = train_test_split(indices, test_size=0.2, shuffle=True)
        val_i, test_i = train_test_split(temp_i, test_size=0.5, shuffle=True)

        scaler = StandardScaler()
        scaler.fit(c_feat_raw[np.unique(c_idx_f[train_i])])
        c_feat_s = torch.FloatTensor(scaler.transform(c_feat_raw))

        loaders = {
            'train': DataLoader(
                DrugDataset(c_idx_f[train_i], d_idx_f[train_i], y_f[train_i], c_feat_s, d1_feat, d2_feat),
                batch_size=BATCH_SIZE, shuffle=True, num_workers=0),  # Windows下建议0以防卡死
            'val': DataLoader(DrugDataset(c_idx_f[val_i], d_idx_f[val_i], y_f[val_i], c_feat_s, d1_feat, d2_feat),
                              batch_size=BATCH_SIZE, shuffle=False, num_workers=0),
            'test': DataLoader(DrugDataset(c_idx_f[test_i], d_idx_f[test_i], y_f[test_i], c_feat_s, d1_feat, d2_feat),
                               batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
        }

        model = DrugResponseModel(c_feat.shape[1], d1_feat.shape[1], d2_feat.shape[1]).to(DEVICE)
        optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
        criterion = torch.nn.HuberLoss()

        best_mse, best_state = float('inf'), None

        for epoch in range(EPOCHS):
            train_loss = train_epoch(model, loaders['train'], criterion, optimizer)

            print(f"Epoch [{epoch + 1:03d}/{EPOCHS}] Train Loss: {train_loss:.4f}", end="\r")

            if (epoch + 1) % 5 == 0 or epoch == EPOCHS - 1:
                t_val, p_val = evaluate(model, loaders['val'], label_scaler)
                val_mse = mean_squared_error(t_val, p_val)
                scheduler.step(val_mse)

                status = ""
                if val_mse < best_mse:
                    best_mse = val_mse
                    best_state = copy.deepcopy(model.state_dict())
                    status = " <- Best model saved!"

                print(f"Epoch [{epoch + 1:03d}/{EPOCHS}] Train Loss: {train_loss:.4f} | Val MSE: {val_mse:.4f}{status}")

        if best_state:
            model.load_state_dict(best_state)
            t_test, p_test = evaluate(model, loaders['test'], label_scaler)
            m = calculate_metrics(t_test, p_test)
            print(f"\nRun {run + 1} Test Results: R2: {m['R2']:.4f}, Pearson: {m['Pearson']:.4f}")
            for k in all_results: all_results[k].append(m[k])

    print(f"\n\n{'*' * 20} Final 5-Run Average {'*' * 20}")
    for k in all_results:
        print(f"{k:<10}: {np.mean(all_results[k]):.4f} ± {np.std(all_results[k]):.4f}")


if __name__ == '__main__':
    main()