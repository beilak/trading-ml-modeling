import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import PatchTSMixerConfig, PatchTSMixerForPrediction
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
import numpy as np



TARGET_COLUMN = "tbm_10d"

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"⚙️ Using device: {DEVICE}")


# ======================== FEATURES ========================

NO_IMPORTANCE_FEATURES = [
    # "is_quarter_start", "sma7_above_sma15", "macd_cross_signal", "sma3_cross_sma7",
    # "macd_hist_acceleration", "cbr_rate_change_flag", "bb_upper_breakout", "sma50_cross_sma200",
    # "sarimax_failed", "bb_lower_breakout", "bb_percent_b", "macd_state", "BBM_20_2.0_2.0",
    # "sma3_above_sma10", "sma3_above_sma7", "sma7_cross_sma15", "bb_width_norm", "sma3_cross_sma10",
    # "is_month_end", "sma70_cross_sma200", 'high_to_sma_30', 'sma_5',
    # 'intraday_move_norm', 'sma_20', 'sma_150', 'fft_abs_0', 'rolling_std_7',
    # 'wv_L0_mean', 'open_to_sma_15', 'close_to_sma_20', 'sarimax_pred_3d_to_today',
    # 'is_quarter_end', 'sma_7', 'sma_30', 'close_to_sma_40', 'sma_10', 'is_month_start',
    # 'sma50_above_sma200', 'sma20_above_sma50', 'sma20_cross_sma50',
]


def select_target_columns():
    return TARGET_COLUMN


def select_feature_columns(df: pd.DataFrame):
    features_to_remove = ["day_of_year", "week_of_year", "month"]
    tbm_cols = [c for c in df.columns if c.startswith("tbm_")]
    # cols_to_drop = ['Date', 'Ticker', 'Open', 'High', 'Low', 'Close', 'Volume'] \
    #                 + tbm_cols + features_to_remove + NO_IMPORTANCE_FEATURES
    cols_to_drop = ['Date', 'Ticker'] + tbm_cols + features_to_remove + NO_IMPORTANCE_FEATURES
    return [c for c in df.columns if c not in cols_to_drop]

    
# ---- 2. Dataset для временных рядов ----
class TimeSeriesDataset(Dataset):
    def __init__(self, X, y, lookback):
        # Преобразуем X в numpy, если это DataFrame
        self.X = X
        if isinstance(X, pd.DataFrame):
            self.X = X.values
        self.y = y
        if isinstance(y, (pd.Series, pd.DataFrame)):
            self.y = y.values
        
        # self.X = torch.tensor(self.X, dtype=torch.float32)        
        self.X = torch.from_numpy(np.array(self.X, dtype=np.float32, copy=True))
        self.y = torch.tensor(self.y, dtype=torch.long)
        self.lookback = lookback

    def __len__(self):
        return len(self.X) - self.lookback

    def __getitem__(self, idx):
        x_seq = self.X[idx:idx + self.lookback]
        y_val = self.y[idx + self.lookback - 1] if len(self.y) > 0 else torch.tensor(0)
        return x_seq, y_val





# ======================== MODEL ========================

# ---- 1. Конфигурация модели ----
def build_model(input_features: int, patch_len=32, num_classes=3):
    config = PatchTSMixerConfig(
        patch_len=patch_len,       # длина временного окна (lookback)
        num_input_channels=input_features,  # число признаков на шаг
        prediction_length=1,       # TBM = предсказание направления
        num_targets=num_classes,   # 3 класса: -1, 0, 1
        d_model=64,                # размер скрытых слоёв (можно увеличить)
        num_layers=2,              # глубина модели
        dropout=0.1
    )
    model = PatchTSMixerForPrediction(config)
    return model.to(DEVICE)


# ---- 3. Тренировка ----
def train_model_TSMixer(X, y, epochs=10, batch_size=32, lr=1e-3, lookback=32):
    model = build_model(input_features=X.shape[1], patch_len=lookback)
    dataset = TimeSeriesDataset(X, y, lookback)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)

            # Прямой проход
            outputs = model(past_values=xb).prediction_outputs  # правильное поле
            outputs = outputs.squeeze(1)  # [batch, num_classes]

            loss = criterion(outputs, yb)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs} | Loss: {total_loss/len(loader):.4f}")

    return model

# ======================== WALK FORWARD ========================

def walk_forward_train_TSMixer_model(df: pd.DataFrame, X: pd.DataFrame, y: pd.Series, epochs=10, lookback=60):
    n_splits = 5
    tscv = TimeSeriesSplit(n_splits=n_splits)
    results = []

    for i, (train_idx, test_idx) in enumerate(tscv.split(X)):
        print(f"\n=== Split {i+1}/{n_splits} ===")
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        model = train_model_TSMixer(X_train, y_train, epochs=epochs, lookback=lookback)
        model.eval()

        # Прогноз
        test_dataset = TimeSeriesDataset(X_test, y_test, lookback)
        preds, probs = [], []
        with torch.no_grad():
            for xb, _ in DataLoader(test_dataset, batch_size=32):
                xb = xb.to(DEVICE)
                # logits = model(inputs=xb).logits.squeeze(1)
                outputs = model(past_values=xb)
                logits = outputs.prediction_outputs.squeeze(1)
                preds.append(torch.argmax(logits, dim=1).cpu().numpy())
                probs.append(torch.softmax(logits, dim=1).cpu().numpy())
        y_pred = np.concatenate(preds)
        y_proba = np.concatenate(probs)

        # Преобразуем float → классы {-1, 0, 1}
        y_pred = np.array(y_pred)
        y_pred = np.where(y_pred > 0.33, 1, np.where(y_pred < -0.33, -1, 0))

        acc = accuracy_score(y_test[lookback:], y_pred)
        print(f"Accuracy: {acc:.4f}")
        print(classification_report(y_test[lookback:], y_pred, target_names=['-1', '0', '1']))

        # try:
        #     roc_auc = roc_auc_score(y_test[lookback:], y_proba, multi_class="ovr")
        # except ValueError:
        #     roc_auc = np.nan

        # Преобразуем классы {-1, 0, 1} → {0, 1, 2}
        print(f"{y_test[lookback:] = }")
        y_test_enc = np.vectorize({-1: 0, 0: 1, 1: 2}.get)(y_test[lookback:])

        try:
            roc_auc = roc_auc_score(y_test_enc, y_proba, multi_class="ovr")
        except ValueError:
            roc_auc = np.nan


        results.append({"Split": i+1, "Accuracy": acc, "ROC_AUC": roc_auc})

    results_df = pd.DataFrame(results)
    print("\n=== Summary ===")
    print(results_df)
    print(f"Mean Accuracy: {results_df['Accuracy'].mean():.4f}")
    return results_df




# def permutation_importance(model, X, y, baseline_acc, lookback=32, n_samples=500):
#     importances = {}
#     X_sample = X.sample(n=n_samples, random_state=42)
#     y_sample = y.loc[X_sample.index]

#     for col in X.columns:
#         X_perm = X_sample.copy()
#         X_perm[col] = np.random.permutation(X_perm[col].values)

#         # предсказание
#         dataset = TimeSeriesDataset(X_perm, y_sample, lookback)
#         loader = DataLoader(dataset, batch_size=32, shuffle=False)
#         preds = []
#         with torch.no_grad():
#             for xb, _ in loader:
#                 xb = xb.to(DEVICE)
#                 xb = xb[:, -model.config.patch_len:, :]
#                 outputs = model(past_values=xb)
#                 logits = outputs.prediction_outputs.squeeze(1)
#                 preds.append(torch.argmax(logits, dim=1).cpu().numpy())
#         y_pred = np.concatenate(preds)

#         acc = accuracy_score(np.vectorize({-1:0,0:1,1:2}.get)(y_sample[lookback:]), y_pred)
#         importances[col] = baseline_acc - acc

#     return pd.DataFrame(sorted(importances.items(), key=lambda x: x[1], reverse=True),
#                         columns=['feature', 'importance'])


# from sklearn.metrics import accuracy_score
# from torch.utils.data import DataLoader

# def get_baseline_acc_TSMixer(model, X_test, y_test, lookback=32, batch_size=32):
#     """
#     Вычисляет baseline accuracy для модели TSMixer на тестовом наборе.
#     """
#     model.eval()
#     test_dataset = TimeSeriesDataset(X_test, y_test, lookback)
#     test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

#     preds = []
#     with torch.no_grad():
#         for xb, _ in test_loader:
#             xb = xb.to(DEVICE)
#             outputs = model(past_values=xb)
#             logits = outputs.prediction_outputs.squeeze(1)
#             preds.append(torch.argmax(logits, dim=1).cpu().numpy())

#     y_pred = np.concatenate(preds)
#     y_true = y_test[lookback:].to_numpy()

#     baseline_acc = accuracy_score(y_true, y_pred)
#     print(f"Baseline accuracy: {baseline_acc:.4f}")
#     return baseline_acc
