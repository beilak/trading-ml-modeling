import os
import json
import joblib
import pandas as pd
import numpy as np
from datetime import datetime, timezone
from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score

from lightgbm import LGBMClassifier

META_FEATURES = ['pred_proba_buy', 'pred_proba_sell', 'proba_diff', 'atr_14', 'rsi_14']
META_CONFIDENCE_THRESHOLD = 0.30

def build_meta_model():
    meta = LogisticRegression(class_weight='balanced', random_state=42)
    return meta


# # ==========================================================
# # --- УТИЛИТЫ ---
# # ==========================================================
def find_latest_model_artifacts():
    return "../models/lgbm_tbm10d"
    # metadata_files = [f for f in os.listdir(folder_path) if f.endswith("_metadata.json")]
    # if not metadata_files:
    #     raise FileNotFoundError("Нет файлов *_metadata.json в папке models.")
    # latest_metadata_file = max(metadata_files)
    # base_path = os.path.join(folder_path, latest_metadata_file.replace("_metadata.json", ""))
    # return base_path


def get_out_of_sample_predictions(X, y, primary_model, n_splits=5):
    print(f"Начинаем TimeSeriesSplit с {n_splits} сплитами...")
    cv = TimeSeriesSplit(n_splits=n_splits)
    oof_preds = pd.DataFrame(index=X.index, columns=['pred_proba_sell', 'pred_proba_hold', 'pred_proba_buy', 'primary_pred'])

    for i, (train_idx, test_idx) in enumerate(cv.split(X)):
        print(f"--- Сплит {i + 1}/{n_splits} ---")
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        primary_model.fit(X_train, y_train)
        probas = primary_model.predict_proba(X_test)
        preds = primary_model.predict(X_test)

        oof_preds.iloc[test_idx, 0:3] = probas
        oof_preds.iloc[test_idx, 3] = preds

    oof_preds.dropna(inplace=True)
    return oof_preds


# ==========================================================
# --- МЕТА-МОДЕЛЬ ---
# ==========================================================
def train_meta_model(oof_preds, df, target_col):
    full_df = oof_preds.join(df)
    df_filtered = full_df[full_df['primary_pred'] != 0].copy()
    if df_filtered.empty:
        print("⚠️ Нет торговых сигналов для обучения мета-модели.")
        return None

    df_filtered['y_meta'] = (df_filtered['primary_pred'] == df_filtered[target_col]).astype(int)
    df_filtered['proba_diff'] = df_filtered['pred_proba_buy'] - df_filtered['pred_proba_sell']

    X_meta = df_filtered[META_FEATURES]
    y_meta = df_filtered['y_meta']

    model = LogisticRegression(class_weight='balanced', random_state=42)
    model.fit(X_meta, y_meta)
    return model


def validate_meta_model(X_meta, y_meta, n_splits=5):
    tscv = TimeSeriesSplit(n_splits=n_splits)
    aucs, f1s = [], []

    for i, (train_idx, test_idx) in enumerate(tscv.split(X_meta)):
        X_train, X_test = X_meta.iloc[train_idx], X_meta.iloc[test_idx]
        y_train, y_test = y_meta.iloc[train_idx], y_meta.iloc[test_idx]

        if len(y_test.unique()) < 2:
            print(f"⚠️ Сплит {i + 1}: один класс, пропуск.")
            continue

        # meta = LogisticRegression(class_weight='balanced', random_state=42)
        meta = build_meta_model()
        meta.fit(X_train, y_train)

        y_proba = meta.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_proba)
        aucs.append(auc)

        y_pred = meta.predict(X_test)
        rep = classification_report(y_test, y_pred, output_dict=True)
        f1s.append(rep['1']['f1-score'])

        print(f"Сплит {i + 1}: AUC={auc:.4f}")

    print(f"Средний AUC={np.mean(aucs):.4f}, F1={np.mean(f1s):.4f}")
    return np.mean(aucs), np.mean(f1s)


def evaluate_final_strategy(oof_preds, df, meta_model, target_col):
    full_df = oof_preds.join(df)
    signals = full_df[full_df['primary_pred'] != 0].copy()
    signals['proba_diff'] = signals['pred_proba_buy'] - signals['pred_proba_sell']

    X_meta = signals[META_FEATURES]
    meta_proba = meta_model.predict_proba(X_meta)[:, 1]
    approved = signals[meta_proba > META_CONFIDENCE_THRESHOLD]

    print(f"✅ Всего сигналов: {len(signals)}, одобрено мета-моделью: {len(approved)}")

    y_true = approved[target_col].astype(int)
    y_pred = approved['primary_pred'].astype(int)

    report = classification_report(
        y_true, y_pred, labels=[-1, 0, 1],
        target_names=['Sell', 'Hold', 'Buy'], zero_division=0
    )
    print(report)
    acc = accuracy_score(y_true, y_pred)
    print(f"Win Rate: {acc:.2%}")
    return acc


# def save_meta_model(meta_model, target_col, base_model_path, output_folder="../models/meta"):
#     os.makedirs(output_folder, exist_ok=True)
#     filename = "final_model"

#     path_model = os.path.join(output_folder, f"{filename}.joblib")
#     joblib.dump(meta_model, path_model)

#     metadata = {
#         "meta_model_name": "trade_signal_filter",
#         "meta_model_class": type(meta_model).__name__,
#         "primary_model_used": os.path.basename(base_model_path),
#         "training_timestamp_utc": datetime.now(timezone.utc).isoformat(),
#         "meta_features_used": META_FEATURES,
#         "meta_confidence_threshold": META_CONFIDENCE_THRESHOLD,
#         "model_parameters": meta_model.get_params(),
#         "target_column": target_col,
#     }
#     path_meta = os.path.join(output_folder, f"{filename}_metadata.json")
#     with open(path_meta, "w", encoding="utf-8") as f:
#         json.dump(metadata, f, indent=4, ensure_ascii=False)

#     print(f"💾 Мета-модель сохранена в {path_model}")
#     print(f"ℹ️ Метаданные сохранены в {path_meta}")
