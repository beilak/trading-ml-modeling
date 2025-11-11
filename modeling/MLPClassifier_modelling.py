from sklearn.model_selection import TimeSeriesSplit
import pandas as pd
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score


TARGET_COLUMN = 'tbm_10d'  # Наша цель для предсказания

NO_IMPORTANCE_FESTURES = [ ]


def select_target_columns():
    return TARGET_COLUMN


def select_feature_columns(df: pd.DataFrame):
    features_to_remove = ["day_of_year", "week_of_year", "month"]

    tbm_cols = [col for col in df.columns if col.startswith('tbm_')]
    cols_to_drop = ['Date', 'Ticker', 'Open', 'High', 'Low', 'Close', 'Volume'] \
                   + tbm_cols + features_to_remove + NO_IMPORTANCE_FESTURES

    feature_cols = [col for col in df.columns if col not in cols_to_drop]
    return feature_cols


# === Построение нейросети ===
# def build_model():
#     model = MLPClassifier(
#         hidden_layer_sizes=(256, 128, 64, 32),
#         activation='relu',
#         solver='adam',
#         learning_rate_init=0.0005,
#         max_iter=500,
#         random_state=42,
#         early_stopping=True,
#         validation_fraction=0.1,
#         alpha=1e-3,
#         verbose=False
#     )
#     return model


def build_model():
    # from sklearn.preprocessing import StandardScaler
    # from sklearn.pipeline import make_pipeline
    # from sklearn.neural_network import MLPClassifier

    # scaler = StandardScaler()
    mlp = MLPClassifier(
        hidden_layer_sizes=(512, 256, 128, 64, 32),
        activation='relu',
        solver='adam',
        learning_rate_init=0.0005,
        max_iter=1000,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.15,
        alpha=1e-2,
        n_iter_no_change=15,
        verbose=False
    )
    # return make_pipeline(scaler, mlp)
    return mlp




def train_model_MLPClassifier(X: pd.DataFrame, y: pd.DataFrame):
    """Обучаем финальную нейросеть на всех данных"""
    model = build_model()
    print("Обучение модели (MLPClassifier)...")

    if len(X) > 1000:
        X_train = X.iloc[-1000:]
        y_train = y.iloc[-1000:]
    else:
        X_train, y_train = X, y

    model.fit(X_train, y_train)
    return model


def walk_forward_train_MLPClasifier_model(df: pd.DataFrame, X: pd.DataFrame, y: pd.DataFrame):
    """Walk-Forward Validation"""
    n_splits = 5
    tscv = TimeSeriesSplit(n_splits=n_splits)

    print(f"\nНачинаем Walk-Forward валидацию (MLPClassifier) с {n_splits} сплитами...")

    all_scores = []
    results_summary = pd.DataFrame(columns=[
        'Split', 'Train Start', 'Train End', 'Test Start', 'Test End',
        'Accuracy', 'ROC AUC 1 (Buy)', 'ROC AUC -1 (Sell)', 'ROC AUC 0',
        'Precision (Buy)', 'Recall (Buy)', 'F1-score (Buy)',
        'Precision (Sell)', 'Recall (Sell)', 'F1-score (Sell)'
    ])

    for i, (train_index, test_index) in enumerate(tscv.split(X)):
        print(f"\n--- Сплит {i+1}/{n_splits} ---")
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        print(f"Размер обучающей выборки: {len(X_train)}")
        print(f"Размер тестовой выборки: {len(X_test)}")

        model = train_model_MLPClassifier(X_train, y_train)

        print("Оценка модели...")
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        all_scores.append(accuracy)

        print(f"\nТочность (Accuracy) на сплите {i+1}: {accuracy:.4f}")

        try:
            roc_auc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr', average=None)
            roc_auc_dict = {cls: auc for cls, auc in zip(model.classes_, roc_auc)}
        except Exception:
            roc_auc_dict = {cls: np.nan for cls in model.classes_}

        print("Детальный отчет по качеству:")
        print(classification_report(y_test, y_pred, target_names=['-1 (Sell)', '0 (Hold)', '1 (Buy)']))
        print(f"ROC AUC: {roc_auc_dict}")

        train_start_date = df.loc[train_index, 'Date'].min().strftime('%Y-%m-%d')
        train_end_date = df.loc[train_index, 'Date'].max().strftime('%Y-%m-%d')
        test_start_date = df.loc[test_index, 'Date'].min().strftime('%Y-%m-%d')
        test_end_date = df.loc[test_index, 'Date'].max().strftime('%Y-%m-%d')

        report = classification_report(y_test, y_pred, output_dict=True)
        buy_metrics = report.get('1', {})
        sell_metrics = report.get('-1', {})

        split_results = {
            'Split': i + 1,
            'Train Start': train_start_date,
            'Train End': train_end_date,
            'Test Start': test_start_date,
            'Test End': test_end_date,
            'Accuracy': accuracy,
            'ROC AUC 1 (Buy)': roc_auc_dict.get(1, np.nan),
            'ROC AUC -1 (Sell)': roc_auc_dict.get(-1, np.nan),
            'ROC AUC 0': roc_auc_dict.get(0, np.nan),
            'Precision (Buy)': buy_metrics.get('precision'),
            'Recall (Buy)': buy_metrics.get('recall'),
            'F1-score (Buy)': buy_metrics.get('f1-score'),
            'Precision (Sell)': sell_metrics.get('precision'),
            'Recall (Sell)': sell_metrics.get('recall'),
            'F1-score (Sell)': sell_metrics.get('f1-score'),
        }

        results_summary.loc[i] = split_results

    print("=" * 100)
    print(f"\nСредняя точность по всем сплитам: {np.mean(all_scores):.4f}")

    numeric_cols = results_summary.select_dtypes(include=np.number).columns
    results_summary[numeric_cols] = results_summary[numeric_cols].round(3)
    print(results_summary.to_string())
