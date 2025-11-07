from sklearn.model_selection import TimeSeriesSplit
import pandas as pd
import numpy as np
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score


TARGET_COLUMN = 'tbm_10d' # Наша цель для предсказания!

NO_IMPORTANCE_FESTURES = [
    "is_quarter_start",
    "sma7_above_sma15",
    "macd_cross_signal",
    "sma3_cross_sma7",
    "macd_hist_acceleration",
    "cbr_rate_change_flag",
    "bb_upper_breakout",
    "sma50_cross_sma200",
    "sarimax_failed",
    "bb_lower_breakout",
    "bb_percent_b",
    "macd_state",
    "BBM_20_2.0_2.0",
    "sma3_above_sma10",
    "sma3_above_sma7",
    "sma7_cross_sma15",
    "bb_width_norm",
    "sma3_cross_sma10",
    "is_month_end",
    "sma70_cross_sma200",
    'high_to_sma_30', 'sma_5', 
    'intraday_move_norm', 'sma_20', 'sma_150', 'fft_abs_0', 'rolling_std_7',
    'wv_L0_mean', 'open_to_sma_15', 'close_to_sma_20', 'sarimax_pred_3d_to_today', 
    'is_quarter_end', 'sma_7', 'sma_30', 'close_to_sma_40', 'sma_10', 'is_month_start',
    'sma50_above_sma200', 'sma20_above_sma50', 'sma20_cross_sma50',
]


def select_target_columns():
    return TARGET_COLUMN


def select_feature_columns(df: pd.DataFrame):
    # Определяем признаки (X) и цель (y)
    features_to_remove = ["day_of_year", "week_of_year", "month"]

    tbm_cols = [col for col in df.columns if col.startswith('tbm_')]
    cols_to_drop = ['Date', 'Ticker', 'Open', 'High', 'Low', 'Close', 'Volume'] + tbm_cols + features_to_remove + NO_IMPORTANCE_FESTURES

    feature_cols = [col for col in df.columns if col not in cols_to_drop]
    return feature_cols

def build_model():
    # Обучаем финальную модель на всех данных
    model = LGBMClassifier(
            objective='multiclass', # Важно указать, что у нас 3 класса
            n_estimators=100,
            max_depth=10,
            random_state=42,
            class_weight='balanced',
            n_jobs=-1,
            verbosity=-1 
        )
    return model


def train_model_LGBMClassifier(X: pd.DataFrame, y: pd.DataFrame):
    # Обучаем финальную модель на всех данных
    model = build_model()
    print("Обучение модели...")
    # Берём последние 1000 записей
    if len(X) > 1000:
        X_train = X.iloc[-1000:]
        y_train = y.iloc[-1000:]
    else:
        X_train = X
        y_train = y
    model.fit(X, y)
    return model


def walk_forward_train_LGBMClasifier_model(df: pd.DataFrame, X: pd.DataFrame, y: pd.DataFrame):
    """
    params: df: Полный dataframe, X: Фичи, y: Таргет
    """
    # --- Правильное разделение данных (Walk-Forward Validation) ---
    # Мы будем использовать 5 "сдвигов". Модель будет обучаться на части данных,
    # а тестироваться на следующем, более новом блоке.
    n_splits = 5
    tscv = TimeSeriesSplit(n_splits=n_splits)

    print(f"\nНачинаем Walk-Forward валидацию с {n_splits} сплитами...")

    # --- Обучение и оценка модели в цикле ---

    all_scores = []

    # для хранения детальных результатов
    results_summary = pd.DataFrame(
        columns=['Split', 'Train Start', 'Train End', 'Test Start', 'Test End', 
                'Accuracy', 'ROC AUC 1 (Buy)', 'ROC AUC -1 (Sell)', 'ROC AUC 0', 'Precision (Buy)', 'Recall (Buy)', 'F1-score (Buy)',
                'Precision (Sell)', 'Recall (Sell)', 'F1-score (Sell)']
    )

    for i, (train_index, test_index) in enumerate(tscv.split(X)):
        print(f"\n--- Сплит {i+1}/{n_splits} ---")
        
        # Разделяем данные на обучающую и тестовую выборки
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        
        print(f"Размер обучающей выборки: {len(X_train)}")
        print(f"Размер тестовой выборки: {len(X_test)}")
        
        model = train_model_LGBMClassifier(X_train, y_train)
        
        # Обучаем модель
        # print("Обучение модели...")
        # model.fit(X_train, y_train)
        
        # Делаем предсказания
        print("Оценка модели...")
        y_pred = model.predict(X_test)
        # print("Получение вероятностей...")
        y_pred_proba = model.predict_proba(X_test)
        
        # Оцениваем качество
        accuracy = accuracy_score(y_test, y_pred)
        # accuracy = accuracy_score(y_test, y_pred_filtered)
        all_scores.append(accuracy)
        
        print(f"\nТочность (Accuracy) на сплите {i+1}: {accuracy:.4f}")
        # print(f"\nТочность (Accuracy) на сплите {i+1} (с порогом {CONFIDENCE_THRESHOLD}): {accuracy:.4f}")
        
        roc_auc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr', average=None)
        roc_auc_dict = {cls: auc_value for cls, auc_value in zip(model.classes_, roc_auc)}


        # Выводим детальный отчет по качеству для каждого класса
        print("Детальный отчет по качеству (Classification Report):")
        print(classification_report(y_test, y_pred, target_names=['-1 (Sell)', '0 (Hold)', '1 (Buy)']))
        # print(f"📊 ROC AUC:   {roc_auc:.3f}")
        print(f"📊 ROC AUC -1:   {roc_auc_dict[-1]}")
        print(f"📊 ROC AUC 0:   {roc_auc_dict[0]}")
        print(f"📊 ROC AUC 1:   {roc_auc_dict[1]}")
        # print(classification_report(y_test, y_pred_filtered, target_names=['-1 (Sell)', '0 (Hold)', '1 (Buy)']))

        # --- Сбор данных для сводной таблицы ---
        # Получаем даты начала и конца для обучающей и тестовой выборок
        train_start_date = df.loc[train_index, 'Date'].min().strftime('%Y-%m-%d')
        train_end_date = df.loc[train_index, 'Date'].max().strftime('%Y-%m-%d')
        test_start_date = df.loc[test_index, 'Date'].min().strftime('%Y-%m-%d')
        test_end_date = df.loc[test_index, 'Date'].max().strftime('%Y-%m-%d')

        # Извлекаем метрики из отчета
        report = classification_report(y_test, y_pred, output_dict=True)
        # report = classification_report(y_test, y_pred_filtered, output_dict=True)
        buy_metrics = report.get('1', {})  # Используем .get() для безопасности, если класса нет
        sell_metrics = report.get('-1', {})

        # Создаем словарь с результатами для текущего сплита
        split_results = {
            'Split': i + 1,
            'Train Start': train_start_date,
            'Train End': train_end_date,
            'Test Start': test_start_date,
            'Test End': test_end_date,
            'Accuracy': accuracy,
            'ROC AUC 1 (Buy)': roc_auc_dict[1], 
            'ROC AUC -1 (Sell)': roc_auc_dict[-1], 
            'ROC AUC 0': roc_auc_dict[0],
            'Precision (Buy)': buy_metrics.get('precision'),
            'Recall (Buy)': buy_metrics.get('recall'),
            'F1-score (Buy)': buy_metrics.get('f1-score'),
            # 'Support (Buy)': buy_metrics.get('support'),
            'Precision (Sell)': sell_metrics.get('precision'),
            'Recall (Sell)': sell_metrics.get('recall'),
            'F1-score (Sell)': sell_metrics.get('f1-score'),
            # 'Support (Sell)': sell_metrics.get('support'),
        }

        results_summary.loc[i] = split_results





    print("="*100)




    print(f"\nСредняя точность по всем сплитам: {np.mean(all_scores):.4f}")



    print("\n\n--- Сводная таблица результатов по сплитам ---")
    # Округляем числовые колонки для лучшей читаемости
    numeric_cols = results_summary.select_dtypes(include=np.number).columns
    results_summary[numeric_cols] = results_summary[numeric_cols].round(3)
    print(results_summary.to_string()) # .to_string() выведет всю таблицу без обрезки
