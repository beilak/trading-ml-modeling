import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

from sklearn.ensemble import VotingClassifier
from sklearn.preprocessing import LabelEncoder
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from LGBMClassifier_modeling import build_model as LGBMClassifierBuilder
from LGBMClassifier_modeling import select_feature_columns as LGBMClassifier_select_feature_columns

from MLPClassifier_modelling import build_model as MLPClassifierBuilder
from MLPClassifier_modelling import select_feature_columns as MLPClassifier_select_feature_columns

from scipy.stats import mode

from sklearn.tree import plot_tree
import matplotlib.pyplot as plt


TARGET_COLUMN = 'tbm_10d' # Наша цель для предсказания!


def select_target_columns():
    return TARGET_COLUMN

def select_feature_columns(df):
    return list(set(LGBMClassifier_select_feature_columns(df) + MLPClassifier_select_feature_columns(df)))


class FeatureAwareVotingClassifier(VotingClassifier):
    """
    VotingClassifier с поддержкой моделей, использующих разные наборы фичей.
    Для каждой модели используется внешний метод select_feature_columns(model_name),
    который возвращает список колонок для модели.
    """
    def __init__(self, estimators, voting='hard'):
        """
        estimators: list of (name, model) tuples
        voting: 'hard' или 'soft'
        select_feature_columns: callable(name) -> list of feature names
        """
        super().__init__(estimators=estimators, voting=voting)



    def fit(self, X, y, **fit_params):
        """
        X: pd.DataFrame
        y: pd.Series / np.array
        """        

        self.le_ = LabelEncoder().fit(y)
        self.classes_ = self.le_.classes_
        for name, model, select_feature_columns in self.estimators:
            cols = select_feature_columns
            model.fit(X[cols], y, **fit_params)
        


        self.fitted_ = True
        return self

    def _get_model_predictions(self, X):
        """
        Получаем предсказания каждой модели с учетом её колонок.
        Для soft-voting приводим вероятности к единому набору классов [-1,0,1].
        """
        predictions = []
        target_classes = [-1, 0, 1]

        for name, model, select_feature_columns in self.estimators:
            cols = select_feature_columns
            if self.voting == "soft":
                proba = model.predict_proba(X[cols])
                # Приводим вероятности к единому порядку классов
                proba_fixed = np.zeros((proba.shape[0], len(target_classes)))
                for i, cls in enumerate(model.classes_):
                    idx = target_classes.index(cls)
                    proba_fixed[:, idx] = proba[:, i]
                predictions.append(proba_fixed)
            else:
                pred = model.predict(X[cols])
                predictions.append(pred)
        return predictions

    def predict(self, X):
        predictions = self._get_model_predictions(X)
        target_classes = np.array([-1, 0, 1])

        if self.voting == "soft":
            avg_proba = np.mean(predictions, axis=0)
            y_pred_idx = np.argmax(avg_proba, axis=1)
            return target_classes[y_pred_idx]
        else:
            predictions = np.array(predictions).T
            maj_vote, _ = mode(predictions, axis=1)
            return maj_vote.ravel()

    def predict_proba(self, X):
        if self.voting != "soft":
            raise AttributeError("predict_proba is not available when voting='hard'")
        predictions = self._get_model_predictions(X)
        return np.mean(predictions, axis=0)




def build_model():
    estimators = [
        ("LGBMClassifier", LGBMClassifierBuilder(), LGBMClassifier_select_feature_columns),
        ("MLPClassifier", MLPClassifierBuilder(), MLPClassifier_select_feature_columns),
    ]

    ensemble_model = FeatureAwareVotingClassifier(
        estimators=estimators,
        voting="soft",  #hard или "soft" если хотим вероятности
    )

    return ensemble_model



def train_model(X: pd.DataFrame, y: pd.DataFrame):
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



def walk_forward_train_ensemble_model(df: pd.DataFrame, X: pd.DataFrame, y: pd.DataFrame):
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
        
        model = train_model(X_train, y_train)
        
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
