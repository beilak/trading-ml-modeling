import pandas as pd
import datetime
from LGBMClassifier_modeling import build_model, select_feature_columns, select_target_columns, train_model_LGBMClassifier
import numpy as np
from abc import ABC, abstractmethod

import os
import json
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod

import os
import json
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
from abc import ABC, abstractmethod

from meta_model_trainer import build_meta_model, train_meta_model, META_CONFIDENCE_THRESHOLD


# ==============================================================
# 🎯 Базовый класс стратегии
# ==============================================================

class BaseStrategy(ABC):
    """
    Базовый класс стратегии.
    Ожидает DataFrame с колонками ['Date', 'Ticker', ...].
    """

    def __init__(self, df: pd.DataFrame):
        if not {"Date", "Ticker"}.issubset(df.columns):
            raise ValueError("DataFrame должен содержать колонки 'Date' и 'Ticker'")

        self.df = df.copy()
        self.df["Date"] = pd.to_datetime(self.df["Date"])
        self.df.sort_values(["Date", "Ticker"], inplace=True)
        self.df.reset_index(drop=True, inplace=True)

        self.tickers = self.df["Ticker"].unique().tolist()
        print(f"✅ Загружено {len(self.df):,} строк по {len(self.tickers)} тикерам.")

    @abstractmethod
    def generate_signals(self, current_date: pd.Timestamp, tickers: list) -> pd.Series:
        """
        Возвращает Series сигналов для заданной даты.
        index = tickers, values ∈ {1, 0, -1}
        """
        ...


# ==============================================================
# 🎲 Случайная стратегия (заглушка)
# ==============================================================

class RandomStrategy(BaseStrategy):
    def __init__(self, df: pd.DataFrame, seed: int = 42):
        super().__init__(df)
        self.seed = seed
        np.random.seed(seed)

    def generate_signals(self, current_date: pd.Timestamp, tickers: list) -> pd.Series:
        """
        Генерирует случайные сигналы 1 (покупка) / 0 (не держать).
        """
        # фильтруем данные на текущую дату, если нужно
        daily_data = self.df[self.df["Date"] == current_date]

        # возвращаем сигналы для указанных тикеров
        signals = pd.Series(
            np.random.randint(0, 2, len(tickers)),
            index=tickers
        )
        return signals

class BaseStrategyEmptyDataError(Exception):
    ...


class LGBMClassifierStrategy(BaseStrategy):
    def __init__(self, df: pd.DataFrame):
        super().__init__(df)
        self.last_train_date = None     # когда последний раз обучали модель
        self.model_cache = None         # последняя обученная модель

    def model(self, df: pd.DataFrame):
        """
        Обучает или возвращает кэшированную модель.
        """
        last_date = df["Date"].max()

        # # ⚙️ Проверяем, нужно ли переобучать
        # if self.last_train_date is not None:
        #     # если с момента последнего обучения меньше месяца — не переобучаем
        #     if (last_date - self.last_train_date).days < 30 and self.model_cache is not None:
        #         print(f"Используем кэшированную модель от {self.last_train_date.date()}")
        #         return self.model_cache

        # 🔄 Переобучаем модель
        train_df = df[df["Date"] < last_date].copy()
        val_df   = df[df["Date"] == last_date].copy()

        train_df.dropna(subset=[select_target_columns()], inplace=True)
        train_df.reset_index(drop=True, inplace=True)
        print(f"\nОбучаем модель на {len(train_df)} строк (до {last_date.date()})")

        if train_df.empty or val_df.empty:
            raise BaseStrategyEmptyDataError

        X = train_df[select_feature_columns(train_df)]
        y = train_df[select_target_columns()]
        model = train_model_LGBMClassifier(X, y)

        # 💾 Сохраняем в кэш
        self.model_cache = model
        self.last_train_date = last_date

        return model

    def generate_signals(self, current_date: pd.Timestamp, tickers: list) -> pd.Series:
        """
        Генерация сигналов с использованием кэшированной модели.
        """
        try:
            model = self.model(self.df[self.df["Date"] <= current_date])
        except BaseStrategyEmptyDataError:
            return pd.Series(0, index=tickers, dtype=int)

        val_df = self.df[self.df["Date"] == current_date].copy()
        if val_df.empty:
            return pd.Series(0, index=tickers, dtype=int)

        X_val = val_df[select_feature_columns(val_df)]
        y_pred = model.predict(X_val)

        val_df["signal"] = y_pred
        signals = (
            val_df.set_index("Ticker")["signal"]
            .reindex(tickers)
            .fillna(0)
            .astype(int)
        )
        return signals


# class LGBMClassifierStrategy(BaseStrategy):
#     def __init__(self, df: pd.DataFrame):
#         super().__init__(df)

#     def model(self, df: pd.DataFrame):
#         """
#         Обучает prime_model на истории (до последней даты),
#         """
#         last_date = df["Date"].max()
#         train_df = df[df["Date"] < last_date]
#         val_df   = df[df["Date"] == last_date]
        
#         train_df.dropna(subset=[select_target_columns()], inplace=True)
#         train_df.reset_index(drop=True, inplace=True)
#         print(f"\nРазмер датасета для обучения: {len(train_df)} строк.")


#         if train_df.empty or val_df.empty:
#             raise BaseStrategyEmptyDataError

#         # 1️⃣ Prime model
#         X = train_df[select_feature_columns(train_df)]
#         y = train_df[select_target_columns()]
#         model = train_model_LGBMClassifier(X, y)
#         return model


#     def generate_signals(self, current_date: pd.Timestamp, tickers: list) -> pd.Series:
#         # 2️⃣ Out-of-sample прогноз
#         # X_val = val_df[select_feature_columns()]
#         # y_pred = model.predict(X_val)

#         try:
#             model = self.model(self.df[self.df["Date"] <= current_date])
#         except BaseStrategyEmptyDataError:
#             return pd.Series(0, index=tickers, dtype=int)

#         last_date = self.df["Date"].max()
#         val_df = self.df[self.df["Date"] == last_date]
#         X_val = val_df[select_feature_columns(val_df)]

#         y_pred = model.predict(X_val)
#         # связываем предсказания с тикерами
#         val_df["signal"] = y_pred

#         # создаём pd.Series с индексом тикеров
#         signals = (
#             val_df.set_index("Ticker")["signal"]
#             .reindex(tickers)       # чтобы сохранить порядок tickers
#             .fillna(0)              # если по какому-то тикеру нет данных — 0
#             .astype(int)
#         )
#         return signals


# # ------------------------------------------------------------------
# # ------------------------------------------------------------------
# class MetaModelStrategy(BaseStrategy):
#     """
#     Реализует идею мета-моделирования:
#     - обучает основную модель (prime_model) пошагово (walk-forward)
#     - накапливает предсказания и фактические результаты
#     - обучает мета-модель, чтобы фильтровать слабые сигналы
#     - возвращает сигналы по тикерам (−1, 0, 1)
#     """

#     META_FEATURES = ['pred_proba_buy', 'pred_proba_sell', 'proba_diff', 'atr_14', 'rsi_14']

#     def __init__(self, df: pd.DataFrame, 
#                  meta_confidence_threshold: float = 0.6):
#         """
#         Args:
#             df: DataFrame с колонками ['Date', 'Ticker', OHLCV, фичи, target]
#             feature_cols: список фич для основной модели
#             target_col: имя целевой колонки
#             meta_confidence_threshold: порог вероятности для мета-модели
#         """
#         super().__init__(df)
#         self.primary_model_builder = build_model
#         self.meta_model_builder = build_meta_model
#         self.feature_cols = select_feature_columns(df)
#         self.target_col: str = select_target_columns()
#         self.meta_confidence_threshold = meta_confidence_threshold

#         # Буфер для накопления данных мета-модели
#         self.meta_buffer = []

#         # Контейнер для итоговой обученной мета-модели
#         self.meta_model = None

#     # ------------------------------------------------------------------
#     # 1️⃣ Walk-forward сбор данных для мета-модели
#     # ------------------------------------------------------------------
#     def collect_meta_training_data(self, rolling_dates: list[pd.Timestamp]):
#         """
#         Пошагово обучает prime_model и собирает реальные предсказания для meta-модели.
#         """
#         print(f"Начало накопления мета-примеров: {len(rolling_dates)} шагов...")

#         for current_date in rolling_dates:
#             train_df = self.df[self.df["Date"] < current_date]
#             val_df = self.df[self.df["Date"] == current_date]

#             if len(train_df) < 100 or val_df.empty:
#                 continue

#             # --- Primary model ---
#             prime_model = self.primary_model_builder()
#             X_train = train_df[self.feature_cols]
#             y_train = train_df[self.target_col]

#             X_val = val_df[self.feature_cols]
#             y_val = val_df[self.target_col]

#             prime_model.fit(X_train, y_train)

#             # --- Предсказания ---
#             probas = prime_model.predict_proba(X_val)
#             preds = prime_model.predict(X_val)

#             val_df = val_df.copy()
#             val_df['pred_proba_sell'] = probas[:, 0]
#             if probas.shape[1] == 3:
#                 val_df['pred_proba_hold'] = probas[:, 1]
#                 val_df['pred_proba_buy'] = probas[:, 2]
#             else:
#                 # fallback: бинарная модель
#                 val_df['pred_proba_buy'] = probas[:, 1]
#                 val_df['pred_proba_sell'] = probas[:, 0]
#                 val_df['pred_proba_hold'] = 0.0
#             val_df['primary_pred'] = preds
#             val_df['proba_diff'] = val_df['pred_proba_buy'] - val_df['pred_proba_sell']
#             val_df['y_meta'] = (val_df['primary_pred'] == val_df[self.target_col]).astype(int)

#             self.meta_buffer.append(val_df)

#         # Собираем накопленные данные
#         meta_df = pd.concat(self.meta_buffer, ignore_index=True)
#         print(f"✅ Мета-данных собрано: {len(meta_df)} строк")
#         return meta_df

#     # ------------------------------------------------------------------
#     # 2️⃣ Обучение мета-модели
#     # ------------------------------------------------------------------
#     def train_meta_model(self, meta_df: pd.DataFrame):
#         """
#         Обучает мета-модель на накопленных данных.
#         """
#         X_meta = meta_df[self.META_FEATURES]
#         y_meta = meta_df['y_meta']

#         meta_model = self.meta_model_builder()
#         meta_model.fit(X_meta, y_meta)
#         self.meta_model = meta_model
#         print(f"✅ Мета-модель обучена на {len(X_meta)} примерах")

#     # ------------------------------------------------------------------
#     # 3️⃣ Генерация сигналов
#     # ------------------------------------------------------------------
#     def generate_signals(self, current_date: pd.Timestamp, tickers: list[str]) -> pd.Series:
#         """
#         Генерирует сигналы (−1, 0, 1) для заданной даты и тикеров.
#         """
#         assert self.meta_model is not None, "Мета-модель не обучена!"

#         # Исторические данные до текущей даты
#         df_now = self.df[self.df["Date"] == current_date]
#         if df_now.empty:
#             return pd.Series(0, index=tickers, dtype=int)

#         prime_model = self.primary_model_builder()
#         X_train = self.df[self.df["Date"] < current_date][self.feature_cols]
#         y_train = self.df[self.df["Date"] < current_date][self.target_col]
#         prime_model.fit(X_train, y_train)

#         X_test = df_now[self.feature_cols]
#         probas = prime_model.predict_proba(X_test)
#         preds = prime_model.predict(X_test)

#         df_now = df_now.copy()
#         df_now['pred_proba_sell'] = probas[:, 0]
#         if probas.shape[1] == 3:
#             df_now['pred_proba_hold'] = probas[:, 1]
#             df_now['pred_proba_buy'] = probas[:, 2]
#         else:
#             df_now['pred_proba_buy'] = probas[:, 1]
#             df_now['pred_proba_sell'] = probas[:, 0]
#             df_now['pred_proba_hold'] = 0.0
#         df_now['primary_pred'] = preds
#         df_now['proba_diff'] = df_now['pred_proba_buy'] - df_now['pred_proba_sell']

#         X_meta = df_now[self.META_FEATURES]
#         meta_proba = self.meta_model.predict_proba(X_meta)[:, 1]

#         # применяем фильтр уверенности
#         approved_mask = meta_proba > self.meta_confidence_threshold
#         df_now["signal"] = 0
#         df_now.loc[approved_mask, "signal"] = np.sign(df_now.loc[approved_mask, "primary_pred"])

#         # финальные сигналы по тикерам
#         signals = df_now.set_index("Ticker")["signal"].reindex(tickers).fillna(0).astype(int)
#         return signals
