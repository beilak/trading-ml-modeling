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


from strategies.general import BaseStrategy, BaseStrategyEmptyDataError


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
    
