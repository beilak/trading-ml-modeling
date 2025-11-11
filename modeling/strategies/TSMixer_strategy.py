# ==============================================================
# ⚡️ Стратегия на базе TSMixer
# ==============================================================
import os
import json
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod



from meta_model_trainer import build_meta_model, train_meta_model, META_CONFIDENCE_THRESHOLD
from strategies.general import BaseStrategy, BaseStrategyEmptyDataError

from TSMixer_Classifier_modeling import (
    train_model_TSMixer,
    select_feature_columns,
    select_target_columns,
    TimeSeriesDataset,
    DataLoader,
    DEVICE,
)
import torch

class TSMixerStrategy(BaseStrategy):
    def __init__(self, df: pd.DataFrame, lookback: int = 32, epochs: int = 10):
        super().__init__(df)
        self.lookback = lookback
        self.epochs = epochs
        # self.last_train_date = None
        # self.model_cache = None

    def model(self, df: pd.DataFrame):
        """
        Обучает модель TSMixer или возвращает кэшированную.
        """
        last_date = df["Date"].max()

        train_df = df[df["Date"] < last_date].copy()
        val_df   = df[df["Date"] == last_date].copy()

        if train_df.empty or val_df.empty:
            raise BaseStrategyEmptyDataError

        X = train_df[select_feature_columns(train_df)]
        y = train_df[select_target_columns()]

        print(f"\n[TSMixer] Обучение модели на {len(train_df)} строк (до {last_date.date()})")
        model = train_model_TSMixer(X, y, epochs=self.epochs, lookback=self.lookback)
        model.eval()

        return model


    def generate_signals(self, current_date: pd.Timestamp, tickers: list) -> pd.Series:
        """
        Генерация сигналов на основе TSMixer.
        Если данных недостаточно или произошла ошибка — возвращает нули.
        """
        try:
            model = self.model(self.df[self.df["Date"] <= current_date])
        except BaseStrategyEmptyDataError:
            return pd.Series(0, index=tickers, dtype=int)

        # --- Берем последние lookback дней данных ---
        hist_df = self.df[self.df["Date"] <= current_date].copy()
        if hist_df["Date"].nunique() < self.lookback:
            print(f"[TSMixer] Недостаточно данных ({hist_df['Date'].nunique()}) для lookback={self.lookback}")
            return pd.Series(0, index=tickers, dtype=int)

        # --- Отбираем последние lookback дней ---
        last_dates = sorted(hist_df["Date"].unique())[-self.lookback:]
        val_df = hist_df[hist_df["Date"].isin(last_dates)].copy()

        if val_df.empty:
            return pd.Series(0, index=tickers, dtype=int)

        # --- Формируем признаки ---
        X_val = val_df[select_feature_columns(val_df)]

        # --- Если данных по какому-то тикеру нет, ставим 0 ---
        try:
            dataset = TimeSeriesDataset(X_val, np.zeros(len(X_val)), self.lookback)
            loader = DataLoader(dataset, batch_size=32, shuffle=False)

            preds = []
            with torch.no_grad():
                for xb, _ in loader:
                    xb = xb.to(DEVICE)
                    outputs = model(past_values=xb)
                    logits = outputs.prediction_outputs.squeeze(1)
                    preds.append(torch.argmax(logits, dim=1).cpu().numpy())

            y_pred = np.concatenate(preds) if preds else np.zeros(len(X_val))
            y_pred = np.where(y_pred > 0.33, 1, np.where(y_pred < -0.33, -1, 0))

            # --- Присваиваем сигналы по тикерам ---
            latest_date_df = val_df[val_df["Date"] == current_date]
            latest_tickers = latest_date_df["Ticker"].tolist()

            signals = pd.Series(0, index=tickers, dtype=int)
            for t in latest_tickers:
                signals.loc[t] = y_pred[-1]  # последний прогноз — для текущей даты

            return signals

        except Exception as e:
            print(f"[TSMixer] Ошибка при генерации сигналов ({current_date.date()}): {e}")
            return pd.Series(0, index=tickers, dtype=int)
