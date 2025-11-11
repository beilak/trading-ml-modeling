import pandas as pd

from ensemble_modelling import select_feature_columns, select_target_columns, train_model
from strategies.general import BaseStrategy, BaseStrategyEmptyDataError


class EnsembleClassifierStrategy(BaseStrategy):
    def __init__(self, df: pd.DataFrame):
        super().__init__(df)

    def model(self, df: pd.DataFrame):
        """
        Обучает или возвращает кэшированную модель.
        """
        last_date = df["Date"].max()

        # 🔄 Переобучаем модель
        train_df = df[df["Date"] < last_date].copy()
        val_df   = df[df["Date"] == last_date].copy()

        train_df.dropna(subset=[select_target_columns()], inplace=True)
        train_df.reset_index(drop=True, inplace=True)
        print(f"\nОбучаем EnsembleClassifier на {len(train_df)} строк (до {last_date.date()})")

        if train_df.empty or val_df.empty:
            raise BaseStrategyEmptyDataError

        X = train_df[select_feature_columns(train_df)]
        y = train_df[select_target_columns()]
        model = train_model(X, y)

        
        return model

    def generate_signals(self, current_date: pd.Timestamp, tickers: list) -> pd.Series:
        """
        Генерация торговых сигналов на основе обученной модели.
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
