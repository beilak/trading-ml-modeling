import pandas as pd
import datetime
from abc import ABC, abstractmethod



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


class BaseStrategyEmptyDataError(Exception):
    ...
