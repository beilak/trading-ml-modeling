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
from strategies.general import BaseStrategy


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