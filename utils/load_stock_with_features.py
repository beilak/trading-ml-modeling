import pandas as pd
import json


def load_stock_with_features():
    # Загрузка основного датафрейма
    df = pd.read_csv('../data/moex_with_features.csv', parse_dates=['Date'])

    # Загрузка списка исключаемых тикеров

    with open('..config/stock_exclude_list.json', 'r') as f:
        exclude_list = json.load(f)

    df = df[~df['Ticker'].isin(exclude_list)]
    
    return df
