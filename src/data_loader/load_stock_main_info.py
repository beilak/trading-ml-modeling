import os
from posix import PRIO_DARWIN_BG
import pandas as pd
from src.clients.bcs_client import BCSReferenceInfo, BCSAuth, BCSConfig, InstrumentsType

import asyncio


async def load_stock_tickers(auth):
    """
    Получает список всех доступных акций через BCS API
    и возвращает только поле ticker.
    """

    bcs = BCSReferenceInfo(auth=auth)

    # Получаем большой список инструментов
    instruments = await bcs.get_instruments(instruments_type=InstrumentsType.stock)

    # Извлекаем только поле ticker
    tickers = [
        item.get("ticker")
        for item in instruments
        if isinstance(item, dict) and item.get("ticker") and item.get('boards') == [{'classCode': 'TQBR', 'exchange': 'MOEX'}]
    ]

    # Делаем уникальный отсортированный список
    tickers = sorted(set(tickers))

    return tickers



async def update_tickers_info(
    auth,
    tickers: list[str],
    filepath: str = "data/stock/tickers_info.csv"
):
    """
    Получает информацию по списку тикеров, оставляет только нужные поля,
    обновляет CSV и возвращает актуальные данные.
    """

    # --- 0. Только нужные поля ---
    needed_fields = [
        "ticker",
        "shortName",
        "displayName",
        "type",
        "issuerName",
        "tradingCurrency",
        "subType",
        "businessSector",
        "businessSectorId",

    ]

    bcs = BCSReferenceInfo(auth=auth)
    raw_list = await bcs.get_tickers_info(tickers=tickers)

    # --- 3. Оставляем только нужные поля ---
    new_list = []
    for item in raw_list:
        if item.get('boards') == [{'classCode': 'TQBR', 'exchange': 'MOEX'}]:
            filtered = {field: item.get(field) for field in needed_fields}
            new_list.append(filtered)

    df_new = pd.DataFrame(new_list)

    df_new.to_csv(filepath, index=False)

    return df_new.to_dict(orient="records")




async def main():
    config = BCSConfig()
    auth = BCSAuth(token=config.BCS_API_TOKEN)
    tickers = await load_stock_tickers(auth)
    print(tickers)
    await update_tickers_info(auth, tickers)



asyncio.run(main())
