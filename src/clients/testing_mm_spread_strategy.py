import asyncio
import uuid

from src.clients.bcs_client import (
    BCSAuth,
    BCSConfig,
    BCSOrder,
    ClassCode,
    OrderSide,
    OrderType,
)
from src.clients.bcs_client_market_data import BCSMarketData

config = BCSConfig()


auth = BCSAuth(
    token=config.BCS_POST_API_TOKEN,
)


async def market_maker_spread_strategy(
    ticker: str,
    last_price,
    spread_pct: float,
    commission_pct: float,
    quantity: int = 1,
) -> dict:
    """
    Маркет-мейкерская стратегия с двусторонним спредом и учетом комиссии от оборота.
    Выставляет лимитную заявку на покупку ниже рынка и на продажу выше рынка.

    :param ticker: Тикер бумаги
    :param spread_pct: Размер спреда от текущей цены (0.01 = 1%)
    :param commission_pct: Комиссия от оборота (0.0005 = 0.05%)
    :param quantity: Количество лотов
    """

    # 2. Рассчитываем цены покупки и продажи с учетом спреда
    buy_price = last_price * (1 - spread_pct)
    sell_price = last_price * (1 + spread_pct)

    # Округляем по шагу цены (например, 2 знака после запятой)
    buy_price = round(buy_price, 2)
    sell_price = round(sell_price, 2)

    # 3. Рассчитываем комиссию и чистый доход
    turnover = (buy_price * quantity) + (sell_price * quantity)
    commission_amount = turnover * commission_pct

    gross_profit = (sell_price * quantity) - (buy_price * quantity)
    net_profit = gross_profit - commission_amount

    print(f"[MM] {ticker}: Last={last_price}, Buy={buy_price}, Sell={sell_price}")
    print(
        f"Gross profit: {gross_profit}, Commission: {commission_amount}, Net profit: {net_profit}"
    )

    # 4. Выставляем лимитные заявки
    # buy_order_id = _place_order(
    #     ticker=ticker, side="buy", price=buy_price, quantity=quantity
    # )

    # Купить (Лимитный)
    buy_resp = await BCSOrder(auth=auth).post_order(
        ticker=ticker,
        class_code=ClassCode.TQBR,
        client_order_id=uuid.uuid4(),
        side=OrderSide.BUY,
        order_type=OrderType.LIMIT,
        order_quantity=quantity,
        price=buy_price,
    )

    # Продать (Лимитный)
    sell_resp = await BCSOrder(auth=auth).post_order(
        ticker=ticker,
        class_code=ClassCode.TQBR,
        client_order_id=uuid.uuid4(),
        side=OrderSide.SELL,
        order_type=OrderType.LIMIT,
        order_quantity=quantity,
        price=sell_price,
    )

    return {
        "buy_order_id": buy_resp,
        "sell_order_id": sell_resp,
        "buy_price": buy_price,
        "sell_price": sell_price,
        "buy_total": buy_price * quantity,
        "sell_total": sell_price * quantity,
        "last_price": last_price,
        "gross_profit": round(gross_profit, 3),
        "commission": round(commission_amount, 3),
        "net_profit": round(net_profit, 3),
    }


async def main():
    market = BCSMarketData(auth=auth)
    # .subscribe(
    #     instruments=[{"ticker": "SBER", "classCode": ClassCode.TQBR.value}],
    #     callback=callback,
    # )

    tickers = [
        {"ticker": "SBER", "classCode": ClassCode.TQBR.value},
        # {"ticker": "GAZP", "classCode": ClassCode.TQBR.value},
        # {"ticker": "TATNP", "classCode": ClassCode.TQBR.value},
    ]

    last_price = await market.get_last_prices(tickers)

    for ticker in tickers:
        # :param spread_pct: Размер спреда от текущей цены (0.01 = 1%)
        # :param commission_pct: Комиссия от оборота (0.0005 = 0.05%)
        result = await market_maker_spread_strategy(
            ticker=ticker["ticker"],
            last_price=last_price[ticker["ticker"]],
            spread_pct=0.001,
            commission_pct=0.0004,
            quantity=1,
        )
        for k, v in result.items():
            print(f"{k}: {v}")


asyncio.run(main())
