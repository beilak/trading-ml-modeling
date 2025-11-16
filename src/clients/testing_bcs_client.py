import asyncio
import uuid

from src.clients.bcs_client import (
    BCSAuth,
    BCSCandles,
    BCSConfig,
    BCSLimits,
    BCSOrder,
    BCSPortfolio,
    BCSReferenceInfo,
    ClassCode,
    InstrumentsType,
    OrderSide,
    OrderType,
)

config = BCSConfig()


async def read():
    auth = BCSAuth(token=config.BCS_API_TOKEN)

    print(await BCSLimits(auth=auth).get_portfolio_limits())
    print("*" * 10)
    print(await BCSPortfolio(auth=auth).get_profile_state())
    print("*" * 10)

    print("--" * 10)
    print(
        await BCSReferenceInfo(auth=auth).get_instruments(
            instruments_type=InstrumentsType.stock
        )
    )
    print("--" * 10)

    print("--" * 10)
    print(await BCSReferenceInfo(auth=auth).get_tickers_info(tickers=["SBER"]))
    print("--" * 10)

    print(await BCSCandles(auth=auth).get_candles(ticker="SBER", class_code="TQBR"))
    #
    #
    #


async def write():
    auth = BCSAuth(
        token=config.BCS_POST_API_TOKEN,
    )

    # Купить (Лимитный)
    resp = await BCSOrder(auth=auth).post_order(
        ticker="SBER",
        class_code=ClassCode.TQBR,
        client_order_id=uuid.uuid4(),
        side=OrderSide.BUY,
        order_type=OrderType.LIMIT,
        order_quantity=10,
        price=296.90,
    )
    print(resp)

    # Купить по Маркету
    # resp = await BCSOrder(auth=auth).post_order(
    #     ticker="SBER",
    #     class_code=ClassCode.TQBR,
    #     client_order_id=uuid.uuid4(),
    #     side=OrderSide.BUY,
    #     order_type=OrderType.MARKET,
    #     order_quantity=1,
    #     # price=295,
    # )
    # print(resp)

    # Продать (Лимитный)
    resp = await BCSOrder(auth=auth).post_order(
        ticker="SBER",
        class_code=ClassCode.TQBR,
        client_order_id=uuid.uuid4(),
        side=OrderSide.SELL,
        order_type=OrderType.LIMIT,
        order_quantity=10,
        price=297.5,
    )
    print(resp)


async def main():
    # await read()
    await write()


asyncio.run(main())
