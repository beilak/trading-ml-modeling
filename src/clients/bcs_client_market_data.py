from __future__ import print_function

import asyncio
import json
import typing as tp

import websockets
from websockets.client import connect

from src.clients.bcs_client import BCSClient


class SubscribeType(int):
    SUBSCRIBE = 0
    UNSUBSCRIBE = 1


class DataType(int):
    QUOTES = 3


class BCSMarketData(BCSClient):
    ws_url: tp.Final[str] = (
        "wss://ws.broker.ru/trade-api-market-data-connector/api/v1/market-data/ws"
    )

    async def _get_token(self) -> str:
        # используем существующий метод клиента для получения токена
        return await self._auth.get_access_tocken(writer=True)

    async def subscribe(
        self, instruments: list[dict[str, str]], callback: tp.Callable[[dict], None]
    ):
        """
        Подписка на котировки инструментов.
        instruments: [{"ticker": "SBER", "classCode": "TQBR"}, ...]
        callback: функция, которая будет получать сообщения от WS
        """
        token = await self._get_token()
        headers = {"Authorization": f"Bearer {token}"}

        async with connect(self.ws_url, extra_headers=headers) as ws:
            # формируем сообщение подписки
            message = {
                "subscribeType": SubscribeType.SUBSCRIBE,
                "dataType": DataType.QUOTES,
                "instruments": instruments,
            }
            await ws.send(json.dumps(message))

            # получаем сообщения в цикле
            async for msg in ws:
                data = json.loads(msg)
                callback(data)

    async def get_last_prices(
        self, instruments: list[dict[str, str]]
    ) -> dict[str, float]:
        token = await self._get_token()
        headers = {"Authorization": f"Bearer {token}"}
        prices: dict[str, float] = {}

        async with connect(self.ws_url, extra_headers=headers) as ws:
            message = {
                "subscribeType": SubscribeType.SUBSCRIBE,
                "dataType": DataType.QUOTES,
                "instruments": instruments,
            }
            await ws.send(json.dumps(message))

            try:
                async for msg in ws:
                    data = json.loads(msg)

                    # Предположим, что структура data примерно такая:
                    # {"instrument": "AAPL", "price": 174.3}
                    instrument = data.get("ticker")
                    last_price = data.get("last")
                    if instrument and last_price is not None:
                        prices[instrument] = last_price

                    # Можно добавить условие закрытия, например, после получения всех инструментов
                    if len(prices) == len(instruments):
                        break

            except asyncio.CancelledError:
                pass  # на случай отмены корутины

        return prices

    async def unsubscribe(
        self,
        instruments: list[dict[str, str]],
        callback: tp.Callable[[dict], None] | None = None,
    ):
        """
        Отписка от котировок.
        """
        token = await self._get_token()
        headers = {"Authorization": f"Bearer {token}"}

        async with websockets.connect(self.ws_url, extra_headers=headers) as ws:
            message = {
                "subscribeType": SubscribeType.UNSUBSCRIBE.value,
                "dataType": DataType.QUOTES.value,
                "instruments": instruments,
            }
            await ws.send(json.dumps(message))

            if callback:
                async for msg in ws:
                    data = json.loads(msg)
                    callback(data)
