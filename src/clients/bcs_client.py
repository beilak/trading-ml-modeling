from contextlib import asynccontextmanager
from enum import StrEnum, auto
import typing as tp
from async_lru import alru_cache
import httpx
from pydantic_settings import BaseSettings, SettingsConfigDict
from datetime import datetime, timezone


class BCSConfig(BaseSettings):
    BCS_API_TOKEN: str

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra='ignore')


class BCSAuth:
    auth_url: tp.Final[str] = "https://be.broker.ru/trade-api-keycloak/realms/tradeapi/protocol/openid-connect/token"
    client_id_reader: tp.Final[str] = "trade-api-read"

    def __init__(self, token: str):
        self._token: str = token
        
    @alru_cache(ttl=80000)
    async def get_access_tocken(self) -> str:
        client: httpx.AsyncClient
        async with httpx.AsyncClient() as client:
            auth_resp = await client.post(
                self.auth_url, 
                data={
                    "client_id": self.client_id_reader,                    
                    "refresh_token": self._token,
                    "grant_type": "refresh_token",
                }
            )

        return auth_resp.json()["access_token"]


class BCSClient:
    def __init__(self, auth: BCSAuth):
        self._auth = auth
    
    @asynccontextmanager
    async def _client(self) -> tp.AsyncGenerator[httpx.AsyncClient, None]:
        token: str  = await self._auth.get_access_tocken()
        async with httpx.AsyncClient(
            headers={"Authorization": f"Bearer {token}"},
        ) as client:
            yield client

class BCSLimits(BCSClient):
    limits_url: tp.Final[str] = "https://be.broker.ru/trade-api-bff-limit/api/v1/limits"    

    async def get_portfolio_limits(self) -> dict:
        async with self._client() as client:
            limits = await client.get(url=self.limits_url)

        return limits.json()


class BCSPortfolio(BCSClient):
    portfolio_url: tp.Final[str] = "https://be.broker.ru/trade-api-bff-portfolio/api/v1/portfolio"

    async def get_profile_state(self) -> dict:
        async with self._client() as client:
            portfolio = await client.get(url=self.portfolio_url)

        return portfolio.json()


class InstrumentsType(StrEnum):
    stock = "STOCK"


class BCSReferenceInfo(BCSClient):
    ticker_info_url: tp.Final[str] = "https://be.broker.ru/trade-api-information-service/api/v1/instruments/by-tickers"
    inst_types_url: tp.Final[str] = "https://be.broker.ru/trade-api-information-service/api/v1/instruments/by-type"

    async def get_instruments(self, instruments_type: InstrumentsType = InstrumentsType.stock) -> dict:
        async with self._client() as client:
            instruments = await client.get(url=self.inst_types_url, params={"type": instruments_type})

        return instruments.json()
    
    async def get_tickers_info(self, tickers: list[str]) -> list[dict]:
        async with self._client() as client:
            tickers = await client.post(url=self.ticker_info_url, json={"tickers": tickers})
            
        return tickers.json()


class TimeFrame(StrEnum):
    M1 = "M1"
    M5 = "M5"
    M15 = "M15"
    M30 = "M30"
    H1 = "H1"
    H4 = "H4"
    D = "D"
    W = "W"
    MN = "MN"

class BCSCandles(BCSClient):
    candles: tp.Final[str] = "https://be.broker.ru/trade-api-market-data-connector/api/v1/candles-chart"

    async def get_candles(self, ticker: str, class_code: str = "TQBR", start_date: str | None = None, end_date: str | None = None, time_frame: TimeFrame = TimeFrame.D):
        date_from = start_date if start_date else "1900-01-01T00:00:00Z"
        date_to = end_date if end_date else datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
        async with self._client() as client:
            candles = await client.get(
                url=self.candles, 
                params={
                    "classCode": class_code,
                    "ticker": ticker,
                    "startDate": date_from,
                    "endDate": date_to,
                    "timeFrame": time_frame,
                },
            )

        return candles.json()