from  src.clients.bcs_client import BCSAuth, BCSConfig, BCSLimits, BCSPortfolio, BCSReferenceInfo, InstrumentsType, BCSCandles
import asyncio


async def main():
    # auth = BCSAuth(token=config.BCS_API_TOKEN).get_access_tocken()
    auth = BCSAuth(token=config.BCS_API_TOKEN)

    # print(await BCSLimits(auth=auth).get_portfolio_limits())
    # print("*"*10)
    # print(await BCSPortfolio(auth=auth).get_profile_state())
    # print("*"*10)

    print("--"*10)
    print(await BCSReferenceInfo(auth=auth).get_instruments(instruments_type=InstrumentsType.stock))
    print("--"*10)

    # print("--"*10)
    # print(await BCSReferenceInfo(auth=auth).get_tickers_info(tickers=["SBER"]))
    # print("--"*10)

    # print( await BCSCandles(auth=auth).get_candles(ticker="SBER", class_code="TQBR"))

config = BCSConfig()
asyncio.run(main())