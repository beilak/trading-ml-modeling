import asyncio
import calendar
import os
from datetime import date, datetime, timezone
from typing import List, Optional, Tuple

import httpx
import pandas as pd
from tenacity import (
    retry,
    retry_if_exception_type,
    retry_if_result,
    stop_after_attempt,
    wait_exponential,
)


class BadStatus(Exception):
    pass


# def _bad_status(resp):
#     """helper to trigger retry when resp.status != 200"""
#     return resp.status != 200


@retry(
    retry=(retry_if_exception_type(Exception),),
    wait=wait_exponential(multiplier=1, min=1, max=10),  # 1s → 2s → 4s → …
    stop=stop_after_attempt(3),  # максимум 3 попыток
    reraise=True,  # пробросить ошибку наружу если не удалось
)
async def _fetch(client, params):
    resp = await client.get(
        "https://api.gdeltproject.org/api/v2/doc/doc?",
        params=params,
    )
    await asyncio.sleep(5)
    if resp.status_code != 200:
        raise BadStatus(f"Bad status code: {resp.status_code} {resp.text}")

    return resp


async def load_news(
    client, ticker_query: str, date_from: date, date_to: date
) -> list[dict]:
    params = {
        "query": f"({ticker_query}) (earnings OR profit OR dividend OR loan OR IPO)",
        "mode": "artlist",
        "format": "json",
        "maxrecords": 150,
        "sort": "DateDesc",
        "startdatetime": date_from.strftime("%Y%m%d%H%M%S"),
        "enddatetime": date_to.strftime("%Y%m%d%H%M%S"),
        "trans": "googtrans",
        "language": "English",
    }

    # resp = await client.get(
    #     "https://api.gdeltproject.org/api/v2/doc/doc?",
    #     params=params,
    #     )

    try:
        resp = await _fetch(client, params)  # с retry
    except Exception:
        return []  # после всех попыток — пустой список

    try:
        articles: list[dict] = []
        for i in resp.json()["articles"]:
            if i["language"] in ("English", "Russian"):
                articles.append(
                    {
                        "date": datetime.strptime(i["seendate"], "%Y%m%dT%H%M%SZ")
                        .replace(tzinfo=timezone.utc)
                        .date(),
                        "language": i["language"],
                        "sourcecountry": i["sourcecountry"],
                        "domain": i["domain"],
                        "url": i["url"],
                        "title": i["title"],
                    }
                )
        return articles
    except Exception:
        return []


# --- Утилиты для диапазонов по месяцам ---
def month_ranges(
    start_date: date, end_date: Optional[date] = None
) -> List[Tuple[datetime, datetime]]:
    """
    Возвращает список кортежей (start_dt, end_dt) по месяцам.
    start_date и end_date - объекты date. Если end_date is None -> сегодня.
    Каждая граница возвращается как datetime: 00:00:00 .. 23:59:59
    """
    if end_date is None:
        end_date = date.today()

    ranges: List[Tuple[datetime, datetime]] = []
    cur_year, cur_month = start_date.year, start_date.month
    cur = date(cur_year, cur_month, 1)

    while cur <= end_date:
        year, month = cur.year, cur.month
        last_day = calendar.monthrange(year, month)[1]
        month_start_dt = datetime(year, month, 1, 0, 0, 0)

        month_end_date = date(year, month, last_day)
        # ограничение правой границы общим end_date
        real_end_date = month_end_date if month_end_date <= end_date else end_date
        month_end_dt = datetime(
            real_end_date.year, real_end_date.month, real_end_date.day, 23, 59, 59
        )

        ranges.append((month_start_dt, month_end_dt))

        # перейти на первый день следующего месяца
        if month == 12:
            cur = date(year + 1, 1, 1)
        else:
            cur = date(year, month + 1, 1)

    return ranges


def three_month_ranges(
    start_date: date, end_date: date = None
) -> List[Tuple[datetime, datetime]]:
    """
    Возвращает список кортежей (start_dt, end_dt) по блокам из 3 месяцев.
    Каждая граница как datetime с временем 00:00:00 .. 23:59:59
    """
    if end_date is None:
        end_date = date.today()

    ranges: List[Tuple[datetime, datetime]] = []
    cur_year, cur_month = start_date.year, start_date.month
    cur = date(cur_year, cur_month, 1)

    while cur <= end_date:
        # Начало блока
        block_start_dt = datetime(cur.year, cur.month, 1, 0, 0, 0)

        # Считаем конец блока через 3 месяца
        block_end_month = cur.month + 2
        block_end_year = cur.year
        if block_end_month > 12:
            block_end_month -= 12
            block_end_year += 1

        last_day = calendar.monthrange(block_end_year, block_end_month)[1]
        block_end_date = date(block_end_year, block_end_month, last_day)
        # Ограничение правой границы end_date
        if block_end_date > end_date:
            block_end_date = end_date
        block_end_dt = datetime(
            block_end_date.year, block_end_date.month, block_end_date.day, 23, 59, 59
        )

        ranges.append((block_start_dt, block_end_dt))

        # Переходим к следующему блоку через 3 месяца
        next_month = cur.month + 3
        next_year = cur.year
        if next_month > 12:
            next_month -= 12
            next_year += 1
        cur = date(next_year, next_month, 1)

    return ranges


# --- Асинхронные обёртки и исполнитель ---
async def _call_load_news_with_sem(
    client,
    sem: asyncio.Semaphore,
    ticker_query: str,
    date_from: datetime,
    date_to: datetime,
):
    """
    Вспомогательная функция, выполняет load_news под семафором.
    Предполагает что load_news - async функция доступная в области видимости.
    """
    async with sem:
        return await load_news(client, ticker_query, date_from, date_to)


async def run_monthly_loads(
    ticker_query: str,
    start_date: date,
    end_date: Optional[date] = None,
    concurrency: int = 1,
) -> list:
    """
    Запускает load_news для каждого месячного диапазона (от start_date до end_date или сегодня).
    Возвращает список результатов (в том порядке, в котором задачи были созданы).
    Параметр concurrency ограничивает число параллельных вызовов.
    """
    ranges = three_month_ranges(start_date, end_date)
    sem = asyncio.Semaphore(concurrency)

    async with httpx.AsyncClient() as client:
        tasks = [
            asyncio.create_task(
                _call_load_news_with_sem(client, sem, ticker_query, r_start, r_end)
            )
            for (r_start, r_end) in ranges
        ]
        # Собираем результаты; если хочешь — можно обрабатывать по готовности через asyncio.as_completed
        results = await asyncio.gather(*tasks, return_exceptions=False)
    return results


def save_news(result_df: pd.DataFrame, path: str = "data/news/news.csv"):
    # Если файла нет — просто сохраняем
    if not os.path.exists(path):
        result_df.to_csv(path, index=False)
        return

    # Файл есть → подгружаем старый
    old_df = pd.read_csv(path)

    # Склеиваем
    combined = pd.concat([old_df, result_df], ignore_index=True)

    # Удаляем дубликаты
    # Критерии дублирования: url + date + ticker
    combined.drop_duplicates(subset=["url", "date", "ticker"], inplace=True)

    # Сохраняем
    combined.to_csv(path, index=False)


async def main():
    df = pd.read_csv("data/stock/tickers_info.csv")
    print(df.head())
    start = date(2021, 10, 1)
    end_date = date(2022, 12, 31)
    all_results = []
    for idx, row in enumerate(df.itertuples()):
        print(f"[{idx}] Processing {row.ticker}")
        query = (
            f"{row.ticker} OR {row.shortName} OR {row.displayName} OR {row.issuerName}"
        )
        print(query)
        results = await run_monthly_loads(
            query, start, end_date=end_date, concurrency=10
        )
        # Разворачиваем результаты (список списков → список)
        flat = [item for month in results for item in month]

        # Если что-то нашли — превращаем в DataFrame
        if flat:
            df_news = pd.DataFrame(flat)

            # Добавляем колонку тикера
            df_news.insert(0, "ticker", row.ticker)

            # Добавляем в общий список
            all_results.append(df_news)

        print(f"Finished processing [{row.ticker}]")
        print("-" * 100)

    # -------- Сбор всех результатов в один DataFrame --------
    if all_results:
        result_df = pd.concat(all_results, ignore_index=True)
    else:
        result_df = pd.DataFrame()  # если ничего нет

    print(result_df.head())
    save_news(result_df, "data/news/news.csv")


asyncio.run(main())
