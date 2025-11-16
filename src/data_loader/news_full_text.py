import asyncio
import os

import httpx
import pandas as pd
from newspaper import Article
from tenacity import retry, stop_after_attempt, wait_exponential
from tqdm import tqdm

INPUT_FILE = "data/news/news.csv"
OUTPUT_FILE = "data/news/news_full.csv"


# -----------------------------
# Утилиты для ключа (ticker|date|url)
# -----------------------------
def make_key_row(row):
    # Приводим к строкам, убираем NaN
    ticker = "" if pd.isna(row.get("ticker")) else str(row["ticker"])
    date = "" if pd.isna(row.get("date")) else str(row["date"])
    url = "" if pd.isna(row.get("url")) else str(row["url"])
    return f"{ticker}|{date}|{url}"


def add_key_column(df):
    # Создаём колонку 'key' для быстрого сравнения
    df = df.copy()
    df["key"] = df.apply(make_key_row, axis=1)
    return df


# -----------------------------
# Загрузка существующих обработанных данных (если есть)
# -----------------------------
def load_existing():
    if os.path.exists(OUTPUT_FILE):
        df = pd.read_csv(OUTPUT_FILE)
        if "key" not in df.columns:
            df = add_key_column(df)
        print(f"Loaded existing processed rows: {len(df)}")
        return df
    return pd.DataFrame(
        columns=[
            "ticker",
            "date",
            "language",
            "sourcecountry",
            "domain",
            "url",
            "title",
            "news_txt",
            "key",
        ]
    )


# -----------------------------
# HTTP fetch с ретраями
# -----------------------------
@retry(stop=stop_after_attempt(5), wait=wait_exponential(min=1, max=10))
async def fetch_html(url: str, client: httpx.AsyncClient) -> str:
    resp = await client.get(url, timeout=15)
    if resp.status_code != 200:
        raise Exception(f"HTTP {resp.status_code}")
    return resp.text


# -----------------------------
# Парсинг через newspaper3k (с HTML)
# -----------------------------
def parse_with_newspaper(url: str, html: str) -> str:
    article = Article(url)
    # передаём input_html — newspaper не будет заново скачивать
    article.download(input_html=html)
    article.parse()
    text = article.text or ""
    # обрезаем и убираем переносы строк, как в примере
    text = text[:1000].replace("\n", " ")
    return text


# -----------------------------
# Обработка одной строки (получаем Series)
# -----------------------------
async def process_row(
    row: pd.Series, client: httpx.AsyncClient, sem: asyncio.Semaphore
):
    # Если уже есть news_txt — пропускаем
    if pd.notna(row.get("news_txt")) and str(row.get("news_txt")).strip() != "":
        return row

    url = row.get("url")
    if not url or pd.isna(url):
        row["news_txt"] = None
        return row

    async with sem:
        try:
            html = await fetch_html(url, client)
            # newspaper3k — блокирующая, выносим в thread
            text = await asyncio.to_thread(parse_with_newspaper, url, html)
            row["news_txt"] = text
        except Exception as e:
            # можно логировать e
            print(f"Error fetching/parsing {url} — {e}")
            row["news_txt"] = None

    return row


# -----------------------------
# Основная логика
# -----------------------------
async def main(concurrency: int = 10, batch_size: int = 50):
    # 1) загружаем входной CSV и existing
    df_input = pd.read_csv(INPUT_FILE)
    df_input = add_key_column(df_input)

    df_existing = load_existing()
    if len(df_existing) > 0:
        # гарантируем наличие key
        df_existing = add_key_column(df_existing)
        # создадим мапинг key -> news_txt
        existing_map = dict(zip(df_existing["key"], df_existing["news_txt"]))
    else:
        existing_map = {}

    # 2) создаём итоговый df, берем существующие news_txt, если есть
    df = df_input.copy()
    # если в исходном файле есть колонка news_txt — учитываем её, но приоритет отдадим существующему news_full
    df["news_txt"] = df.get("news_txt", pd.NA)
    # заполняем из existing_map
    df["news_txt"] = df["key"].map(existing_map).fillna(df["news_txt"])

    # 3) какие строки нужно загрузить
    todos = df[df["news_txt"].isna()].copy()
    n_todos = len(todos)
    print(f"Всего строк: {len(df)}. Нужно загрузить: {n_todos}")

    if n_todos == 0:
        print("Нечего скачивать — всё уже обработано.")
        # сохраняем просто на всякий случай
        df.to_csv(OUTPUT_FILE, index=False)
        return

    sem = asyncio.Semaphore(concurrency)

    async with httpx.AsyncClient() as client:
        # собираем задачи
        tasks = []
        # сохраним индекс ключа, чтобы потом корректно обновить
        for idx, row in todos.iterrows():
            tasks.append(asyncio.create_task(process_row(row, client, sem)))

        pbar = tqdm(total=len(tasks), desc="fetching", unit="it")

        results_buf = []  # буфер обработанных Series
        # по мере выполнения задач
        for coro in asyncio.as_completed(tasks):
            row_processed = await coro  # это pandas Series
            results_buf.append(row_processed)
            pbar.update(1)

            # периодически сохраняем batch
            if len(results_buf) >= batch_size:
                # превращаем буфер в DataFrame и обновляем df по key
                res_df = pd.DataFrame(results_buf)
                if "key" not in res_df.columns:
                    res_df = add_key_column(res_df)
                # обновление: ставим индекс по key и update
                df.set_index("key", inplace=True)
                res_df.set_index("key", inplace=True)
                df.update(res_df[["news_txt"]])
                df.reset_index(inplace=True)
                # сохраняем
                df.to_csv(OUTPUT_FILE, index=False)
                results_buf = []

        pbar.close()

        # финальное обновление оставшихся результатов
        if results_buf:
            res_df = pd.DataFrame(results_buf)
            if "key" not in res_df.columns:
                res_df = add_key_column(res_df)
            df.set_index("key", inplace=True)
            res_df.set_index("key", inplace=True)
            df.update(res_df[["news_txt"]])
            df.reset_index(inplace=True)

    # финальная запись
    df.to_csv(OUTPUT_FILE, index=False)
    print("Готово. Результат сохранён в", OUTPUT_FILE)


if __name__ == "__main__":
    asyncio.run(main(concurrency=10, batch_size=50))
