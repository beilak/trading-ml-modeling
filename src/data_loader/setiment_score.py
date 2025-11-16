import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("yiyanghkust/finbert-tone")
model_en = AutoModelForSequenceClassification.from_pretrained(
    "yiyanghkust/finbert-tone"
)
tokenizer_ru = AutoTokenizer.from_pretrained("blanchefort/rubert-base-cased-sentiment")
model_ru = AutoModelForSequenceClassification.from_pretrained(
    "blanchefort/rubert-base-cased-sentiment"
)

model_en.eval()
model_ru.eval()


def get_sentiment_en(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True)
    outputs = model_en(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1).detach().numpy()[0]
    return dict(zip(["positive", "negative", "neutral"], probs))


def get_sentiment_ru(text):
    """
    Возвращает словарь {'positive','negative','neutral'} для русского текста
    """
    inputs = tokenizer_ru(text, return_tensors="pt", truncation=True, max_length=128)
    with torch.no_grad():
        outputs = model_ru(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1).numpy()[0]

    # DeepPavlov RuBERT возвращает порядок: [negative, neutral, positive]
    return {
        "negative": float(probs[0]),
        "neutral": float(probs[1]),
        "positive": float(probs[2]),
    }


def sentiment_to_float(sentiment_dict, use_neutral_weight=True):
    """
    Преобразует словарь FinBERT {'positive','negative','neutral'}
    в float от -1 до +1.

    use_neutral_weight=True учитывает нейтральность.
    """
    pos = sentiment_dict.get("positive", 0.0)
    neg = sentiment_dict.get("negative", 0.0)
    neu = sentiment_dict.get("neutral", 0.0)

    score = pos - neg

    if use_neutral_weight:
        score *= 1 - neu

    # Ограничиваем [-1, 1]
    score = max(-1.0, min(1.0, score))
    return score


def aggregate_sentiments(sentiment_list, use_neutral_weight=True):
    """
    sentiment_list — список словарей FinBERT: [{'positive':..,'negative':..,'neutral':..}, ...]
    Возвращает float от -1 до +1
    """
    if not sentiment_list:
        return 0.0  # пустой список → нейтрально

    scores = [sentiment_to_float(s, use_neutral_weight) for s in sentiment_list]
    # усредняем
    return float(sum(scores) / len(scores))


# =============== ЗАДАЧА 1 ===============

# Загружаем файл
df = pd.read_csv("data/news/news_full.csv")


# Готовим колонку с полным текстом
def combine_text(row):
    if pd.isna(row["news_txt"]):
        return str(row["title"])
    return str(row["title"]) + " " + str(row["news_txt"])


df["full_text"] = df.apply(combine_text, axis=1)

# Вычисляем скоринг построчно
# sentiment_scores = []
sentiment_dicts = []  # сохраняем сами dict-и для дальнейшей агрегации

print("Calculating sentiment per news item...")
# for text, lang in tqdm(zip(df["full_text"], df["language"]), total=len(df)):
for row in tqdm(df.itertuples(), total=len(df)):
    # for row in df.itertuples():
    text = row.full_text
    lang = row.language
    if lang.lower().startswith("eng"):  # английский
        s = get_sentiment_en(text)
    elif lang.lower().startswith("ru"):
        s = get_sentiment_ru(text)
    else:
        continue

    sentiment_dicts.append(s)
    # sentiment_scores.append(sentiment_to_float(s))
df["sentiment_dict"] = sentiment_dicts
# df["sentiment_score"] = sentiment_scores


# =============== ЗАДАЧА 2 ===============

print("Aggregating by Ticker + Date...")


def aggregate_group(group):
    lst = group["sentiment_dict"].tolist()
    score = aggregate_sentiments(lst)

    return pd.Series({"score": score})


result = df.groupby(["ticker", "date"]).apply(aggregate_group).reset_index()
result["score"] = result["score"].round(4)

# =============== ЗАДАЧА 3 ===============

result.to_csv("data/news/news_score.csv", index=False)

print("Saved to data/news/news_score.csv")
print(result.head())
