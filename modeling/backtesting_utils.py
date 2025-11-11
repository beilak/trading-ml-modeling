from strategies.general import BaseStrategy
import numpy as np
import pandas as pd
import vectorbt as vbt
import os
from joblib import Parallel, delayed
from scipy.stats import skew, kurtosis
import matplotlib.pyplot as plt

figsize=(16,8)

n_jobs = -1 # Max pralell use CPU


def compute_weights_for_date(dt, price_index, tickers, strategy: BaseStrategy):
    print(f"Compute for {dt}")
    if dt not in price_index:
        return None, None

    sig = strategy.generate_signals(dt, tickers)
    active = sig[sig == 1].index
    if len(active) == 0:
        return dt, None
    weights = {ticker: 1 / len(active) for ticker in active}
    return dt, weights


def _collect_target_weights(price, tickers, results):
    # --- Собираем обратно ---
    target_weights = pd.DataFrame(0, index=price.index, columns=tickers)
    for dt, weights in results:
        if dt is None or weights is None:
            continue
        for t, w in weights.items():
            target_weights.loc[dt, t] = w
    return target_weights


def run_paralel(rebalance_dates, price, tickers, strategy):
    # --- Распараллеленный расчёт ---
    results = Parallel(n_jobs=n_jobs, backend='loky')(
    delayed(compute_weights_for_date)(dt, price.index, tickers, strategy)
    for dt in rebalance_dates[100:]
    )

    return _collect_target_weights(price, tickers, results)


def run_sequential(rebalance_dates, price, tickers, strategy):
    """
    Последовательный (непараллельный) расчёт target_weights.
    """
    results = []
    for dt in rebalance_dates[100:]:
        res = compute_weights_for_date(dt, price.index, tickers, strategy)
        results.append(res)

    return _collect_target_weights(price, tickers, results)


def run_ml_weekly_strategy(df,
                            strategy: BaseStrategy,
                            init_cash=1_000_000.00,
                            fees=0.005,
                            freq='1D',
                            do_paralel=True,
                            # size_type='targetpercent',
                            # cash_sharing=True,
                            # call_seq='auto'
                        ):
    """Еженедельная стратегия на основе сигналов ML"""
    # --- Подготовка данных ---
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values(['Date', 'Ticker'])
    price = df.pivot(index='Date', columns='Ticker', values='Close').sort_index()
    tickers = price.columns.tolist()

    # --- Ребалансировка каждую неделю ---
    rebalance_dates = price.index.to_series().resample('W-MON').last()
    # target_weights = pd.DataFrame(0, index=price.index, columns=tickers)


    # # --- Распараллеленный расчёт ---
    # results = Parallel(n_jobs=n_jobs, backend='loky')(
    # delayed(compute_weights_for_date)(dt, price.index, tickers, strategy)
    # for dt in rebalance_dates[100:]
    # )
    # # --- Собираем обратно ---
    # target_weights = pd.DataFrame(0, index=price.index, columns=tickers)
    # for dt, weights in results:
    #     if dt is None or weights is None:
    #         continue
    #     for t, w in weights.items():
    #         target_weights.loc[dt, t] = w

    if do_paralel:
        target_weights = run_paralel(rebalance_dates, price, tickers, strategy)
    else:
        target_weights = run_sequential(rebalance_dates, price, tickers, strategy)


    # --- Заполнение весов через forward fill между ребалансировками ---
    target_weights = target_weights.replace(0, np.nan).ffill().fillna(0)

    # --- Создание портфеля ---
    pf = vbt.Portfolio.from_orders(
        close=price,
        size=target_weights,
        size_type='targetpercent',
        init_cash=init_cash,
        fees=fees,
        freq=freq,
        cash_sharing=True,
        call_seq='auto'
    )

    # return pf, signal, target_weights
    return pf, target_weights



def weights_stat_report(target_weights: pd.DataFrame):
    # Сколько акций держим в среднем (не нулевых)
    num_positions = (target_weights > 0).sum(axis=1)
    # num_positions — Series с количеством акций в портфеле на каждую дату
    min_nonzero = num_positions[num_positions > 0].min()

    print("Среднее количество акций в портфеле:", num_positions.mean())
    print("Минимальное количество акций в портфеле:", num_positions.min())
    print("Минимальное ненулевое количество акций в портфеле:", min_nonzero)
    print("Максимальное количество акций в портфеле:", num_positions.max())

    # Средние, медианные, стандартные веса
    mean_weights = target_weights.mean()
    median_weights = target_weights.median()
    std_weights = target_weights.std()




def plot_nist_stock_in_profile(target_weights: pd.DataFrame):
    num_positions = (target_weights > 0).sum(axis=1)
    plt.figure(figsize=figsize)
    plt.hist(num_positions, bins=range(num_positions.min(), num_positions.max()+2), color='skyblue', edgecolor='black', align='left')
    plt.title("Распределение количества акций в портфеле")
    plt.xlabel("Количество акций")
    plt.ylabel("Частота")
    plt.show()


def plot_stock_weights_in_profile(target_weights: pd.DataFrame):
    avg_weights = target_weights.mean().sort_values(ascending=False)

    plt.figure(figsize=figsize)
    avg_weights.plot(kind='bar', color='steelblue')
    plt.title("Средний вес каждой акции в портфеле")
    plt.ylabel("Средний вес")
    plt.xlabel("Акции")
    plt.show()

def plot_hist_stock_weights(target_weights: pd.DataFrame):
    plt.figure(figsize=(8,5))
    plt.hist(target_weights.values.flatten(), bins=50, color='lightgreen', edgecolor='black')
    plt.title("Распределение весов акций")
    plt.xlabel("Вес акции")
    plt.ylabel("Частота")
    plt.show()


def plot_hist_returns(pf):
    # Дневные доходности — Series с индексом Date
    daily_returns = pf.returns()

    plt.figure(figsize=figsize)
    plt.hist(daily_returns, bins=50, color='skyblue', edgecolor='black')
    plt.title("Распределение дневных доходностей")
    plt.xlabel("Daily Return")
    plt.ylabel("Frequency")
    plt.show()



    # Агрегируем в недельные (сумма доходностей за неделю)
    weekly_returns = daily_returns.resample('W').sum()
    plt.figure(figsize=figsize)
    plt.hist(weekly_returns, bins=50, color='skyblue', edgecolor='black')
    plt.title("Распределение недельных доходностей")
    plt.xlabel("Weekly Return")
    plt.ylabel("Frequency")
    plt.show()

    # Агрегируем в недельные (сумма доходностей за неделю)
    m_returns = daily_returns.resample('ME').sum()
    plt.figure(figsize=figsize)
    plt.hist(m_returns, bins=50, color='skyblue', edgecolor='black')
    plt.title("Распределение доходностей по месецам")
    plt.xlabel("Weekly Return")
    plt.ylabel("Frequency")
    plt.show()




def extended_stats(pf, alpha=0.05):
    """
    Расширенные метрики портфеля на основе vectorbt Portfolio pf.
    alpha — уровень для VaR (например, 0.05 = 5%)
    """
    stats = pf.stats()  # базовые метрики
    returns = pf.returns()

    extended = stats.copy() if isinstance(stats, pd.Series) else pd.Series(stats)

    # -----------------------------
    # 1️⃣ Volatility (Std Dev)
    # -----------------------------
    extended['Volatility [%]'] = returns.std() * 100

    # -----------------------------
    # 2️⃣ Sortino Ratio
    # -----------------------------
    downside_std = returns[returns < 0].std()
    extended['Sortino Ratio'] = (returns.mean() / downside_std) * np.sqrt(252) if downside_std != 0 else np.nan

    # -----------------------------
    # 3️⃣ Skewness & Kurtosis
    # -----------------------------
    extended['Skewness'] = skew(returns)
    extended['Kurtosis'] = kurtosis(returns)

    # -----------------------------
    # 4️⃣ Value at Risk (VaR)
    # -----------------------------
    extended[f'VaR {int(alpha*100)}%'] = np.percentile(returns, alpha*100)

    # -----------------------------
    # 5️⃣ Conditional VaR (CVaR)
    # -----------------------------
    var_threshold = extended[f'VaR {int(alpha*100)}%']
    extended[f'CVaR {int(alpha*100)}%'] = returns[returns <= var_threshold].mean()

    # -----------------------------
    # 6️⃣ Profit/Loss Ratio
    # -----------------------------
    if hasattr(pf, 'trades') and pf.trades.records is not None:
        trades = pf.trades.records
        avg_win = trades[trades['return'] > 0]['return'].mean()
        avg_loss = trades[trades['return'] < 0]['return'].mean()
        extended['Profit/Loss Ratio'] = avg_win / abs(avg_loss) if avg_loss != 0 else np.nan

    # --- Годовые доходности ---
    returns = pf.returns()  # Series с дневными доходностями
    returns.index = pd.to_datetime(returns.index)

    # Группируем по году и считаем суммарную доходность каждого года
    annual_returns = (1 + returns).resample("YE").prod() - 1
    annual_returns_percent = annual_returns * 100  # в процентах

    # Добавляем годовые доходности в статистику
    for year, value in zip(annual_returns_percent.index.year, annual_returns_percent.values):
        extended[f"Return {year} [%]"] = value

    # Средняя, минимальная, максимальная годовая доходность
    extended["Average Annual Return [%]"] = annual_returns_percent.mean()
    extended["Min Annual Return [%]"] = annual_returns_percent.min()
    extended["Max Annual Return [%]"] = annual_returns_percent.max()

    return extended


def rolling_sharp(pf,  rolling_N = 100):
    returns = pf.returns()  # Series с дневными доходностями

    # -----------------------------
    # Rolling Sharpe Ratio (окно rolling_N дня)
    # -----------------------------
    rolling_sharpe = returns.rolling(rolling_N).apply(lambda x: x.mean() / x.std() * np.sqrt(252), raw=False)

    plt.figure(figsize=figsize)
    plt.plot(rolling_sharpe, color='blue')
    plt.title(f"Rolling {rolling_N}-day Sharpe Ratio")
    plt.xlabel("Date")
    plt.ylabel("Sharpe Ratio")
    plt.grid(True, alpha=0.3)
    plt.show()


def rolling_valatility(pf, rolling_N = 100):
    # -----------------------------
    # Rolling Volatility (окно rolling_N дня)
    # -----------------------------
    returns = pf.returns()  # Series с дневными доходностями
    rolling_vol = returns.rolling(rolling_N).std() * np.sqrt(252)  # годовая нормализация

    plt.figure(figsize=figsize)
    plt.plot(rolling_vol, color='red')
    plt.title(f"Rolling {rolling_N=}-day Volatility")
    plt.xlabel("Date")
    plt.ylabel("Volatility")
    plt.grid(True, alpha=0.3)
    plt.show()


def plot_portfolio_value(pf):
    # Получаем дневные цены портфеля
    portfolio_value = pf.value  # Series с дневной стоимостью портфеля

    plt.figure(figsize=figsize)
    plt.plot(portfolio_value().index, portfolio_value().values, label="Portfolio Value")
    plt.title("Стоимость портфеля")
    plt.xlabel("Дата")
    plt.ylabel("Стоимость")
    plt.grid(True)
    plt.legend()
    plt.show()



def plot_dayly_return(pf):
    # Получаем Series дневных доходностей
    daily_returns = pf.returns()

    plt.figure(figsize=figsize)
    plt.plot(daily_returns.index, daily_returns.values, color="steelblue", linewidth=1.2)
    plt.title("Daily Returns", fontsize=14)
    plt.xlabel("Дата")
    plt.ylabel("Доходность")
    plt.grid(True, alpha=0.3)
    plt.axhline(0, color='black', linewidth=0.8, linestyle='--')  # линия нуля
    plt.tight_layout()
    plt.show()




def plot_allocation(rb_pf, target_weights):
    rb_asset_value = rb_pf.asset_value(group_by=False)
    rb_value = rb_pf.value()
    rb_idxs = np.flatnonzero((rb_pf.asset_flow() != 0).any(axis=1))
    rb_dates = rb_pf.wrapper.index[rb_idxs]
    fig = (rb_asset_value.vbt / rb_value).vbt.plot(
        trace_names=target_weights.columns,
        trace_kwargs=dict(
            stackgroup='one'
        )
    )
    for rb_date in rb_dates:
        fig.add_shape(
            dict(
                xref='x',
                yref='paper',
                x0=rb_date,
                x1=rb_date,
                y0=0,
                y1=1,
                line_color=fig.layout.template.layout.plot_bgcolor
            )
        )
    fig.show()







def load_backtesting_dataset():
    data_path = "../data/moex_final_dataset.csv"
    df = pd.read_csv(data_path)
    df.sort_values(by=['Date', 'Ticker'], inplace=True)
    return df



def save_portfolio_stats(stats: pd.Series, file_path: str):
    """
    Сохраняет статистику портфеля в файл.

    :param stats: pd.Series с расширенной статистикой портфеля
    :param file_path: путь к файлу, например 'results/strategy1_stats.csv'
    :param file_format: формат сохранения: 'csv' или 'json' (по умолчанию 'csv')
    """
    # Проверяем существование папки
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    # Преобразуем Series в DataFrame для сохранения
    df_stats = stats.reset_index()
    df_stats.columns = ["Metric", "Value"]
    df_stats.to_csv(file_path, index=False)
    print(f"✅ Статистика сохранена в CSV: {file_path}")

    # Описание файла для сохранения
    """
    | Метрика                        | Определение                                         | Интервалы / Интерпретация                                                                                             |
    | ------------------------------ | --------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------- |
    | **Start**                      | Дата начала анализа                                 | Просто дата, нет интервалов                                                                                           |
    | **End**                        | Дата окончания анализа                              | Просто дата, нет интервалов                                                                                           |
    | **Period**                     | Длительность анализа                                | Короткий (<1 год), средний (1–3 года), длинный (>3 года)                                                              |
    | **Start Value**                | Начальная стоимость портфеля                        | —                                                                                                                     |
    | **End Value**                  | Конечная стоимость портфеля                         | —                                                                                                                     |
    | **Total Return [%]**           | Общая доходность портфеля                           | <0% → убыток, 0–20% → низкая, 20–50% → средняя, >50% → отличная                                                       |
    | **Benchmark Return [%]**       | Доходность бенчмарка                                | —                                                                                                                     |
    | **Max Gross Exposure [%]**     | Максимальная суммарная экспозиция                   | 100% → полностью задействован капитал, >100% → с плечом                                                               |
    | **Total Fees Paid**            | Комиссии                                            | —                                                                                                                     |
    | **Max Drawdown [%]**           | Максимальная просадка                               | <10% → низкая, 10–20% → умеренная, 20–40% → высокая, >40% → очень высокая                                             |
    | **Max Drawdown Duration**      | Длительность максимальной просадки                  | Чем меньше, тем лучше                                                                                                 |
    | **Total Trades**               | Количество сделок                                   | —                                                                                                                     |
    | **Total Closed Trades**        | Количество закрытых сделок                          | —                                                                                                                     |
    | **Total Open Trades**          | Сделки в процессе                                   | —                                                                                                                     |
    | **Open Trade PnL**             | PnL по открытым сделкам                             | —                                                                                                                     |
    | **Win Rate [%]**               | Доля прибыльных сделок                              | <50% → плохо, 50–60% → средне, 60–70% → хорошо, >70% → очень хорошо                                                   |
    | **Best Trade [%]**             | Максимальная прибыль по сделке                      | Чем выше — лучше                                                                                                      |
    | **Worst Trade [%]**            | Максимальный убыток по сделке                       | Чем меньше отрицательное число — лучше                                                                                |
    | **Avg Winning Trade [%]**      | Средняя прибыль по выигрышным сделкам               | Чем выше — лучше                                                                                                      |
    | **Avg Losing Trade [%]**       | Средний убыток по убыточным сделкам                 | Чем меньше — лучше                                                                                                    |
    | **Avg Winning Trade Duration** | Средняя длительность выигрышной сделки              | —                                                                                                                     |
    | **Avg Losing Trade Duration**  | Средняя длительность убыточной сделки               | —                                                                                                                     |
    | **Profit Factor**              | Суммарная прибыль / суммарный убыток                | <1 → убыточно, 1–2 → средне, 2–3 → хорошо, >3 → отлично                                                               |
    | **Expectancy**                 | Средний ожидаемый PnL на сделку                     | >0 → стратегия прибыльна                                                                                              |
    | **Sharpe Ratio**               | Средняя доходность / волатильность                  | <0 → плохо, 0–1 → средне, 1–2 → хорошо, >2 → отлично                                                                  |
    | **Calmar Ratio**               | Доходность / Max Drawdown                           | <0.2 → низкая эффективность, 0.2–0.5 → средняя, >0.5 → высокая эффективность                                          |
    | **Omega Ratio**                | Отношение положительных доходностей к отрицательным | >1 → прибыльная стратегия, выше — лучше                                                                               |
    | **Sortino Ratio**              | Sharpe с учётом только отрицательной волатильности  | <0.5 → низкая, 0.5–1 → средняя, >1 → хорошая                                                                          |
    | **Volatility [%]**             | Стандартное отклонение доходностей                  | <5% → низкая, 5–15% → средняя, >15% → высокая                                                                         |
    | **Skewness**                   | Асимметрия распределения доходностей                | <0 → левосторонний хвост (редкие большие убытки), 0 → симметрично, >0 → правосторонний хвост (редкие большие прибыли) |
    | **Kurtosis**                   | “Жирность хвостов” распределения                    | 3 → нормальное, >3 → частые экстремальные значения, <3 → редкие экстремумы                                            |
    | **VaR 5%**                     | Потеря, которую не превысим с вероятностью 95%      | Чем меньше (отрицательное число), тем больше потенциальный риск                                                       |
    | **CVaR 5%**                    | Средний убыток в худших 5% случаев                  | Чем меньше, тем больше риск                                                                                           |
    | **Profit/Loss Ratio**          | Средняя прибыль / средний убыток на сделку          | >1 → выигрышные сделки больше проигрышных, <1 → наоборот                                                              |

    """