import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set page config
st.set_page_config(page_title="Stock Feature Visualizer", layout="wide")

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv('data/moex_with_features.csv', parse_dates=['Date'])
    return df

df = load_data()

# Sidebar inputs
st.sidebar.title("Filters")

# Date range
min_date = df['Date'].min()
max_date = df['Date'].max()
date_range = st.sidebar.date_input("Select Date Range", [min_date, max_date], min_value=min_date, max_value=max_date)

# Features (excluding non-feature columns)
non_feature_cols = ['Date', 'Ticker', 'open', 'high', 'low', 'close', 'volume', 'OHLCV']
feature_cols = [col for col in df.columns if col not in non_feature_cols]
selected_feature = st.sidebar.selectbox("Select Feature", feature_cols)

# Tickers
Tickers = df['Ticker'].unique().tolist()
selected_Tickers = st.sidebar.multiselect("Select Tickers", Tickers, default=Tickers)

# Filter data
filtered_df = df[
    (df['Date'] >= pd.to_datetime(date_range[0])) &
    (df['Date'] <= pd.to_datetime(date_range[1])) &
    (df['Ticker'].isin(selected_Tickers))
]

# MAIN content
st.title("📈 Feature Trend & Statistics Viewer")

if filtered_df.empty:
    st.warning("No data available for the selected filters.")
else:
    # LINE PLOT
    st.subheader(f"Line Plot: `{selected_feature}` over Time")
    fig, ax = plt.subplots(figsize=(12, 5))
    for Ticker in selected_Tickers:
        Ticker_data = filtered_df[filtered_df['Ticker'] == Ticker]
        ax.plot(Ticker_data['Date'], Ticker_data[selected_feature], label=Ticker)
    ax.set_title(f"{selected_feature} over time")
    ax.set_xlabel("Date")
    ax.set_ylabel(selected_feature)
    ax.legend()
    st.pyplot(fig)

    # HISTOGRAM
    st.subheader("Distribution Histogram")
    fig2, ax2 = plt.subplots(figsize=(10, 4))
    sns.histplot(filtered_df[selected_feature], bins=50, kde=True, ax=ax2)
    ax2.set_title(f"Distribution of {selected_feature}")
    st.pyplot(fig2)

    # STATISTICS TABLE
    # 📊 Feature Statistics with Filtering
    st.subheader("📊 Feature Statistics (with filters)")

    # Compute stats
    stats_df = filtered_df.groupby('Ticker')[selected_feature].agg(
        ['count', 'min', 'max', 'mean', 'median', 'std', 'skew']
    ).reset_index()

    # Rename columns for clarity
    stats_df.columns = ['Ticker', 'count', 'min', 'max', 'mean', 'median', 'std', 'skew']

    # Filtering form
    with st.expander("🔍 Filter Statistics Table"):
        col1, col2 = st.columns(2)
        
        with col1:
            min_count = st.number_input("Minimum of 'count'", value=float(stats_df['count'].min()))
            max_count = st.number_input("Maximum of 'count'", value=float(stats_df['count'].max()))


            min_min = st.number_input("Minimum of 'min'", value=float(stats_df['min'].min()))
            max_min = st.number_input("Maximum of 'min'", value=float(stats_df['min'].max()))
            
            min_mean = st.number_input("Minimum of 'mean'", value=float(stats_df['mean'].min()))
            max_mean = st.number_input("Maximum of 'mean'", value=float(stats_df['mean'].max()))
            
            min_std = st.number_input("Minimum of 'std'", value=float(stats_df['std'].min()))
            max_std = st.number_input("Maximum of 'std'", value=float(stats_df['std'].max()))
            
        with col2:
            min_max = st.number_input("Minimum of 'max'", value=float(stats_df['max'].min()))
            max_max = st.number_input("Maximum of 'max'", value=float(stats_df['max'].max()))
            
            min_median = st.number_input("Minimum of 'median'", value=float(stats_df['median'].min()))
            max_median = st.number_input("Maximum of 'median'", value=float(stats_df['median'].max()))
            
            min_skew = st.number_input("Minimum of 'skew'", value=float(stats_df['skew'].min()))
            max_skew = st.number_input("Maximum of 'skew'", value=float(stats_df['skew'].max()))

    # Apply filters
    filtered_stats_df = stats_df[
        (stats_df['count'] >= min_count) & (stats_df['count'] <= max_count) &
        (stats_df['min'] >= min_min) & (stats_df['min'] <= max_min) &
        (stats_df['max'] >= min_max) & (stats_df['max'] <= max_max) &
        (stats_df['mean'] >= min_mean) & (stats_df['mean'] <= max_mean) &
        (stats_df['median'] >= min_median) & (stats_df['median'] <= max_median) &
        (stats_df['std'] >= min_std) & (stats_df['std'] <= max_std) &
        (stats_df['skew'] >= min_skew) & (stats_df['skew'] <= max_skew)
    ]

    # Show result
    st.dataframe(filtered_stats_df.style.format(precision=3), use_container_width=True)
