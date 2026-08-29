import os
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

# Configure Web Page Layout
st.set_page_config(
    page_title="MA55 Channel, RSI, MACD & SMA Cross Screener",
    page_icon="📊",
    layout="wide",
)

# --- CONFIGURATION CONSTANTS ---
TICKER_FILE = "tickers.txt"
MA_PERIOD = 55
RSI_PERIOD = 14
SMA_FAST_PERIOD = 13
SMA_SLOW_PERIOD = 34
MAX_CANDLES_AGO = 10  # Look back up to 10 candles for breakout signals

# Mapping display timeframes to yfinance interval & period parameters
TIMEFRAME_CONFIG = {
    "1 Day": {"interval": "1d", "period": "200d", "unit": "Days"},
    "4 Hours": {"interval": "1h", "period": "60d", "unit": "4H Bars (Est.)"},
    "1 Hour": {"interval": "1h", "period": "60d", "unit": "Hours"},
    "15 Mins": {"interval": "15m", "period": "30d", "unit": "15m Bars"},
}


def load_tickers(filepath=TICKER_FILE):
    if not os.path.exists(filepath):
        st.warning(f"⚠️ '{filepath}' not found. Using default tickers.")
        return ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA"]

    with open(filepath, "r") as f:
        tickers = [
            line.strip().upper()
            for line in f
            if line.strip() and not line.startswith("#")
        ]

    return list(dict.fromkeys(tickers))


def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0)
    loss = (-delta).where(delta < 0, 0)

    avg_gain = gain.ewm(
        alpha=1 / period, min_periods=period, adjust=False
    ).mean()
    avg_loss = loss.ewm(
        alpha=1 / period, min_periods=period, adjust=False
    ).mean()

    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def calculate_macd(series, fast=12, slow=26, signal=9):
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    hist = macd_line - signal_line
    return macd_line, signal_line, hist


def find_latest_cross(fast_series, slow_series):
    """Finds the type of cross (Bull/Bear) and how many candles ago it occurred."""
    diff = fast_series - slow_series
    # Identify index locations where sign changes (crossovers)
    crosses = (np.sign(diff) != np.sign(diff.shift(1))) & (diff.shift(1).notna())

    cross_indices = np.where(crosses)[0]
    if len(cross_indices) == 0:
        return "None", None

    latest_cross_idx = cross_indices[-1]
    candles_ago = len(series_len := fast_series) - 1 - latest_cross_idx

    # Determine direction of the cross
    if diff.iloc[latest_cross_idx] > 0 and diff.iloc[latest_cross_idx - 1] <= 0:
        cross_type = "Bull Cross"
    elif diff.iloc[latest_cross_idx] < 0 and diff.iloc[latest_cross_idx - 1] >= 0:
        cross_type = "Bear Cross"
    else:
        cross_type = "None"

    return cross_type, candles_ago


def resample_4h(df):
    """Resample 1-hour data into 4-hour candles for accurate 4H MA calculation."""
    resampled = (
        df.resample("4h")
        .agg({
            "Open": "first",
            "High": "max",
            "Low": "min",
            "Close": "last",
            "Volume": "sum",
        })
        .dropna()
    )
    return resampled


@st.cache_data(ttl=900, show_spinner=False)
def run_screener(ticker_list, timeframe_label):
    if not ticker_list:
        return pd.DataFrame()

    tf_info = TIMEFRAME_CONFIG[timeframe_label]
    interval = tf_info["interval"]
    period = tf_info["period"]

    data = yf.download(
        ticker_list, period=period, interval=interval, group_by="ticker"
    )
    results = []

    for ticker in ticker_list:
        try:
            if len(ticker_list) == 1:
                df = data.copy()
            else:
                if ticker not in data.columns.levels[0]:
                    continue
                df = data[ticker].dropna()

            # Resample to 4H if selected
            if timeframe_label == "4 Hours":
                df = resample_4h(df)

            min_required = max(MA_PERIOD, SMA_SLOW_PERIOD, 26 + 9) + MAX_CANDLES_AGO + 2
            if len(df) < min_required:
                continue

            # Indicator Calculations
            df["MA_High"] = df["High"].rolling(window=MA_PERIOD).mean()
            df["MA_Low"] = df["Low"].rolling(window=MA_PERIOD).mean()
            df["RSI"] = calculate_rsi(df["Close"], RSI_PERIOD)

            # SMA 13 & SMA 34
            df["SMA13"] = df["Close"].rolling(window=SMA_FAST_PERIOD).mean()
            df["SMA34"] = df["Close"].rolling(window=SMA_SLOW_PERIOD).mean()

            # MACD (12, 26, 9)
            df["MACD"], df["MACD_Signal"], df["MACD_Hist"] = calculate_macd(df["Close"])

            # Detect MACD and SMA Crosses
            macd_cross_status, macd_cross_ago = find_latest_cross(df["MACD"], df["MACD_Signal"])
            sma_cross_status, sma_cross_ago = find_latest_cross(df["SMA13"], df["SMA34"])

            last_rsi = (
                round(df["RSI"].iloc[-1], 2)
                if not pd.isna(df["RSI"].iloc[-1])
                else None
            )

            # Signal evaluation for MA55 Channel Breakouts / Touches
            for i in range(1, MAX_CANDLES_AGO + 2):
                curr = df.iloc[-i]
                prev = df.iloc[-i - 1]

                c_close, c_high, c_low = (
                    curr["Close"],
                    curr["High"],
                    curr["Low"],
                )
                c_ma_h, c_ma_l = curr["MA_High"], curr["MA_Low"]
                p_close, p_ma_h, p_ma_l = (
                    prev["Close"],
                    prev["MA_High"],
                    prev["MA_Low"],
                )

                if pd.isna(c_ma_h) or pd.isna(c_ma_l):
                    continue

                status = None
                if p_close <= p_ma_h and c_close > c_ma_h:
                    status = "Bull"
                elif p_close >= p_ma_l and c_close < c_ma_l:
                    status = "Bear"
                elif (
                    (p_close > p_ma_h)
                    and (c_low <= c_ma_h <= c_high)
                    and (c_close > c_ma_h)
                ):
                    status = "Touch Bull"
                elif (
                    (p_close < p_ma_l)
                    and (c_low <= c_ma_l <= c_high)
                    and (c_close < c_ma_l)
                ):
                    status = "Touch Bear"

                if status:
                    results.append({
                        "Ticker": ticker,
                        "Status": status,
                        "Candles Ago": i - 1,
                        "Last Price": round(c_close, 2),
                        "MA High": round(c_ma_h, 2),
                        "MA Low": round(c_ma_l, 2),
                        "RSI (14)": last_rsi,
                        "MACD Signal": macd_cross_status,
                        "MACD Cross Ago": macd_cross_ago,
                        "SMA (13/34)": sma_cross_status,
                        "SMA Cross Ago": sma_cross_ago,
                    })
                    break

        except Exception:
            continue

    return pd.DataFrame(results)


# --- STYLING FUNCTIONS ---
def style_status(val):
    if "Bull" in str(val):
        return "background-color: #1b382b; color: #4eff9e; font-weight: bold;"
    elif "Bear" in str(val):
        return "background-color: #3d1c1d; color: #ff6b6b; font-weight: bold;"
    return ""


def style_rsi(val, oversold, overbought):
    if pd.isna(val):
        return ""
    if val <= oversold:
        return "background-color: #1b382b; color: #4eff9e; font-weight: bold;"
    elif val >= overbought:
        return "background-color: #3d1c1d; color: #ff6b6b; font-weight: bold;"
    return ""


def apply_table_styles(df, oversold_val, overbought_val):
    return (
        df.style.map(style_status, subset=["Status", "MACD Signal", "SMA (13/34)"])
        .map(
            style_rsi,
            subset=["RSI (14)"],
            oversold=oversold_val,
            overbought=overbought_val,
        )
    )


# ==================== STREAMLIT UI ====================

st.title("📊 Multi-Timeframe MA55, RSI, MACD & SMA Screener")
st.caption(
    "Dynamic automated market screening for trading signals, MACD crossovers, and SMA 13/34 crossovers."
)

tickers = load_tickers(TICKER_FILE)

# Sidebar Parameter Controls
with st.sidebar:
    st.header("Screener Controls")

    selected_tf = st.selectbox(
        "⏱ Select Timeframe",
        options=list(TIMEFRAME_CONFIG.keys()),
        index=0,  # Default to 1 Day
    )

    st.write(f"📁 Loaded Tickers: **{len(tickers)}**")
    st.write(f"📈 MA Channel: **{MA_PERIOD} Period ({selected_tf})**")
    st.write(f"📊 SMA Cross: **13 / 34 ({selected_tf})**")

    st.markdown("---")
    st.subheader("RSI Thresholds")

    rsi_oversold = st.slider(
        "Oversold Threshold (Green)",
        min_value=10,
        max_value=45,
        value=30,
        step=1,
    )

    rsi_overbought = st.slider(
        "Overbought Threshold (Red)",
        min_value=55,
        max_value=90,
        value=70,
        step=1,
    )

    st.markdown("---")
    if st.button("🔄 Force Manual Refresh", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

with st.spinner(f"Screening {len(tickers)} symbols on {selected_tf} timeframe..."):
    df_results = run_screener(tickers, selected_tf)

time_unit = TIMEFRAME_CONFIG[selected_tf]["unit"]

if not df_results.empty:
    priority_map = {"Bull": 0, "Bear": 1, "Touch Bull": 2, "Touch Bear": 3}
    df_results["Priority"] = df_results["Status"].map(priority_map)
    df_results = df_results.sort_values(
        ["Priority", "Candles Ago"]
    ).drop(columns="Priority")

    oversold_count = len(df_results[df_results["RSI (14)"] <= rsi_oversold])
    overbought_count = len(df_results[df_results["RSI (14)"] >= rsi_overbought])

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Total Signals", len(df_results))
    col2.metric("Bull Breakouts", len(df_results[df_results["Status"] == "Bull"]))
    col3.metric("Bear Breakouts", len(df_results[df_results["Status"] == "Bear"]))
    col4.metric(f"Oversold (≤ {rsi_oversold})", oversold_count)
    col5.metric(f"Overbought (≥ {rsi_overbought})", overbought_count)

    st.markdown("---")

    # Split output into Active (<= 3 candles ago) vs Earlier (> 3 candles ago)
    df_recent = df_results[df_results["Candles Ago"] <= 3]
    df_older = df_results[df_results["Candles Ago"] > 3]

    column_formatting = {
        "Ticker": st.column_config.TextColumn("Ticker"),
        "Status": st.column_config.TextColumn("Signal Type"),
        "Candles Ago": st.column_config.NumberColumn(f"Candles Ago ({time_unit})"),
        "Last Price": st.column_config.NumberColumn("Last Price", format="$%.2f"),
        "MA High": st.column_config.NumberColumn("MA High (55)", format="$%.2f"),
        "MA Low": st.column_config.NumberColumn("MA Low (55)", format="$%.2f"),
        "RSI (14)": st.column_config.NumberColumn("RSI (14)", format="%.2f"),
        "MACD Signal": st.column_config.TextColumn("MACD Cross"),
        "MACD Cross Ago": st.column_config.NumberColumn(f"MACD Ago ({time_unit})"),
        "SMA (13/34)": st.column_config.TextColumn("SMA 13/34 Cross"),
        "SMA Cross Ago": st.column_config.NumberColumn(f"SMA Ago ({time_unit})"),
    }

    st.subheader(f"🔥 Active Signals (Last 3 {time_unit})")
    if not df_recent.empty:
        styled_recent = apply_table_styles(df_recent, rsi_oversold, rsi_overbought)
        st.dataframe(styled_recent, use_container_width=True, hide_index=True, column_config=column_formatting)
    else:
        st.info(f"No active signals detected in the last 3 {time_unit.lower()}.")

    st.markdown("---")

    st.subheader(f"📋 Earlier Signals (> 3 {time_unit})")
    if not df_older.empty:
        styled_older = apply_table_styles(df_older, rsi_oversold, rsi_overbought)
        st.dataframe(styled_older, use_container_width=True, hide_index=True, column_config=column_formatting)
    else:
        st.caption("No older signals present in current dataset.")
else:
    st.info(f"No signals detected across {len(tickers)} symbols on the {selected_tf} timeframe.")
