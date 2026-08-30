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
MAX_CANDLES_AGO = 10

TIMEFRAME_CONFIG = {
    "1 Day": {"interval": "1d", "period": "200d", "unit": "Days"},
    "4 Hours": {"interval": "1h", "period": "60d", "unit": "4H Bars"},
    "1 Hour": {"interval": "1h", "period": "60d", "unit": "Hours"},
    "15 Mins": {"interval": "15m", "period": "30d", "unit": "15m Bars"},
}


def load_tickers(filepath=TICKER_FILE):
    if not os.path.exists(filepath):
        st.warning(f"⚠️ '{filepath}' not found. Using default tickers.")
        return ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA"]
    with open(filepath, "r") as f:
        tickers = [line.strip().upper() for line in f if line.strip() and not line.startswith("#")]
    return list(dict.fromkeys(tickers))


def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0)
    loss = (-delta).where(delta < 0, 0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
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
    diff = fast_series - slow_series
    crosses = (np.sign(diff) != np.sign(diff.shift(1))) & (diff.shift(1).notna())
    cross_indices = np.where(crosses)[0]
    if len(cross_indices) == 0:
        return "None", None
    latest_cross_idx = cross_indices[-1]
    candles_ago = len(fast_series) - 1 - latest_cross_idx
    if diff.iloc[latest_cross_idx] > 0 and diff.iloc[latest_cross_idx - 1] <= 0:
        return "Bull Cross", candles_ago
    elif diff.iloc[latest_cross_idx] < 0 and diff.iloc[latest_cross_idx - 1] >= 0:
        return "Bear Cross", candles_ago
    return "None", None


def resample_4h(df):
    return df.resample("4h").agg({
        "Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"
    }).dropna()


@st.cache_data(ttl=900, show_spinner=False)
def run_screener(ticker_list, timeframe_label):
    if not ticker_list:
        return pd.DataFrame()
    
    tf_info = TIMEFRAME_CONFIG[timeframe_label]
    interval = tf_info["interval"]
    period = tf_info["period"]
    
    data = yf.download(ticker_list, period=period, interval=interval, group_by="ticker")
    data_daily = yf.download(ticker_list, period="1mo", interval="1d", group_by="ticker")
    
    results = []
    for ticker in ticker_list:
        try:
            if len(ticker_list) == 1:
                df = data.copy()
                df_d = data_daily.copy()
            else:
                if ticker not in data.columns.levels[0]:
                    continue
                df = data[ticker].dropna()
                df_d = data_daily[ticker].dropna() if ticker in data_daily.columns.levels[0] else pd.DataFrame()
            
            if timeframe_label == "4 Hours":
                df = resample_4h(df)
            
            min_required = max(MA_PERIOD, SMA_SLOW_PERIOD, 35) + MAX_CANDLES_AGO + 2
            if len(df) < min_required or len(df_d) < 2:
                continue
            
            prev_day_high = df_d["High"].iloc[-2]
            prev_day_low = df_d["Low"].iloc[-2]
            
            df["MA_High"] = df["High"].rolling(window=MA_PERIOD).mean()
            df["MA_Low"] = df["Low"].rolling(window=MA_PERIOD).mean()
            df["RSI"] = calculate_rsi(df["Close"], RSI_PERIOD)
            df["SMA13"] = df["Close"].rolling(window=SMA_FAST_PERIOD).mean()
            df["SMA34"] = df["Close"].rolling(window=SMA_SLOW_PERIOD).mean()
            df["MACD"], df["MACD_Signal"], df["MACD_Hist"] = calculate_macd(df["Close"])
            
            macd_cross_status, macd_cross_ago = find_latest_cross(df["MACD"], df["MACD_Signal"])
            sma_cross_status, sma_cross_ago = find_latest_cross(df["SMA13"], df["SMA34"])
            last_rsi = round(df["RSI"].iloc[-1], 2) if not pd.isna(df["RSI"].iloc[-1]) else None
            
            for i in range(1, MAX_CANDLES_AGO + 2):
                curr = df.iloc[-i]
                prev = df.iloc[-i - 1]
                c_close, c_high, c_low = curr["Close"], curr["High"], curr["Low"]
                c_ma_h, c_ma_l = curr["MA_High"], curr["MA_Low"]
                p_close, p_ma_h, p_ma_l = prev["Close"], prev["MA_High"], prev["MA_Low"]
                
                if pd.isna(c_ma_h) or pd.isna(c_ma_l):
                    continue
                
                status = None
                if p_close <= p_ma_h and c_close > c_ma_h:
                    status = "Bull"
                elif p_close >= p_ma_l and c_close < c_ma_l:
                    status = "Bear"
                elif (p_close > p_ma_h) and (c_low <= c_ma_h <= c_high) and (c_close > c_ma_h):
                    status = "Touch Bull"
                elif (p_close < p_ma_l) and (c_low <= c_ma_l <= c_high) and (c_close < c_ma_l):
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
                        "Prev Day High": round(prev_day_high, 2),
                        "Prev Day Low": round(prev_day_low, 2),
                    })
                    break
        except Exception:
            continue
    return pd.DataFrame(results)


# ==================== GITHUB EXPORT HELPERS ====================

def format_signal_emoji(status):
    """Return GitHub-compatible emoji indicators."""
    if pd.isna(status):
        return "⚪ —"
    status = str(status)
    if "Bull" in status and "Touch" not in status:
        return "🟢 **Bull**"
    elif "Bear" in status and "Touch" not in status:
        return "🔴 **Bear**"
    elif "Touch Bull" in status:
        return "🟡 Touch Bull"
    elif "Touch Bear" in status:
        return "🟠 Touch Bear"
    elif "Bull Cross" in status:
        return "🟢 Bull Cross ⬆️"
    elif "Bear Cross" in status:
        return "🔴 Bear Cross ⬇️"
    return f"⚪ {status}"


def format_rsi_github(val, oversold, overbought):
    """Format RSI with GitHub-compatible emoji indicators."""
    if pd.isna(val):
        return "—"
    val = float(val)
    if val <= oversold:
        return f"🟢 **{val:.1f}**"
    elif val >= overbought:
        return f"🔴 **{val:.1f}**"
    return f"⚪ {val:.1f}"


def format_price_github(val):
    if pd.isna(val):
        return "—"
    return f"`${val:.2f}`"


def df_to_github_markdown(df, oversold, overbought):
    """
    Convert DataFrame to GitHub-flavored markdown table.
    Uses emoji instead of CSS (GitHub strips colors).
    """
    if df.empty:
        return "*No signals detected.*"
    
    df_md = df.copy()
    
    # Apply emoji formatting
    df_md["Status"] = df_md["Status"].apply(format_signal_emoji)
    df_md["MACD Signal"] = df_md["MACD Signal"].apply(format_signal_emoji)
    df_md["SMA (13/34)"] = df_md["SMA (13/34)"].apply(format_signal_emoji)
    df_md["RSI (14)"] = df_md["RSI (14)"].apply(lambda x: format_rsi_github(x, oversold, overbought))
    
    for col in ["Last Price", "MA High", "MA Low", "Prev Day High", "Prev Day Low"]:
        if col in df_md.columns:
            df_md[col] = df_md[col].apply(format_price_github)
    
    # Clean column names for display
    df_md = df_md.rename(columns={
        "Candles Ago": "Bars Ago",
        "MACD Cross Ago": "MACD Bars",
        "SMA Cross Ago": "SMA Bars"
    })
    
    return df_md.to_markdown(index=False)


def df_to_github_html(df, oversold, overbought):
    """
    Generate a self-contained HTML file that works on GitHub Pages.
    Uses inline styles that are supported in GitHub Pages.
    """
    if df.empty:
        return "<p>No signals detected.</p>"
    
    df_html = df.copy()
    
    def color_status(val):
        if pd.isna(val):
            return val
        val = str(val)
        if val == "Bull":
            return f'<span style="color:#238636;font-weight:bold;">● Bull</span>'
        elif val == "Bear":
            return f'<span style="color:#da3633;font-weight:bold;">● Bear</span>'
        elif "Touch Bull" in val:
            return f'<span style="color:#1f6feb;">○ Touch Bull</span>'
        elif "Touch Bear" in val:
            return f'<span style="color:#d29922;">○ Touch Bear</span>'
        elif "Bull Cross" in val:
            return f'<span style="color:#238636;">↗ Bull Cross</span>'
        elif "Bear Cross" in val:
            return f'<span style="color:#da3633;">↘ Bear Cross</span>'
        return val
    
    def color_rsi(val):
        if pd.isna(val):
            return val
        val = float(val)
        if val <= oversold:
            return f'<span style="color:#238636;font-weight:bold;">{val:.1f}</span>'
        elif val >= overbought:
            return f'<span style="color:#da3633;font-weight:bold;">{val:.1f}</span>'
        return f"{val:.1f}"
    
    def color_price(val, col_name, row):
        if pd.isna(val):
            return val
        price = f"${val:.2f}"
        # Highlight if price broke prev day high/low
        if col_name == "Prev Day High" and row.get("Last Price", 0) > val:
            return f'<span style="color:#238636;font-weight:bold;">{price}</span>'
        if col_name == "Prev Day Low" and row.get("Last Price", 0) < val:
            return f'<span style="color:#da3633;font-weight:bold;">{price}</span>'
        return price
    
    # Apply formatting
    for col in ["Status", "MACD Signal", "SMA (13/34)"]:
        df_html[col] = df_html[col].apply(color_status)
    
    df_html["RSI (14)"] = df_html["RSI (14)"].apply(color_rsi)
    
    for col in ["Last Price", "MA High", "MA Low", "Prev Day High", "Prev Day Low"]:
        if col in df_html.columns:
            df_html[col] = df_html.apply(lambda row: color_price(row[col], col, row), axis=1)
    
    # Build HTML
    html = """
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Screener Results</title>
        <style>
            body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif; 
                   max-width: 1200px; margin: 40px auto; padding: 0 20px; background: #0d1117; color: #c9d1d9; }
            h1 { color: #f0f6fc; border-bottom: 1px solid #30363d; padding-bottom: 10px; }
            table { width: 100%; border-collapse: collapse; margin-top: 20px; font-size: 14px; }
            th { background: #161b22; border: 1px solid #30363d; padding: 10px; text-align: left; 
                 color: #f0f6fc; font-weight: 600; position: sticky; top: 0; }
            td { border: 1px solid #21262d; padding: 8px 10px; }
            tr:nth-child(even) { background: #161b22; }
            tr:hover { background: #1c2128; }
            .timestamp { color: #8b949e; font-size: 12px; margin-top: 20px; }
        </style>
    </head>
    <body>
        <h1>📊 Screener Results</h1>
    """ + df_html.to_html(index=False, escape=False) + """
        <p class="timestamp">Generated: """ + pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S") + """</p>
    </body>
    </html>
    """
    return html


# ==================== STREAMLIT UI ====================

st.title("📊 Multi-Timeframe MA55, RSI, MACD & SMA Screener")
st.caption("Dynamic market screening with GitHub-compatible export formats.")

tickers = load_tickers(TICKER_FILE)

with st.sidebar:
    st.header("Screener Controls")
    selected_tf = st.selectbox("⏱ Select Timeframe", options=list(TIMEFRAME_CONFIG.keys()), index=0)
    st.write(f"📁 Tickers: **{len(tickers)}**")
    st.write(f"📈 MA Channel: **{MA_PERIOD} ({selected_tf})**")
    st.write(f"📊 SMA Cross: **13 / 34 ({selected_tf})**")
    
    st.markdown("---")
    st.subheader("RSI Thresholds")
    rsi_oversold = st.slider("Oversold (Green)", 10, 45, 30, 1)
    rsi_overbought = st.slider("Overbought (Red)", 55, 90, 70, 1)
    
    st.markdown("---")
    if st.button("🔄 Force Refresh", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

with st.spinner(f"Screening {len(tickers)} symbols on {selected_tf}..."):
    df_results = run_screener(tickers, selected_tf)

time_unit = TIMEFRAME_CONFIG[selected_tf]["unit"]

if not df_results.empty:
    priority_map = {"Bull": 0, "Bear": 1, "Touch Bull": 2, "Touch Bear": 3}
    df_results["Priority"] = df_results["Status"].map(priority_map)
    df_results = df_results.sort_values(["Priority", "Candles Ago"]).drop(columns="Priority")
    
    # Metrics
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Signals", len(df_results))
    c2.metric("Bull Breakouts", len(df_results[df_results["Status"] == "Bull"]))
    c3.metric("Bear Breakouts", len(df_results[df_results["Status"] == "Bear"]))
    c4.metric(f"Oversold (≤{rsi_oversold})", len(df_results[df_results["RSI (14)"] <= rsi_oversold]))
    c5.metric(f"Overbought (≥{rsi_overbought})", len(df_results[df_results["RSI (14)"] >= rsi_overbought]))
    
    st.markdown("---")
    
    # TABS: Streamlit View | GitHub Markdown | GitHub HTML
    tab1, tab2, tab3 = st.tabs(["📊 Streamlit View", "🐙 GitHub Markdown", "🌐 GitHub Pages HTML"])
    
    df_recent = df_results[df_results["Candles Ago"] <= 3]
    df_older = df_results[df_results["Candles Ago"] > 3]
    
    col_config = {
        "Ticker": st.column_config.TextColumn("Ticker", width="small"),
        "Status": st.column_config.TextColumn("Signal"),
        "Candles Ago": st.column_config.NumberColumn(f"Bars ({time_unit})", width="small"),
        "Last Price": st.column_config.NumberColumn("Price", format="$%.2f"),
        "MA High": st.column_config.NumberColumn("MA55 High", format="$%.2f"),
        "MA Low": st.column_config.NumberColumn("MA55 Low", format="$%.2f"),
        "RSI (14)": st.column_config.NumberColumn("RSI", format="%.1f"),
        "MACD Signal": st.column_config.TextColumn("MACD"),
        "MACD Cross Ago": st.column_config.NumberColumn("MACD Bars", width="small"),
        "SMA (13/34)": st.column_config.TextColumn("SMA Cross"),
        "SMA Cross Ago": st.column_config.NumberColumn("SMA Bars", width="small"),
        "Prev Day High": st.column_config.NumberColumn("PD High", format="$%.2f"),
        "Prev Day Low": st.column_config.NumberColumn("PD Low", format="$%.2f"),
    }
    
    with tab1:
        # Native Streamlit view with clean formatting
        st.subheader(f"🔥 Active (≤3 {time_unit})")
        if not df_recent.empty:
            st.dataframe(df_recent, use_container_width=True, hide_index=True, column_config=col_config)
        else:
            st.info("No active signals.")
        
        st.subheader(f"📋 Earlier (>3 {time_unit})")
        if not df_older.empty:
            st.dataframe(df_older, use_container_width=True, hide_index=True, column_config=col_config)
        else:
            st.caption("No older signals.")
    
    with tab2:
        st.info("Copy this markdown into GitHub READMEs, Issues, or PRs. Emojis render natively on GitHub.")
        
        md_recent = df_to_github_markdown(df_recent, rsi_oversold, rsi_overbought)
        md_older = df_to_github_markdown(df_older, rsi_oversold, rsi_overbought)
        
        if not df_recent.empty:
            st.subheader(f"🔥 Active Signals (≤3 {time_unit})")
            st.code(md_recent, language="markdown")
            st.download_button("📥 Download Active .md", md_recent, "active_signals.md", "text/markdown")
        
        if not df_older.empty:
            st.subheader(f"📋 Earlier Signals (>3 {time_unit})")
            st.code(md_older, language="markdown")
            st.download_button("📥 Download Earlier .md", md_older, "earlier_signals.md", "text/markdown")
        
        # Full combined
        full_md = f"## 📊 Screener Results — {selected_tf}\n\n### 🔥 Active Signals\n\n{md_recent}\n\n### 📋 Earlier Signals\n\n{md_older}\n\n_Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}_"
        st.markdown("---")
        st.subheader("📦 Combined Full Report")
        st.code(full_md, language="markdown")
        st.download_button("📥 Download Full Report .md", full_md, "screener_report.md", "text/markdown")
    
    with tab3:
        st.info("Self-contained HTML for GitHub Pages. Colors work via inline styles.")
        html_content = df_to_github_html(df_results, rsi_oversold, rsi_overbought)
        st.code(html_content[:2000] + "\n... [truncated for preview] ...", language="html")
        st.download_button("📥 Download index.html", html_content, "index.html", "text/html")

else:
    st.info(f"No signals detected across {len(tickers)} symbols on {selected_tf}.")
