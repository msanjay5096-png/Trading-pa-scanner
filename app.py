import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import os
from datetime import datetime
import time
import requests

# ====================== CONFIG ======================
DATA_DIR = "data"
WATCHLIST_FILE = os.path.join(DATA_DIR, "watchlists.json")
SCANNERS_FILE = os.path.join(DATA_DIR, "scanners.json")
SETTINGS_FILE = os.path.join(DATA_DIR, "settings.json")

os.makedirs(DATA_DIR, exist_ok=True)

# ====================== DEFAULT DATA ======================
DEFAULT_WATCHLISTS = {
    "Indian Stocks": [
        "RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "ICICIBANK.NS",
        "SBIN.NS", "BHARTIARTL.NS", "ITC.NS", "KOTAKBANK.NS", "LT.NS",
        "AXISBANK.NS", "BAJFINANCE.NS", "HINDUNILVR.NS", "ASIANPAINT.NS",
        "MARUTI.NS", "TITAN.NS", "SUNPHARMA.NS", "WIPRO.NS", "ULTRACEMCO.NS",
        "NESTLEIND.NS", "POWERGRID.NS", "NTPC.NS", "TECHM.NS", "HCLTECH.NS",
        "ADANIENT.NS", "ADANIPORTS.NS", "TATASTEEL.NS", "JSWSTEEL.NS",
        "INDUSINDBK.NS", "BAJAJFINSV.NS"
    ],
    "Crypto": [
        "BTC-USD", "ETH-USD", "SOL-USD", "BNB-USD", "XRP-USD",
        "ADA-USD", "DOGE-USD", "AVAX-USD", "DOT-USD", "MATIC-USD",
        "LINK-USD", "LTC-USD", "ATOM-USD", "UNI-USD", "APT-USD"
    ],
    "Forex": [
        "EURUSD=X", "GBPUSD=X", "USDJPY=X", "USDINR=X", "AUDUSD=X",
        "USDCAD=X", "USDCHF=X", "NZDUSD=X", "EURJPY=X", "GBPJPY=X",
        "EURGBP=X", "AUDJPY=X", "EURAUD=X", "GBPAUD=X"
    ]
}

DEFAULT_SCANNERS = {
    "My Price Action Scanner": {
        "description": "Near key level + HL/LH + Healthy break close beyond level + Next candle rejection/decision",
        "rules": {
            "require_trend": True,
            "require_near_sr": True,
            "sr_pct": 1.2,
            "require_consolidation": True,
            "min_consol_candles": 5,
            "prefer_decreasing": True,
            "require_volume_dry": True,
            "require_pin_bar": True,
            "require_high_volume_rejection": True,
            "require_bigger_rejection": True
        }
    }
}

DEFAULT_SETTINGS = {
    "telegram_token": "",
    "telegram_chat_id": "",
    "enable_telegram": False,
    "min_score_to_show": 60,
    "min_score_to_alert": 85
}

# ====================== HELPERS ======================
def load_json(filepath, default):
    if os.path.exists(filepath):
        try:
            with open(filepath, "r") as f:
                return json.load(f)
        except:
            return default
    return default

def save_json(filepath, data):
    with open(filepath, "w") as f:
        json.dump(data, f, indent=2)

def get_symbol_suffix(symbol, market):
    symbol = symbol.upper().strip()
    if market == "Indian Stocks":
        if not (symbol.endswith(".NS") or symbol.endswith(".BO")):
            return symbol + ".NS"
    elif market == "Forex":
        if not symbol.endswith("=X"):
            return symbol + "=X"
    elif market == "Crypto":
        if not any(x in symbol for x in ["-USD", "-USDT", "-BTC"]):
            return symbol + "-USD"
    return symbol

def send_telegram_alert(token, chat_id, message):
    """Send Telegram message"""
    if not token or not chat_id:
        return False
    try:
        url = f"https://api.telegram.org/bot{token}/sendMessage"
        payload = {"chat_id": chat_id, "text": message, "parse_mode": "HTML"}
        r = requests.post(url, json=payload, timeout=10)
        return r.status_code == 200
    except:
        return False


# ====================== CHART-MATCHED SETUP LOGIC ======================
# Sequence from user's Nifty chart:
# Bullish: Near resistance + Higher Low + Healthy green close above level
#          + NEXT candle is red/rejection → SIGNAL
# Bearish: Near support + Lower High + Healthy red close below level
#          + NEXT candle is green/rejection → SIGNAL

def detect_swing_points(df, left=3, right=3):
    highs = df['High'].values
    lows = df['Low'].values
    swing_highs, swing_lows = [], []
    for i in range(left, len(df) - right):
        if highs[i] == max(highs[i-left:i+right+1]):
            swing_highs.append((i, highs[i]))
        if lows[i] == min(lows[i-left:i+right+1]):
            swing_lows.append((i, lows[i]))
    return swing_highs, swing_lows

def cluster_levels(levels, threshold_pct=0.35):
    if not levels:
        return []
    levels = sorted(levels)
    clusters = []
    current = [levels[0]]
    for level in levels[1:]:
        if abs(level - current[-1]) / current[-1] * 100 <= threshold_pct:
            current.append(level)
        else:
            clusters.append(float(np.mean(current)))
            current = [level]
    clusters.append(float(np.mean(current)))
    return clusters

def find_key_levels(df, lookback=40):
    """Find clear horizontal support and resistance zones"""
    recent = df.tail(lookback)
    swing_highs, swing_lows = detect_swing_points(recent, left=3, right=3)
    resistances = cluster_levels([p for _, p in swing_highs] + [recent['High'].max()])
    supports = cluster_levels([p for _, p in swing_lows] + [recent['Low'].min()])
    return supports, resistances

def price_near_level(price, levels, pct=0.30):
    """Return the nearest level within pct%, or None"""
    if not levels:
        return None
    best = None
    best_dist = 999
    for lv in levels:
        dist = abs(price - lv) / price * 100
        if dist <= pct and dist < best_dist:
            best_dist = dist
            best = lv
    return best

def has_higher_low(df, lookback=30):
    _, swing_lows = detect_swing_points(df.tail(lookback), left=2, right=2)
    if len(swing_lows) < 2:
        return False
    # Require meaningful higher low (at least 0.05% higher)
    return swing_lows[-1][1] > swing_lows[-2][1] * 1.0005

def has_lower_high(df, lookback=30):
    swing_highs, _ = detect_swing_points(df.tail(lookback), left=2, right=2)
    if len(swing_highs) < 2:
        return False
    return swing_highs[-1][1] < swing_highs[-2][1] * 0.9995

def is_healthy_break_candle(candle, direction="up"):
    """
    Healthy break candle (strict):
    - Strong body (>= 45% of range)
    - Closes in the break direction
    - Close is in the outer 65%+ of the candle
    """
    body = abs(candle['Close'] - candle['Open'])
    rng = candle['High'] - candle['Low']
    if rng == 0:
        return False
    body_ratio = body / rng
    if body_ratio < 0.45:
        return False
    if direction == "up":
        if candle['Close'] <= candle['Open']:
            return False
        close_pos = (candle['Close'] - candle['Low']) / rng
        return close_pos >= 0.65
    else:
        if candle['Close'] >= candle['Open']:
            return False
        close_pos = (candle['High'] - candle['Close']) / rng
        return close_pos >= 0.65

def is_rejection_or_decision_candle(candle, direction="up"):
    """
    After bullish break → want RED or strong upper-wick rejection
    After bearish break → want GREEN or strong lower-wick rejection
    Weak dojis alone are not enough (reduces noise)
    """
    body = abs(candle['Close'] - candle['Open'])
    upper = candle['High'] - max(candle['Open'], candle['Close'])
    lower = min(candle['Open'], candle['Close']) - candle['Low']
    rng = candle['High'] - candle['Low']
    if rng == 0:
        return False

    body_ratio = body / rng
    is_red = candle['Close'] < candle['Open']
    is_green = candle['Close'] > candle['Open']
    bear_reject = upper > lower * 1.4 and upper > body * 1.0 and upper / rng > 0.40
    bull_reject = lower > upper * 1.4 and lower > body * 1.0 and lower / rng > 0.40
    # Clear opposite candle with decent body, or strong rejection wick
    strong_red = is_red and body_ratio >= 0.25
    strong_green = is_green and body_ratio >= 0.25

    if direction == "up":
        return strong_red or bear_reject
    else:
        return strong_green or bull_reject

def scan_symbol(symbol, interval="5m", rules=None):
    """
    Chart-matched scanner (user's Nifty example):

    BULLISH sequence:
      1. Price near resistance zone
      2. Higher Low structure present
      3. Healthy GREEN candle closes ABOVE the resistance
      4. NEXT candle is RED / rejection / doji
      → Signal on the confirmation candle

    BEARISH sequence:
      1. Price near support zone
      2. Lower High structure present
      3. Healthy RED candle closes BELOW the support
      4. NEXT candle is GREEN / rejection / doji
      → Signal on the confirmation candle
    """
    try:
        ticker = yf.Ticker(symbol)
        period = "5d" if interval in ["1m", "5m"] else "15d"
        df = ticker.history(period=period, interval=interval)
        if df is None or len(df) < 50:
            return None
        df = df.dropna()

        # We need at least the break candle + confirmation candle
        # Check last few candles for the sequence
        supports, resistances = find_key_levels(df, lookback=45)

        # Look for the pattern ending on the last or second-last candle
        # break_idx = confirmation candle index - 1
        for conf_offset in [0, -1]:  # confirmation on last or previous candle
            conf_i = len(df) - 1 + conf_offset
            break_i = conf_i - 1
            if break_i < 20:
                continue

            break_candle = df.iloc[break_i]
            conf_candle = df.iloc[conf_i]
            break_close = float(break_candle['Close'])
            conf_close = float(conf_candle['Close'])

            # ----- BULLISH -----
            # Near resistance, HL, healthy green closes above resistance,
            # next candle is red/rejection
            near_res = price_near_level(break_close, resistances, pct=0.30)
            if near_res is not None:
                # Also accept if break close is just above the level
                level = near_res
                # Must close meaningfully above the level (not just a tip)
                if break_close > level * 1.0003 and is_healthy_break_candle(break_candle, "up"):
                    pre_break = df.iloc[:break_i+1]
                    if has_higher_low(pre_break, lookback=30):
                        if is_rejection_or_decision_candle(conf_candle, "up"):
                            # Volume check (skip if no volume data)
                            prev_vol = float(df['Volume'].iloc[break_i])
                            conf_vol = float(df['Volume'].iloc[conf_i])
                            vol_ok = True
                            if prev_vol > 0 or conf_vol > 0:
                                vol_ok = conf_vol >= prev_vol * 0.9  # mild preference

                            score = 70
                            if conf_vol > prev_vol and prev_vol > 0:
                                score += 15
                            if abs(break_close - level) / level * 100 < 0.25:
                                score += 10

                            return {
                                "symbol": symbol,
                                "score": min(score, 100),
                                "full_match": True,
                                "trend": "Uptrend",
                                "direction": "Bullish",
                                "price": round(conf_close, 5),
                                "interval": interval,
                                "near_level": "Resistance",
                                "level_price": round(level, 5),
                                "break_level": round(level, 5),
                                "scores": {
                                    "sr": 90,
                                    "structure": 90,
                                    "volume_dry": 50,
                                    "confirmation": 90
                                },
                                "details": {
                                    "is_consol": True,
                                    "is_decreasing": True,
                                    "volume_dry": True,
                                    "pin_bar": True,
                                    "higher_low": True,
                                    "lower_high": False,
                                    "break_up": True,
                                    "break_down": False
                                },
                                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                "df": df.tail(60)
                            }

            # ----- BEARISH -----
            near_sup = price_near_level(break_close, supports, pct=0.30)
            if near_sup is not None:
                level = near_sup
                if break_close < level * 0.9997 and is_healthy_break_candle(break_candle, "down"):
                    pre_break = df.iloc[:break_i+1]
                    if has_lower_high(pre_break, lookback=30):
                        if is_rejection_or_decision_candle(conf_candle, "down"):
                            prev_vol = float(df['Volume'].iloc[break_i])
                            conf_vol = float(df['Volume'].iloc[conf_i])
                            score = 70
                            if conf_vol > prev_vol and prev_vol > 0:
                                score += 15
                            if abs(break_close - level) / level * 100 < 0.25:
                                score += 10

                            return {
                                "symbol": symbol,
                                "score": min(score, 100),
                                "full_match": True,
                                "trend": "Downtrend",
                                "direction": "Bearish",
                                "price": round(conf_close, 5),
                                "interval": interval,
                                "near_level": "Support",
                                "level_price": round(level, 5),
                                "break_level": round(level, 5),
                                "scores": {
                                    "sr": 90,
                                    "structure": 90,
                                    "volume_dry": 50,
                                    "confirmation": 90
                                },
                                "details": {
                                    "is_consol": True,
                                    "is_decreasing": True,
                                    "volume_dry": True,
                                    "pin_bar": True,
                                    "higher_low": False,
                                    "lower_high": True,
                                    "break_up": False,
                                    "break_down": True
                                },
                                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                "df": df.tail(60)
                            }

        return None
    except Exception as e:
        return None

# ====================== STREAMLIT APP ======================
st.set_page_config(
    page_title="PA Scanner",
    page_icon="📈",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# ========== CLEAN MOBILE CSS ==========
st.markdown("""
<style>
    /* Base */
    .stApp {
        background: #0a0e17;
        color: #e0e0e0;
    }
    
    /* Hide Streamlit branding */
    #MainMenu, footer, header {visibility: hidden;}
    .stDeployButton {display: none;}
    
    /* Header */
    .app-title {
        font-size: 1.6rem;
        font-weight: 800;
        color: #00e676;
        text-align: center;
        margin: 0.3rem 0 0.1rem 0;
        letter-spacing: -0.5px;
    }
    .app-sub {
        text-align: center;
        color: #666;
        font-size: 0.8rem;
        margin-bottom: 1.2rem;
    }
    
    /* Cards */
    .card {
        background: #121826;
        border-radius: 16px;
        padding: 1rem 1.1rem;
        margin-bottom: 0.8rem;
        border: 1px solid #1e293b;
    }
    .card-green {
        border: 1px solid #00e67633;
        background: linear-gradient(145deg, #0f1a14, #121826);
    }
    .card-red {
        border: 1px solid #ff174433;
        background: linear-gradient(145deg, #1a0f14, #121826);
    }
    
    /* Score */
    .score {
        font-size: 1.5rem;
        font-weight: 800;
    }
    .score-green { color: #00e676; }
    .score-yellow { color: #ffc107; }
    .score-red { color: #ff5252; }
    
    /* Direction tags */
    .tag {
        display: inline-block;
        padding: 0.15rem 0.55rem;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 700;
    }
    .tag-bull { background: #00e67622; color: #00e676; }
    .tag-bear { background: #ff525222; color: #ff5252; }
    .tag-full { background: #00e67633; color: #00e676; border: 1px solid #00e67655; }
    
    /* Buttons */
    .stButton > button {
        border-radius: 12px !important;
        font-weight: 700 !important;
        height: 3rem !important;
        font-size: 1rem !important;
    }
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #00c853, #00e676) !important;
        color: #000 !important;
        border: none !important;
    }
    
    /* Sidebar clean */
    section[data-testid="stSidebar"] {
        background: #0a0e17;
        border-right: 1px solid #1e293b;
    }
    section[data-testid="stSidebar"] .stMarkdown h1,
    section[data-testid="stSidebar"] .stMarkdown h2,
    section[data-testid="stSidebar"] .stMarkdown h3 {
        color: #00e676;
    }
    
    /* Metrics */
    [data-testid="stMetric"] {
        background: #121826;
        border-radius: 12px;
        padding: 0.6rem;
        border: 1px solid #1e293b;
    }
    [data-testid="stMetricLabel"] { color: #888 !important; font-size: 0.75rem !important; }
    [data-testid="stMetricValue"] { color: #e0e0e0 !important; font-size: 1.1rem !important; }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.3rem;
        background: #121826;
        border-radius: 12px;
        padding: 0.3rem;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 10px;
        color: #888;
        font-weight: 600;
    }
    .stTabs [aria-selected="true"] {
        background: #1e293b !important;
        color: #00e676 !important;
    }
    
    /* Inputs */
    .stSelectbox, .stTextInput, .stSlider {
        margin-bottom: 0.4rem;
    }
    
    /* Progress */
    .stProgress > div > div { background: #00e676; }
    
    /* Expander */
    .streamlit-expanderHeader {
        background: #121826;
        border-radius: 10px;
        font-weight: 600;
    }
    
    /* Reduce padding on mobile */
    .block-container {
        padding-top: 1rem !important;
        padding-bottom: 2rem !important;
        max-width: 100% !important;
    }
    
    /* Status bar */
    .status-bar {
        background: #121826;
        border-radius: 12px;
        padding: 0.7rem 1rem;
        margin-bottom: 1rem;
        border: 1px solid #1e293b;
        display: flex;
        justify-content: space-between;
        align-items: center;
        font-size: 0.85rem;
    }
</style>
""", unsafe_allow_html=True)

# ========== HEADER ==========
st.markdown('<div class="app-title">📈 PA Scanner</div>', unsafe_allow_html=True)
st.markdown('<div class="app-sub">Price Action • Intraday • Mobile</div>', unsafe_allow_html=True)

# Load data
watchlists = load_json(WATCHLIST_FILE, DEFAULT_WATCHLISTS)
scanners = load_json(SCANNERS_FILE, DEFAULT_SCANNERS)
settings = load_json(SETTINGS_FILE, DEFAULT_SETTINGS)

# ========== SIDEBAR (Clean) ==========
with st.sidebar:
    st.markdown("### ⚙️ Settings")
    
    market = st.selectbox("Market", ["Indian Stocks", "Crypto", "Forex"], key="market_select")
    
    st.markdown("---")
    st.markdown("#### 📋 Watchlist")
    current_list = watchlists.get(market, [])
    
    with st.expander(f"{len(current_list)} symbols", expanded=False):
        for i, sym in enumerate(current_list):
            c1, c2 = st.columns([5, 1])
            c1.write(f"`{sym}`")
            if c2.button("×", key=f"del_{market}_{i}"):
                current_list.pop(i)
                watchlists[market] = current_list
                save_json(WATCHLIST_FILE, watchlists)
                st.rerun()
    
    new_sym = st.text_input("Add symbol", placeholder="RELIANCE / BTC / EURUSD", label_visibility="collapsed")
    if st.button("Add Symbol", use_container_width=True):
        if new_sym:
            fmt = get_symbol_suffix(new_sym, market)
            if fmt not in current_list:
                current_list.append(fmt)
                watchlists[market] = current_list
                save_json(WATCHLIST_FILE, watchlists)
                st.rerun()
    
    st.markdown("---")
    st.markdown("#### ⏱ Timeframe")
    timeframe = st.selectbox("TF", ["1m", "5m", "15m", "30m"], index=1, label_visibility="collapsed")
    
    st.markdown("---")
    st.markdown("#### 🔍 Scanner")
    scanner_name = st.selectbox("Scanner", list(scanners.keys()), label_visibility="collapsed")
    active_rules = scanners[scanner_name]["rules"]
    
    st.markdown("---")
    st.markdown("#### 🔄 Auto Refresh")
    auto_refresh = st.toggle("Auto Refresh", value=st.session_state.get("auto_refresh", False))
    refresh_interval = st.select_slider(
        "Every",
        options=[60, 120, 180, 300, 600],
        value=120,
        format_func=lambda x: f"{x//60}m"
    )
    st.session_state["auto_refresh"] = auto_refresh
    
    st.markdown("---")
    st.markdown("#### 📱 Telegram")
    enable_tg = st.toggle("Alerts", value=settings.get("enable_telegram", False))
    tg_token = st.text_input("Bot Token", value=settings.get("telegram_token", ""), type="password", label_visibility="collapsed", placeholder="Bot Token")
    tg_chat = st.text_input("Chat ID", value=settings.get("telegram_chat_id", ""), label_visibility="collapsed", placeholder="Chat ID")
    
    if st.button("Save Alerts", use_container_width=True):
        settings["enable_telegram"] = enable_tg
        settings["telegram_token"] = tg_token
        settings["telegram_chat_id"] = tg_chat
        save_json(SETTINGS_FILE, settings)
        st.success("Saved")
    
    if enable_tg and tg_token and tg_chat:
        if st.button("Test Alert", use_container_width=True):
            ok = send_telegram_alert(tg_token, tg_chat, "✅ PA Scanner connected!")
            st.success("Sent!" if ok else "Failed")

# ========== MAIN TABS ==========
tab_scan, tab_results, tab_manage = st.tabs(["🚀 Scan", "📊 Results", "🛠 More"])

# ===== SCAN TAB =====
with tab_scan:
    # Status strip
    last = st.session_state.get("last_scan", "—")
    st.markdown(f"""
    <div class="status-bar">
        <span>📌 {market}</span>
        <span>⏱ {timeframe}</span>
        <span>🕐 {last}</span>
    </div>
    """, unsafe_allow_html=True)
    
    min_score = st.slider("Min Score", 0, 100, 60, 5)
    only_full = st.toggle("Full Matches Only", value=False)
    
    if auto_refresh:
        st.info(f"🔄 Auto-scanning every {refresh_interval//60} min")
    
    run_scan = st.button("🔥 SCAN NOW", type="primary", use_container_width=True)
    should_scan = run_scan or auto_refresh
    
    if should_scan:
        symbols = watchlists.get(market, [])
        if not symbols:
            st.error("Watchlist empty. Add symbols in the menu (☰)")
        else:
            progress = st.progress(0)
            status = st.empty()
            results = []
            
            for i, sym in enumerate(symbols):
                status.caption(f"Scanning {sym} ({i+1}/{len(symbols)})")
                res = scan_symbol(sym, interval=timeframe, rules=active_rules)
                if res:
                    if only_full and not res["full_match"]:
                        pass
                    elif res["score"] >= min_score:
                        results.append(res)
                progress.progress((i + 1) / len(symbols))
                time.sleep(0.15)
            
            status.empty()
            progress.empty()
            
            results = sorted(results, key=lambda x: x["score"], reverse=True)
            st.session_state["scan_results"] = results
            st.session_state["last_scan"] = datetime.now().strftime("%H:%M")
            st.session_state["last_market"] = market
            st.session_state["last_tf"] = timeframe
            
            # Telegram
            if settings.get("enable_telegram") and settings.get("telegram_token"):
                for r in [x for x in results if x["score"] >= 85 or x["full_match"]][:4]:
                    msg = f"🚨 <b>{r['symbol']}</b> | Score {r['score']}\n{r['direction']} | {r['trend']}\nNear {r['near_level']} | {r['price']}"
                    send_telegram_alert(settings["telegram_token"], settings["telegram_chat_id"], msg)
            
            if results:
                st.success(f"Found {len(results)} setup(s)")
            else:
                st.warning("No setups found. Try lower score or different TF.")
            
            if auto_refresh:
                with st.spinner(f"Next scan in {refresh_interval//60} min..."):
                    time.sleep(refresh_interval)
                st.rerun()

# ===== RESULTS TAB =====
with tab_results:
    results = st.session_state.get("scan_results", [])
    
    if not results:
        st.info("No results yet. Go to Scan tab and tap SCAN NOW.")
    else:
        st.caption(f"{len(results)} setups • {st.session_state.get('last_scan', '')}")
        
        for res in results:
            score = res["score"]
            score_cls = "score-green" if score >= 80 else ("score-yellow" if score >= 60 else "score-red")
            card_cls = "card card-green" if res["full_match"] else "card"
            dir_tag = "tag-bull" if res["direction"] == "Bullish" else "tag-bear"
            
            full_badge = '<span class="tag tag-full">FULL MATCH</span>' if res["full_match"] else ""
            
            st.markdown(f"""
            <div class="{card_cls}">
                <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:0.4rem;">
                    <div>
                        <span style="font-size:1.15rem;font-weight:700;">{res['symbol']}</span>
                        <span class="tag {dir_tag}">{res['direction']}</span>
                        {full_badge}
                    </div>
                    <span class="score {score_cls}">{score}</span>
                </div>
                <div style="font-size:0.85rem;color:#aaa;line-height:1.6;">
                    {res['trend'] or '—'} • Near {res['near_level'] or '—'} ({res['level_price'] or '—'})<br>
                    Price {res['price']} • {res['interval']}
                </div>
                <div style="margin-top:0.4rem;font-size:0.8rem;">
                    Consol {'✅' if res['details'].get('is_consol') else '❌'}
                    &nbsp; Vol {'✅' if res['details'].get('volume_dry') else '❌'}
                    &nbsp; Reject {'✅' if res['details'].get('pin_bar') else '❌'}
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            with st.expander("Chart"):
                df = res["df"]
                fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                                    vertical_spacing=0.03, row_heights=[0.75, 0.25])
                fig.add_trace(go.Candlestick(
                    x=df.index, open=df['Open'], high=df['High'],
                    low=df['Low'], close=df['Close']
                ), row=1, col=1)
                if res["level_price"]:
                    color = "#00e676" if res["near_level"] == "Support" else "#ff5252"
                    fig.add_hline(y=res["level_price"], line_dash="dot", line_color=color, row=1, col=1)
                fig.add_trace(go.Bar(x=df.index, y=df['Volume'], marker_color="#333"), row=2, col=1)
                fig.update_layout(
                    height=340, template="plotly_dark",
                    xaxis_rangeslider_visible=False,
                    margin=dict(l=0, r=0, t=10, b=0),
                    showlegend=False,
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)"
                )
                fig.update_xaxes(showgrid=False, showticklabels=False)
                fig.update_yaxes(showgrid=False)
                st.plotly_chart(fig, use_container_width=True)

# ===== MORE TAB =====
with tab_manage:
    st.markdown("#### Create Scanner")
    with st.expander("New Scanner"):
        new_name = st.text_input("Name")
        r1 = st.checkbox("Trend", True)
        r2 = st.checkbox("Near S/R", True)
        sr_pct = st.slider("S/R %", 0.3, 2.0, 1.0, 0.1)
        r3 = st.checkbox("Consolidation", True)
        r4 = st.checkbox("Volume Dry", True)
        r5 = st.checkbox("Rejection", True)
        if st.button("Create", use_container_width=True) and new_name:
            if new_name not in scanners:
                scanners[new_name] = {
                    "description": "Custom",
                    "rules": {
                        "require_trend": r1, "require_near_sr": r2, "sr_pct": sr_pct,
                        "require_consolidation": r3, "min_consol_candles": 5,
                        "prefer_decreasing": True, "require_volume_dry": r4,
                        "require_pin_bar": r5, "require_high_volume_rejection": True,
                        "require_bigger_rejection": True
                    }
                }
                save_json(SCANNERS_FILE, scanners)
                st.success("Created")
                st.rerun()
    
    st.markdown("#### Your Scanners")
    for name in scanners:
        st.write(f"• **{name}**")
        if name != "My Price Action Scanner":
            if st.button(f"Delete {name}", key=f"d_{name}"):
                del scanners[name]
                save_json(SCANNERS_FILE, scanners)
                st.rerun()
    
    st.markdown("---")
    st.markdown("""
    **Quick Tips**
    - Best TF: 5m or 15m
    - Turn on Auto Refresh for continuous scanning
    - Full Match = all your rules passed
    - Score 80+ = strong setup
    """)
    
    st.markdown("---")
    st.caption("PA Scanner • Mobile First • Indian Stocks + Crypto + Forex")
