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
from datetime import timezone, timedelta

IST = timezone(timedelta(hours=5, minutes=30))

def to_ist_str(ts):
    """Format timestamp in India Standard Time (+05:30)."""
    try:
        if ts is None:
            return "—"
        if hasattr(ts, "to_pydatetime"):
            ts = ts.to_pydatetime()
        if getattr(ts, "tzinfo", None) is not None:
            return ts.astimezone(IST).strftime("%d-%b-%Y %H:%M IST")
        return (ts.replace(tzinfo=timezone.utc).astimezone(IST)).strftime("%d-%b-%Y %H:%M IST")
    except Exception:
        return str(ts)[:16]


# ====================== CONFIG ======================
DATA_DIR = "data"
WATCHLIST_FILE = os.path.join(DATA_DIR, "watchlists.json")
SCANNERS_FILE = os.path.join(DATA_DIR, "scanners.json")
SETTINGS_FILE = os.path.join(DATA_DIR, "settings.json")
HISTORY_FILE = os.path.join(DATA_DIR, "scan_history.json")

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
    if symbol.startswith("^"):
        return symbol
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
    if not token or not chat_id:
        return False
    try:
        url = f"https://api.telegram.org/bot{token}/sendMessage"
        payload = {"chat_id": chat_id, "text": message, "parse_mode": "HTML"}
        r = requests.post(url, json=payload, timeout=10)
        return r.status_code == 200
    except:
        return False

# ====================== SCANNER CORE LOGIC ======================
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
    recent = df.tail(lookback)
    swing_highs, swing_lows = detect_swing_points(recent, left=3, right=3)
    resistances = cluster_levels([p for _, p in swing_highs] + [recent['High'].max()])
    supports = cluster_levels([p for _, p in swing_lows] + [recent['Low'].min()])
    return supports, resistances

def price_near_level(price, levels, pct=0.30):
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
    return swing_lows[-1][1] > swing_lows[-2][1] * 1.0005

def has_lower_high(df, lookback=30):
    swing_highs, _ = detect_swing_points(df.tail(lookback), left=2, right=2)
    if len(swing_highs) < 2:
        return False
    return swing_highs[-1][1] < swing_highs[-2][1] * 0.9995

def is_healthy_break_candle(candle, direction="up"):
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
    strong_red = is_red and body_ratio >= 0.25
    strong_green = is_green and body_ratio >= 0.25

    if direction == "up":
        return strong_red or bear_reject
    else:
        return strong_green or bull_reject

def scan_symbol(symbol, interval="5m", rules=None):
    try:
        ticker = yf.Ticker(symbol)
        tf_map = {
            "1m": "1m", "3m": "2m", "5m": "5m",
            "15m": "15m", "1h": "1h", "4h": "1h", "1d": "1d", "30m": "30m",
        }
        yf_interval = tf_map.get(interval, interval)
        period = "5d" if interval in ["1m", "3m", "5m"] else ("60d" if interval in ["15m", "30m", "1h", "4h"] else "1y")
        df = ticker.history(period=period, interval=yf_interval)
        if df is None or len(df) < 50:
            return None
        df = df.dropna()

        supports, resistances = find_key_levels(df, lookback=45)

        for conf_offset in [0, -1]:
            conf_i = len(df) - 1 + conf_offset
            break_i = conf_i - 1
            if break_i < 20:
                continue

            break_candle = df.iloc[break_i]
            conf_candle = df.iloc[conf_i]
            break_close = float(break_candle['Close'])
            conf_close = float(conf_candle['Close'])

            # Bullish Pattern
            near_res = price_near_level(break_close, resistances, pct=0.30)
            if near_res is not None:
                level = near_res
                if break_close > level * 1.0003 and is_healthy_break_candle(break_candle, "up"):
                    pre_break = df.iloc[:break_i+1]
                    if has_higher_low(pre_break, lookback=30):
                        if is_rejection_or_decision_candle(conf_candle, "up"):
                            prev_vol = float(df['Volume'].iloc[break_i])
                            conf_vol = float(df['Volume'].iloc[conf_i])
                            if rules and rules.get("require_high_volume_rejection", True):
                                if prev_vol > 0 or conf_vol > 0:
                                    if not (conf_vol > prev_vol):
                                        continue

                            score = 70
                            if conf_vol > prev_vol and prev_vol > 0:
                                score += 15
                            if abs(break_close - level) / level * 100 < 0.25:
                                score += 10

                            try:
                                break_ts = df.index[break_i]
                                conf_ts = df.index[conf_i]
                            except Exception:
                                break_ts = conf_ts = None

                            seg = pre_break.tail(30)
                            sh, sl = detect_swing_points(seg, left=2, right=2)
                            trend_pts = []
                            if len(sl) >= 2:
                                trend_pts = [
                                    [str(seg.index[sl[-2][0]]), float(sl[-2][1])],
                                    [str(seg.index[sl[-1][0]]), float(sl[-1][1])],
                                ]
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
                                "break_time_ist": to_ist_str(break_ts),
                                "setup_time_ist": to_ist_str(conf_ts),
                                "scanned_at_ist": datetime.now(IST).strftime("%d-%b-%Y %H:%M IST"),
                                "trend_points": trend_pts,
                                "df": df.tail(60)
                            }

            # Bearish Pattern
            near_sup = price_near_level(break_close, supports, pct=0.30)
            if near_sup is not None:
                level = near_sup
                if break_close < level * 0.9997 and is_healthy_break_candle(break_candle, "down"):
                    pre_break = df.iloc[:break_i+1]
                    if has_lower_high(pre_break, lookback=30):
                        if is_rejection_or_decision_candle(conf_candle, "down"):
                            prev_vol = float(df['Volume'].iloc[break_i])
                            conf_vol = float(df['Volume'].iloc[conf_i])
                            if rules and rules.get("require_high_volume_rejection", True):
                                if prev_vol > 0 or conf_vol > 0:
                                    if not (conf_vol > prev_vol):
                                        continue

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
                                "break_time_ist": to_ist_str(df.index[break_i]),
                                "setup_time_ist": to_ist_str(df.index[conf_i]),
                                "scanned_at_ist": datetime.now(IST).strftime("%d-%b-%Y %H:%M IST"),
                                "df": df.tail(60)
                            }
        return None
    except Exception:
        return None


# ====================== STREAMLIT UI ======================
st.set_page_config(
    page_title="PA Scanner",
    page_icon="⚡",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Custom High-End Modern Styling matching the provided visual reference
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"], .stApp, button, input, label {
  font-family: 'Plus Jakarta Sans', system-ui, sans-serif !important;
}

.stApp {
  background: #090c15 !important;
  color: #f0f4ff !important;
}

#MainMenu, footer, header, [data-testid="stSidebar"], .stDeployButton { 
  display: none !important; 
}

.block-container {
  padding-top: 1rem !important;
  padding-bottom: 5.5rem !important;
  padding-left: 0.8rem !important;
  padding-right: 0.8rem !important;
  max-width: 420px !important;
}

/* App Header Title */
.header-title {
  font-size: 1.6rem;
  font-weight: 800;
  background: linear-gradient(90deg, #00e5ff 0%, #29b6f6 100%);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  display: flex;
  align-items: center;
  gap: 8px;
}

/* Glass Panels */
.panel {
  background: rgba(18, 24, 40, 0.75);
  border: 1px solid rgba(0, 229, 255, 0.18);
  border-radius: 20px;
  padding: 1rem;
  margin-bottom: 0.85rem;
  box-shadow: 0 8px 24px rgba(0, 0, 0, 0.4);
  backdrop-filter: blur(12px);
}

.panel-title {
  font-size: 0.75rem;
  font-weight: 700;
  color: #78909c;
  text-transform: uppercase;
  letter-spacing: 0.1em;
  margin-bottom: 0.5rem;
}

/* Buttons */
.stButton > button {
  border-radius: 14px !important;
  font-weight: 700 !important;
  border: none !important;
  width: 100% !important;
}

.stButton > button[kind="primary"] {
  background: linear-gradient(135deg, #00e676 0%, #00c853 100%) !important;
  color: #03150a !important;
  box-shadow: 0 0 24px rgba(0, 230, 118, 0.4) !important;
  min-height: 3.6rem !important;
  font-size: 1.25rem !important;
  letter-spacing: 0.05em !important;
  border-radius: 22px !important;
}

/* Remove default progress bar double line styling */
.stProgress {
  height: 6px !important;
  margin-top: 0.4rem !important;
  margin-bottom: 0.8rem !important;
}

.stProgress > div {
  background: rgba(255, 255, 255, 0.08) !important;
  border-radius: 10px !important;
  border: none !important;
  height: 6px !important;
}

.stProgress > div > div {
  background: linear-gradient(90deg, #00e5ff, #00e676) !important;
  border-radius: 10px !important;
  height: 6px !important;
  box-shadow: 0 0 10px #00e676;
}

/* Horizontal Radio Market Pills */
div[role="radiogroup"] {
  display: flex !important;
  gap: 0.5rem !important;
  margin-bottom: 0.8rem !important;
}
div[role="radiogroup"] > label {
  flex: 1 !important;
  background: rgba(18, 24, 40, 0.8) !important;
  border: 1px solid rgba(0, 229, 255, 0.25) !important;
  border-radius: 16px !important;
  padding: 0.6rem 0.2rem !important;
  text-align: center !important;
  color: #b0bec5 !important;
  font-weight: 700 !important;
}
div[role="radiogroup"] label:has(input:checked) {
  background: rgba(0, 229, 255, 0.12) !important;
  border-color: #00e5ff !important;
  color: #00e5ff !important;
  box-shadow: 0 0 15px rgba(0, 229, 255, 0.25) !important;
}

/* Result Card */
.card {
  background: rgba(22, 30, 48, 0.9);
  border-radius: 16px;
  padding: 0.85rem 1rem;
  margin-bottom: 0.65rem;
  border: 1px solid rgba(0, 229, 255, 0.2);
}
.card-green { border-color: rgba(0, 230, 118, 0.5); box-shadow: 0 0 15px rgba(0, 230, 118, 0.12); }
.card-red { border-color: rgba(255, 82, 82, 0.45); box-shadow: 0 0 15px rgba(255, 82, 82, 0.1); }
.score { font-size: 1.3rem; font-weight: 800; }
.score-green { color: #00e676; }
.score-yellow { color: #ffea00; }
.score-red { color: #ff5252; }

/* Watchlist Cards Horizontal Grid */
.wl-grid {
  display: grid;
  grid-template-columns: repeat(2, 1fr);
  gap: 0.6rem;
}
.wl-card {
  background: rgba(16, 22, 36, 0.85);
  border: 1px solid rgba(0, 229, 255, 0.15);
  border-radius: 14px;
  padding: 0.6rem;
}
.wl-card .sym { font-weight: 800; font-size: 0.95rem; color: #ffffff; }
.wl-card .meta { font-size: 0.72rem; color: #78909c; }
.wl-card .price { font-weight: 700; font-size: 0.85rem; color: #eceff1; margin-top: 0.2rem; }
.wl-card .change-pos { font-size: 0.72rem; color: #00e676; font-weight: 700; }

/* Bottom Nav bar */
.bottom-nav {
  position: fixed; left: 0; right: 0; bottom: 0;
  background: rgba(10, 14, 24, 0.95);
  border-top: 1px solid rgba(0, 229, 255, 0.15);
  backdrop-filter: blur(20px);
  display: flex; justify-content: space-around; align-items: center;
  padding: 0.5rem 0.4rem 0.8rem 0.4rem;
  z-index: 9999;
}
.bottom-nav .item {
  text-align: center; color: #607d8b; font-size: 0.68rem; font-weight: 700;
}
.bottom-nav .item.active { color: #00e5ff; }
.bottom-nav .icon { font-size: 1.2rem; display: block; }
.bottom-nav .scan-fab {
  width: 52px; height: 52px; border-radius: 50%;
  background: linear-gradient(135deg, #00e676, #00c853);
  color: #03150a; font-size: 1.3rem;
  display: flex; align-items: center; justify-content: center;
  box-shadow: 0 0 20px rgba(0, 230, 118, 0.5);
  margin-top: -20px;
}
</style>
""", unsafe_allow_html=True)

# Watchlists & Settings
watchlists = load_json(WATCHLIST_FILE, DEFAULT_WATCHLISTS)
scanners = load_json(SCANNERS_FILE, DEFAULT_SCANNERS)
settings = load_json(SETTINGS_FILE, DEFAULT_SETTINGS)
history = load_json(HISTORY_FILE, [])

if "market" not in st.session_state:
    st.session_state["market"] = "Indian Stocks"
if "scan_results" not in st.session_state:
    st.session_state["scan_results"] = []

# --- Header Bar ---
col1, col2 = st.columns([4, 1])
with col1:
    st.markdown('<div class="header-title">☰ PA Scanner</div>', unsafe_allow_html=True)
with col2:
    if st.button("⚙️", key="hdr_settings"):
        st.session_state["view"] = "settings"

# --- Market Selector Pills ---
_mopts = ["Indian Stocks", "Crypto", "Forex"]
_mlabels = {"Indian Stocks": "📈 Stocks", "Crypto": "₿ Crypto", "Forex": "💱 Forex"}
market = st.radio(
    "Market",
    options=_mopts,
    format_func=lambda x: _mlabels.get(x, x),
    horizontal=True,
    label_visibility="collapsed",
    key="market_radio"
)
st.session_state["market"] = market
current_list = watchlists.get(market, [])

# --- SCANNER CONFIG PANEL ---
st.markdown('<div class="panel">', unsafe_allow_html=True)
st.markdown('<div class="panel-title">⏱ TIMEFRAME</div>', unsafe_allow_html=True)
timeframe = st.selectbox(
    "Timeframe",
    ["1m", "3m", "5m", "15m", "1h", "4h", "1d"],
    index=2,
    key="tf_select",
    label_visibility="collapsed"
)

st.markdown('<div class="panel-title" style="margin-top:0.7rem;">MIN SCORE ⭐</div>', unsafe_allow_html=True)
min_score = st.slider("Min Score", 0, 100, 60, 5, key="score_slider", label_visibility="collapsed")
auto_refresh = st.toggle("AUTO REFRESH 🔄", value=False, key="auto_ref_toggle")
st.markdown('</div>', unsafe_allow_html=True)

# 1. Single Progress Track Element
progress_slot = st.empty()
status_slot = st.empty()

# 2. Results Container Positioned BETWEEN Progress Track & Scan Button
results_slot = st.container()

# 3. Main Action Button
run_scan = st.button("🎯 SCAN NOW", type="primary", use_container_width=True, key="scan_main")

# --- TRIGGER SCAN PROCESS ---
if run_scan:
    if not current_list:
        st.error("Watchlist empty. Add symbols first.")
    else:
        results = []
        for i, sym in enumerate(current_list):
            status_slot.caption(f"Scanning {sym} ({i+1}/{len(current_list)})")
            
            # Update single progress bar
            progress_slot.progress((i + 1) / len(current_list))
            
            res = scan_symbol(sym, interval=timeframe, rules=scanners["My Price Action Scanner"]["rules"])
            if res and res["score"] >= min_score:
                results.append(res)
            time.sleep(0.05)
            
        status_slot.empty()
        results = sorted(results, key=lambda x: x["score"], reverse=True)
        st.session_state["scan_results"] = results
        st.session_state["last_scan"] = datetime.now().strftime("%H:%M")

# Render Scan Results inside the slot (between progress track and button)
with results_slot:
    results = st.session_state.get("scan_results", [])
    if results:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown(f'<div class="panel-title">SCAN RESULTS ({len(results)})</div>', unsafe_allow_html=True)
        for res in results:
            score = res["score"]
            score_cls = "score-green" if score >= 80 else ("score-yellow" if score >= 60 else "score-red")
            card_cls = "card card-green" if res.get("full_match") else "card"
            
            st.markdown(f"""
            <div class="{card_cls}">
              <div style="display:flex;justify-content:space-between;align-items:center;">
                <span style="font-size:1.1rem;font-weight:800;">{res.get('symbol','')}</span>
                <span class="score {score_cls}">{score}</span>
              </div>
              <div style="margin-top:0.3rem;color:#90a4c8;font-size:0.85rem;">
                Direction: <b>{res.get('direction','')}</b> | Level: <b>{res.get('near_level','')}</b><br>
                Price: <b>{res.get('price','')}</b> | Time: {res.get('setup_time_ist','')}
              </div>
            </div>
            """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

# --- WATCHLIST PANEL ---
st.markdown('<div class="panel">', unsafe_allow_html=True)
st.markdown('<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:0.5rem;"><span class="panel-title" style="margin:0;">★ WATCHLIST</span><span style="font-size:0.75rem;color:#00e5ff;font-weight:700;">View all ></span></div>', unsafe_allow_html=True)

# Compact Watchlist Preview Grid matching the image layout
cols = st.columns(2)
preview_items = [
    {"sym": "AAPL", "type": "Stocks", "price": "195.42", "change": "+1.32%"},
    {"sym": "BTC/USDT", "type": "Crypto", "price": "67,842.15", "change": "+2.57%"},
    {"sym": "EUR/USD", "type": "Forex", "price": "1.0876", "change": "+0.18%"},
    {"sym": "GBP/USD", "type": "Forex", "price": "1.2654", "change": "-0.21%"}
]

for idx, item in enumerate(preview_items):
    with cols[idx % 2]:
        st.markdown(f"""
        <div class="wl-card">
            <div class="sym">{item['sym']}</div>
            <div class="meta">{item['type']}</div>
            <div class="price">{item['price']}</div>
            <div class="change-pos">{item['change']}</div>
        </div>
        """, unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

# --- FIXED BOTTOM NAVIGATION BAR ---
st.markdown("""
<div class="bottom-nav">
  <div class="item">
    <span class="icon">📑</span> Dashboard
  </div>
  <div class="item active">
    <span class="icon">🎯</span> Scan
  </div>
  <div class="scan-fab">🎯</div>
  <div class="item">
    <span class="icon">🔔</span> Alerts
  </div>
  <div class="item">
    <span class="icon">⚙️</span> Settings
  </div>
</div>
""", unsafe_allow_html=True)
