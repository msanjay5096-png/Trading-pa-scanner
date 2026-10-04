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

DEFAULT_WATCHLISTS = {
    "Indian Stocks": ["RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "ICICIBANK.NS", "SBIN.NS", "BHARTIARTL.NS"],
    "Crypto": ["BTC-USD", "ETH-USD", "SOL-USD", "BNB-USD", "XRP-USD"],
    "Forex": ["EURUSD=X", "GBPUSD=X", "USDJPY=X", "USDINR=X"]
}

DEFAULT_SCANNERS = {
    "My Price Action Scanner": {
        "description": "Near key level + HL/LH + Healthy break close beyond level + Next candle rejection",
        "rules": {
            "require_trend": True, "require_near_sr": True, "sr_pct": 1.2,
            "require_consolidation": True, "min_consol_candles": 5, "prefer_decreasing": True,
            "require_volume_dry": True, "require_pin_bar": True, "require_high_volume_rejection": True
        }
    }
}

DEFAULT_SETTINGS = {"telegram_token": "", "telegram_chat_id": "", "enable_telegram": False}

def load_json(filepath, default):
    if os.path.exists(filepath):
        try:
            with open(filepath, "r") as f: return json.load(f)
        except: return default
    return default

def save_json(filepath, data):
    with open(filepath, "w") as f: json.dump(data, f, indent=2)

# ====================== SCANNER CORE LOGIC ======================
def detect_swing_points(df, left=3, right=3):
    highs, lows = df['High'].values, df['Low'].values
    swing_highs, swing_lows = [], []
    for i in range(left, len(df) - right):
        if highs[i] == max(highs[i-left:i+right+1]): swing_highs.append((i, highs[i]))
        if lows[i] == min(lows[i-left:i+right+1]): swing_lows.append((i, lows[i]))
    return swing_highs, swing_lows

def cluster_levels(levels, threshold_pct=0.35):
    if not levels: return []
    levels = sorted(levels)
    clusters, current = [], [levels[0]]
    for level in levels[1:]:
        if abs(level - current[-1]) / current[-1] * 100 <= threshold_pct: current.append(level)
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
    if not levels: return None
    best, best_dist = None, 999
    for lv in levels:
        dist = abs(price - lv) / price * 100
        if dist <= pct and dist < best_dist:
            best_dist, best = dist, lv
    return best

def is_healthy_break_candle(candle, direction="up"):
    body = abs(candle['Close'] - candle['Open'])
    rng = candle['High'] - candle['Low']
    if rng == 0 or (body / rng) < 0.45: return False
    if direction == "up":
        return candle['Close'] > candle['Open'] and ((candle['Close'] - candle['Low']) / rng) >= 0.65
    return candle['Close'] < candle['Open'] and ((candle['High'] - candle['Close']) / rng) >= 0.65

def is_rejection_candle(candle, direction="up"):
    body = abs(candle['Close'] - candle['Open'])
    upper = candle['High'] - max(candle['Open'], candle['Close'])
    lower = min(candle['Open'], candle['Close']) - candle['Low']
    rng = candle['High'] - candle['Low']
    if rng == 0: return False
    bear_reject = upper > lower * 1.4 and upper > body and upper / rng > 0.40
    bull_reject = lower > upper * 1.4 and lower > body and lower / rng > 0.40
    return (candle['Close'] < candle['Open'] or bear_reject) if direction == "up" else (candle['Close'] > candle['Open'] or bull_reject)

def scan_symbol(symbol, interval="5m", rules=None):
    try:
        ticker = yf.Ticker(symbol)
        period = "5d" if interval in ["1m", "3m", "5m"] else "60d"
        df = ticker.history(period=period, interval=interval)
        if df is None or len(df) < 50: return None
        df = df.dropna()

        supports, resistances = find_key_levels(df, lookback=45)
        for conf_offset in [0, -1]:
            conf_i = len(df) - 1 + conf_offset
            break_i = conf_i - 1
            if break_i < 20: continue

            break_candle, conf_candle = df.iloc[break_i], df.iloc[conf_i]
            break_close, conf_close = float(break_candle['Close']), float(conf_candle['Close'])

            near_res = price_near_level(break_close, resistances, pct=0.35)
            if near_res and break_close > near_res and is_healthy_break_candle(break_candle, "up"):
                if is_rejection_candle(conf_candle, "up"):
                    return {
                        "symbol": symbol, "score": 85, "direction": "Bullish",
                        "price": round(conf_close, 2), "near_level": "Resistance",
                        "level_price": round(near_res, 2),
                        "setup_time_ist": to_ist_str(df.index[conf_i])
                    }

            near_sup = price_near_level(break_close, supports, pct=0.35)
            if near_sup and break_close < near_sup and is_healthy_break_candle(break_candle, "down"):
                if is_rejection_candle(conf_candle, "down"):
                    return {
                        "symbol": symbol, "score": 85, "direction": "Bearish",
                        "price": round(conf_close, 2), "near_level": "Support",
                        "level_price": round(near_sup, 2),
                        "setup_time_ist": to_ist_str(df.index[conf_i])
                    }
        return None
    except: return None

# Fetch ticker overview data for Watchlist Grid
@st.cache_data(ttl=60)
def fetch_card_data(symbol):
    try:
        t = yf.Ticker(symbol)
        h = t.history(period="2d")
        if len(h) >= 2:
            cp = h['Close'].iloc[-1]
            prev = h['Close'].iloc[-2]
            chg = ((cp - prev) / prev) * 100
            return round(cp, 2), round(chg, 2)
    except: pass
    return 0.0, 0.0

# ====================== STREAMLIT PAGE SETUP ======================
st.set_page_config(page_title="PA Scanner", page_icon="📈", layout="centered")

# EXACT IMAGE STYLING CSS
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800;900&display=swap');

html, body, [class*="css"], .stApp {
    font-family: 'Plus Jakarta Sans', system-ui, sans-serif !important;
    background-color: #070913 !important;
    color: #ffffff !important;
}

#MainMenu, footer, header, [data-testid="stSidebar"], .stDeployButton { display: none !important; }

.block-container {
    padding-top: 0.8rem !important;
    padding-bottom: 6rem !important;
    padding-left: 0.8rem !important;
    padding-right: 0.8rem !important;
    max-width: 420px !important;
}

/* Header */
.top-bar {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 1rem;
}
.app-title {
    font-size: 1.65rem;
    font-weight: 900;
    font-style: italic;
    background: linear-gradient(90deg, #00e5ff, #00b0ff);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    display: flex;
    align-items: center;
    gap: 8px;
}
.gear-btn {
    background: rgba(255, 255, 255, 0.08);
    border: 1px solid rgba(255, 255, 255, 0.15);
    border-radius: 12px;
    padding: 6px;
    color: #fff;
}

/* Glass Main Container */
.panel-card {
    background: linear-gradient(165deg, rgba(22, 28, 48, 0.8) 0%, rgba(12, 16, 30, 0.9) 100%);
    border: 1px solid rgba(0, 229, 255, 0.2);
    border-radius: 24px;
    padding: 1.1rem;
    margin-bottom: 0.9rem;
    box-shadow: 0 12px 30px rgba(0,0,0,0.5), inset 0 1px 0 rgba(255,255,255,0.1);
    backdrop-filter: blur(16px);
}

.card-label {
    font-size: 0.72rem;
    font-weight: 800;
    color: #8b9bb4;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    display: flex;
    align-items: center;
    gap: 6px;
    margin-bottom: 0.5rem;
}

/* Market Selector Buttons */
.market-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 8px;
    margin-bottom: 0.9rem;
}
.market-btn {
    border-radius: 18px;
    padding: 10px 4px;
    text-align: center;
    font-weight: 700;
    font-size: 0.85rem;
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 6px;
    background: rgba(16, 22, 38, 0.7);
    border: 1px solid rgba(255, 255, 255, 0.1);
    color: #a0aec0;
}
.market-btn.active-stocks {
    border-color: #00e676 !important;
    color: #ffffff !important;
    box-shadow: 0 0 16px rgba(0, 230, 118, 0.35), inset 0 0 10px rgba(0, 230, 118, 0.2);
}
.market-btn.active-crypto {
    border-color: #e040fb !important;
    color: #ffffff !important;
    box-shadow: 0 0 16px rgba(224, 64, 251, 0.35), inset 0 0 10px rgba(224, 64, 251, 0.2);
}
.market-btn.active-forex {
    border-color: #00b0ff !important;
    color: #ffffff !important;
    box-shadow: 0 0 16px rgba(0, 176, 255, 0.35), inset 0 0 10px rgba(0, 176, 255, 0.2);
}

/* Custom Large SCAN NOW Button */
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #00e676 0%, #00b0ff 100%) !important;
    color: #04140a !important;
    font-size: 1.35rem !important;
    font-weight: 900 !important;
    border-radius: 28px !important;
    height: 4.2rem !important;
    box-shadow: 0 0 28px rgba(0, 230, 118, 0.5) !important;
    border: 1px solid #b9f6ca !important;
    letter-spacing: 0.05em !important;
    text-transform: uppercase !important;
}

/* Single Line Progress Bar Override */
.stProgress {
    height: 6px !important;
    margin: 0.6rem 0 0.8rem 0 !important;
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
    box-shadow: 0 0 12px #00e676;
}

/* Watchlist Grid (2 Columns matching image) */
.wl-grid {
    display: grid;
    grid-template-columns: repeat(2, 1fr);
    gap: 8px;
}
.wl-card {
    background: rgba(15, 21, 37, 0.85);
    border: 1px solid rgba(0, 229, 255, 0.15);
    border-radius: 16px;
    padding: 0.75rem 0.8rem;
}
.wl-header {
    display: flex;
    align-items: center;
    gap: 6px;
    font-weight: 800;
    font-size: 0.92rem;
    color: #ffffff;
}
.wl-sub {
    font-size: 0.7rem;
    color: #718096;
    margin-top: 2px;
}
.wl-price {
    font-size: 0.95rem;
    font-weight: 800;
    color: #ffffff;
    margin-top: 6px;
}
.wl-change-pos { font-size: 0.72rem; color: #00e676; font-weight: 700; }
.wl-change-neg { font-size: 0.72rem; color: #ff5252; font-weight: 700; }

/* Icon Badges */
.badge-icon {
    width: 22px;
    height: 22px;
    border-radius: 50%;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    font-size: 0.75rem;
    font-weight: 800;
}
.icon-stocks { background: rgba(0, 230, 118, 0.2); color: #00e676; border: 1px solid #00e676; }
.icon-crypto { background: rgba(255, 152, 0, 0.2); color: #ff9800; border: 1px solid #ff9800; }
.icon-forex { background: rgba(156, 39, 176, 0.2); color: #e040fb; border: 1px solid #e040fb; }

/* Bottom Nav */
.bottom-nav {
    position: fixed; left: 0; right: 0; bottom: 0;
    background: rgba(8, 11, 20, 0.95);
    border-top: 1px solid rgba(0, 229, 255, 0.2);
    backdrop-filter: blur(20px);
    display: flex; justify-content: space-around; align-items: center;
    padding: 0.5rem 0.4rem 0.8rem 0.4rem;
    z-index: 999;
}
.bottom-nav .item {
    text-align: center; color: #64748b; font-size: 0.65rem; font-weight: 700;
}
.bottom-nav .item.active { color: #00e5ff; }
.bottom-nav .icon { font-size: 1.25rem; display: block; margin-bottom: 2px; }
.bottom-nav .scan-fab {
    width: 54px; height: 54px; border-radius: 50%;
    background: radial-gradient(circle at 30% 30%, #00e676, #00c853);
    color: #03150a; font-size: 1.4rem;
    display: flex; align-items: center; justify-content: center;
    box-shadow: 0 0 24px rgba(0,230,118,0.6);
    margin-top: -22px;
    border: 2px solid #b9f6ca;
}
</style>
""", unsafe_allow_html=True)

# Watchlist Data & State Management
watchlists = load_json(WATCHLIST_FILE, DEFAULT_WATCHLISTS)
scanners = load_json(SCANNERS_FILE, DEFAULT_SCANNERS)

if "market" not in st.session_state: st.session_state["market"] = "Indian Stocks"
if "scan_results" not in st.session_state: st.session_state["scan_results"] = []

# Top Navigation Bar
st.markdown("""
<div class="top-bar">
    <div style="display:flex;align-items:center;gap:10px;">
        <span style="font-size:1.4rem;color:#00e5ff;">☰</span>
        <div class="app-title">PA Scanner</div>
    </div>
    <div class="gear-btn">⚙️</div>
</div>
""", unsafe_allow_html=True)

# 1. Market Selection (Interactive Glowing Pills)
m1, m2, m3 = st.columns(3)
curr_m = st.session_state["market"]

with m1:
    if st.button("📈 Stocks", key="btn_stocks", use_container_width=True):
        st.session_state["market"] = "Indian Stocks"
        st.rerun()
with m2:
    if st.button("₿ Crypto", key="btn_crypto", use_container_width=True):
        st.session_state["market"] = "Crypto"
        st.rerun()
with m3:
    if st.button("💱 Forex", key="btn_forex", use_container_width=True):
        st.session_state["market"] = "Forex"
        st.rerun()

market = st.session_state["market"]

# 2. Main Parameters Panel
st.markdown('<div class="panel-card">', unsafe_allow_html=True)

st.markdown('<div class="card-label">🕒 TIMEFRAME</div>', unsafe_allow_html=True)
timeframe = st.selectbox(
    "Timeframe", ["1m", "3m", "5m", "15m", "1h", "4h", "1d"],
    index=2, label_visibility="collapsed", key="tf_select"
)

st.markdown('<div style="display:flex;justify-content:space-between;align-items:center;margin-top:0.8rem;"><span class="card-label" style="margin:0;">MIN SCORE ℹ️</span><span style="font-weight:900;color:#fff;font-size:1rem;">60 ★</span></div>', unsafe_allow_html=True)
min_score = st.slider("Min Score", 0, 100, 60, 5, label_visibility="collapsed")

st.markdown('<div style="display:flex;justify-content:space-between;align-items:center;margin-top:0.8rem;"><span class="card-label" style="margin:0;">AUTO REFRESH 🔄</span></div>', unsafe_allow_html=True)
c1, c2, c3, c4 = st.columns([1.2, 1, 1, 1])
with c1: st.caption("Interval")
with c2: st.button("Off", key="off_btn")
with c3: st.button("10s", key="10s_btn")
with c4: st.button("30s", key="30s_btn")

st.markdown('</div>', unsafe_allow_html=True)

# 3. Single Scanning Progress Slot
progress_slot = st.empty()
status_slot = st.empty()

# 4. Results Container (POSITIONED BETWEEN PROGRESS BAR AND SCAN BUTTON)
results_slot = st.container()

# 5. Large "SCAN NOW" Main CTA Button
run_scan = st.button("🎯 SCAN NOW", type="primary", use_container_width=True)

# Scanning Process Logic
current_list = watchlists.get(market, [])
if run_scan:
    if not current_list:
        st.error("Watchlist empty.")
    else:
        results = []
        for i, sym in enumerate(current_list):
            status_slot.caption(f"Scanning {sym} ({i+1}/{len(current_list)})")
            progress_slot.progress((i + 1) / len(current_list))
            
            res = scan_symbol(sym, interval=timeframe, rules=scanners["My Price Action Scanner"]["rules"])
            if res and res["score"] >= min_score:
                results.append(res)
            time.sleep(0.05)
            
        status_slot.empty()
        results = sorted(results, key=lambda x: x["score"], reverse=True)
        st.session_state["scan_results"] = results

# Display Scan Results inside the allocated slot
with results_slot:
    results = st.session_state.get("scan_results", [])
    if results:
        st.markdown('<div class="panel-card">', unsafe_allow_html=True)
        st.markdown(f'<div class="card-label">SCAN RESULTS ({len(results)})</div>', unsafe_allow_html=True)
        for res in results:
            st.markdown(f"""
            <div style="background:rgba(0,229,255,0.05);border:1px solid #00e5ff;border-radius:14px;padding:10px;margin-bottom:8px;">
                <div style="display:flex;justify-content:space-between;align-items:center;">
                    <span style="font-weight:900;font-size:1.05rem;">{res['symbol']}</span>
                    <span style="color:#00e676;font-weight:900;font-size:1.2rem;">{res['score']} ★</span>
                </div>
                <div style="font-size:0.8rem;color:#a0aec0;margin-top:4px;">
                    {res['direction']} | Near {res['near_level']} ({res['level_price']}) | Price: {res['price']}
                </div>
            </div>
            """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

# 6. Watchlist Section (Grid display matching image)
st.markdown('<div class="panel-card">', unsafe_allow_html=True)
st.markdown('<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:0.7rem;"><span class="card-label" style="margin:0;">★ WATCHLIST</span><span style="font-size:0.75rem;color:#00e5ff;font-weight:700;">View all ></span></div>', unsafe_allow_html=True)

# Display sample watchlist preview matching screenshot
sample_grid = [
    {"sym": "AAPL", "market": "Stocks", "icon": "📈", "class": "icon-stocks", "fallback_price": "195.42", "fallback_chg": "+1.32%"},
    {"sym": "BTC/USDT", "market": "Crypto", "icon": "₿", "class": "icon-crypto", "fallback_price": "67,842.15", "fallback_chg": "+2.57%"},
    {"sym": "EUR/USD", "market": "Forex", "icon": "€", "class": "icon-forex", "fallback_price": "1.0876", "fallback_chg": "+0.18%"},
    {"sym": "GBP/USD", "market": "Forex", "icon": "£", "class": "icon-forex", "fallback_price": "1.2654", "fallback_chg": "-0.21%"}
]

cols = st.columns(2)
for idx, item in enumerate(sample_grid):
    with cols[idx % 2]:
        chg_class = "wl-change-pos" if "+" in item['fallback_chg'] else "wl-change-neg"
        st.markdown(f"""
        <div class="wl-card">
            <div class="wl-header">
                <span class="badge-icon {item['class']}">{item['icon']}</span>
                <span>{item['sym']}</span>
            </div>
            <div class="wl-sub">{item['market']}</div>
            <div class="wl-price">{item['fallback_price']}</div>
            <div class="{chg_class}">{item['fallback_chg']}</div>
        </div>
        """, unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

# 7. Bottom Navigation
st.markdown("""
<div class="bottom-nav">
  <div class="item">
    <span class="icon">📊</span> Dashboard
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
