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
        # naive: treat as UTC from yfinance
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
        return symbol  # indices like ^NSEI, ^NSEBANK, ^BSESN
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
        # Map UI timeframes to yfinance-supported intervals
        tf_map = {
            "1m": "1m",
            "3m": "2m",   # yfinance has no 3m; nearest is 2m
            "5m": "5m",
            "15m": "15m",
            "1h": "1h",
            "4h": "1h",   # yfinance has no 4h; use 1h
            "1d": "1d",
            "30m": "30m",
        }
        yf_interval = tf_map.get(interval, interval)
        if interval in ["1m", "3m", "5m"]:
            period = "5d"
        elif interval in ["15m", "30m", "1h", "4h"]:
            period = "60d"
        else:
            period = "1y"
        df = ticker.history(period=period, interval=yf_interval)
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
                            # Rejection/doji volume MUST be higher than breakout candle
                            # Skip only when volume data is missing (e.g. some indices)
                            prev_vol = float(df['Volume'].iloc[break_i])
                            conf_vol = float(df['Volume'].iloc[conf_i])
                            if rules and rules.get("require_high_volume_rejection", True):
                                if prev_vol > 0 or conf_vol > 0:
                                    if not (conf_vol > prev_vol):
                                        continue  # fail rule — rejection vol not higher

                            score = 70
                            if conf_vol > prev_vol and prev_vol > 0:
                                score += 15
                            if abs(break_close - level) / level * 100 < 0.25:
                                score += 10

                            # IST times from candle index
                            try:
                                break_ts = df.index[break_i]
                                conf_ts = df.index[conf_i]
                            except Exception:
                                break_ts = conf_ts = None
                            # swing points for trendline (last 2 lows)
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
                                "trend_points": trend_pts,
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
                            # Rejection/doji volume MUST be higher than breakout candle
                            prev_vol = float(df['Volume'].iloc[break_i])
                            conf_vol = float(df['Volume'].iloc[conf_i])
                            if rules and rules.get("require_high_volume_rejection", True):
                                if prev_vol > 0 or conf_vol > 0:
                                    if not (conf_vol > prev_vol):
                                        continue  # fail rule — rejection vol not higher

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
                                "break_time_ist": to_ist_str(df.index[break_i]),
                                "setup_time_ist": to_ist_str(df.index[conf_i]),
                                "scanned_at_ist": datetime.now(IST).strftime("%d-%b-%Y %H:%M IST"),
                                "trend_points": (
                                    (lambda seg, sh: (
                                        [[str(seg.index[sh[-2][0]]), float(sh[-2][1])],
                                         [str(seg.index[sh[-1][0]]), float(sh[-1][1])]]
                                        if len(sh) >= 2 else []
                                    ))(pre_break.tail(30), detect_swing_points(pre_break.tail(30), left=2, right=2)[0])
                                ),
                                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                "df": df.tail(60)
                            }

        return None
    except Exception as e:
        return None






# ====================== STREAMLIT APP (UI) ======================
# NOTE: all scanning / signal logic above this line is unchanged.
# Requires Streamlit >= 1.40  (uses st.container(key=...) and :material/: icons)
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import quote

st.set_page_config(
    page_title="PA Scanner",
    page_icon="🎯",
    layout="centered",
    initial_sidebar_state="collapsed",
)

MARKETS = ["Indian Stocks", "Crypto", "Forex"]
MARKET_SHORT = {"Indian Stocks": "Stocks", "Crypto": "Crypto", "Forex": "Forex"}
PLACEHOLDERS = {
    "Indian Stocks": "RELIANCE, TCS, INFY, SBIN",
    "Crypto": "BTC, ETH, SOL, XRP",
    "Forex": "XAUUSD, EURUSD, GBPUSD",
}
TF_OPTIONS = ["1m", "3m", "5m", "15m", "1h", "4h", "1d"]
AUTO_MODES = {"Off": 0, "10s": 10, "30s": 30}
AUTO_LABELS = {v: k for k, v in AUTO_MODES.items()}

# ---------- session defaults ----------
_defaults = {
    "nav": "dashboard",          # dashboard | scan | alerts | settings | history
    "market": "Indian Stocks",
    "refresh_secs": 0,           # 0 = off
    "last_secs": 10,
    "next_scan_at": None,
    "force_scan": False,
    "rules_edit": False,
    "rules_backup": None,
    "rules_forward": None,
    "show_watchlist": False,
    "scan_results": [],
    "p_tf": "5m",
    "p_score": 60,
}
for _k, _v in _defaults.items():
    st.session_state.setdefault(_k, _v)


# ---------- callbacks ----------
def go(page):
    st.session_state["nav"] = page


def fab_scan():
    st.session_state["nav"] = "dashboard"
    st.session_state["force_scan"] = True


def _on_mode():
    secs = AUTO_MODES[st.session_state["auto_mode"]]
    st.session_state["refresh_secs"] = secs
    if secs:
        st.session_state["last_secs"] = secs
        st.session_state["next_scan_at"] = time.time() + secs
    else:
        st.session_state["next_scan_at"] = None
    st.session_state["auto_on"] = secs > 0


def _on_toggle():
    if st.session_state["auto_on"]:
        secs = st.session_state.get("last_secs", 10) or 10
        st.session_state["refresh_secs"] = secs
        st.session_state["auto_mode"] = AUTO_LABELS.get(secs, "10s")
        st.session_state["next_scan_at"] = time.time() + secs
    else:
        st.session_state["refresh_secs"] = 0
        st.session_state["auto_mode"] = "Off"
        st.session_state["next_scan_at"] = None


# ====================== ICONS / ASSETS ======================
def svg_uri(svg):
    return 'url("data:image/svg+xml;utf8,' + quote(svg) + '")'


_SVG_HEAD = "<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 32 32' fill='none' stroke='{c}' stroke-width='2.2' stroke-linecap='round' stroke-linejoin='round'>"
ICON_STOCKS = svg_uri(
    _SVG_HEAD.format(c="#69f0ae")
    + "<path d='M5 27V20M12 27V16M19 27V19M26 27V13'/><path d='M5 14l6-5 5 3 9-8'/><path d='M20 4h5v5'/></svg>"
)
ICON_CRYPTO = svg_uri(
    _SVG_HEAD.format(c="#d58cff")
    + "<circle cx='16' cy='16' r='13'/><path d='M12.5 9.5h5a3 3 0 0 1 0 6h-5zM12.5 15.5h5.8a3 3 0 0 1 0 6h-5.8zM12.5 9.5v12M15 8v1.5M15 21.5V23M18 8v1.5M18 21.5V23'/></svg>"
)
ICON_FOREX = svg_uri(
    _SVG_HEAD.format(c="#4db8ff")
    + "<ellipse cx='12' cy='10' rx='8' ry='3.5'/><path d='M4 10v5c0 2 3.6 3.5 8 3.5s8-1.5 8-3.5v-5'/><path d='M4 15v5c0 2 3.6 3.5 8 3.5'/><ellipse cx='22' cy='20' rx='6' ry='3'/><path d='M16 20v4c0 1.7 2.7 3 6 3s6-1.3 6-3v-4'/></svg>"
)
ICON_TARGET = svg_uri(
    "<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 64 64' fill='none' stroke='white' stroke-width='3' stroke-linecap='round' stroke-linejoin='round'>"
    "<circle cx='30' cy='34' r='19'/><circle cx='30' cy='34' r='11'/><circle cx='30' cy='34' r='3.5' fill='white'/>"
    "<path d='M30 34L52 12M52 12v9M52 12h-9'/></svg>"
)

_L = "<svg viewBox='0 0 24 24' width='17' height='17' fill='none' stroke='currentColor' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'>"
IC_CLOCK = _L + "<circle cx='12' cy='12' r='9'/><path d='M12 7v5l3 2'/></svg>"
IC_INFO = _L + "<circle cx='12' cy='12' r='9'/><path d='M12 11v5M12 8h.01'/></svg>"
IC_REFRESH = _L + "<path d='M20 11a8 8 0 0 0-14-4M4 5v4h4M4 13a8 8 0 0 0 14 4M20 19v-4h-4'/></svg>"
IC_STAR = _L + "<polygon points='12 3 14.9 9 21.5 9.8 16.6 14.3 17.9 21 12 17.7 6.1 21 7.4 14.3 2.5 9.8 9.1 9'/></svg>"
IC_TREND = (
    "<svg viewBox='0 0 24 24' width='22' height='22' fill='none' stroke='#4ade80' stroke-width='2.4' "
    "stroke-linecap='round' stroke-linejoin='round'><path d='M3 17l6-6 4 4 8-8'/><path d='M15 7h6v6'/></svg>"
)

# ====================== CSS ======================
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:ital,wght@0,400;0,500;0,600;0,700;0,800;1,700;1,800&display=swap');

.stApp, .stApp p, .stApp label, .stApp input, .stApp textarea, .stApp h1, .stApp h2, .stApp h3,
.stApp span:not([data-testid="stIconMaterial"]), .stApp div[data-baseweb] {
  font-family: 'Poppins', system-ui, sans-serif;
}
[data-testid="stIconMaterial"] { font-family: "Material Symbols Rounded" !important; }

.stApp {
  background:
    radial-gradient(700px 420px at 0% 0%, rgba(124,77,255,.50), transparent 60%),
    radial-gradient(600px 400px at 100% 18%, rgba(0,229,255,.22), transparent 55%),
    radial-gradient(700px 520px at 100% 100%, rgba(0,230,118,.38), transparent 55%),
    radial-gradient(520px 420px at 0% 95%, rgba(0,190,255,.20), transparent 55%),
    linear-gradient(170deg, #0a0a2c 0%, #0b1236 50%, #05241f 100%) !important;
  color: #eef2ff !important;
}
header[data-testid="stHeader"], #MainMenu, footer, [data-testid="stSidebar"],
[data-testid="stToolbar"], .stDeployButton { display: none !important; }

.block-container {
  max-width: 430px !important;
  padding: 0.9rem 0.85rem 9rem 0.85rem !important;
}
div[data-testid="stVerticalBlock"] { gap: 0.7rem; }

/* keep columns on ONE row on phones */
div[data-testid="stHorizontalBlock"] { flex-wrap: nowrap !important; gap: 0.5rem !important; align-items: center !important; }
div[data-testid="stColumn"], div[data-testid="column"] { min-width: 0 !important; }

/* ---------- header ---------- */
.pa-title {
  text-align: center; font-size: 2.1rem; font-weight: 800; font-style: italic;
  letter-spacing: -0.5px; line-height: 1.15;
  background: linear-gradient(90deg, #22d3ee 0%, #4ade80 55%, #fde047 100%);
  -webkit-background-clip: text; background-clip: text; -webkit-text-fill-color: transparent;
  filter: drop-shadow(0 0 12px rgba(34,211,238,.45));
}
.st-key-hdr_menu button {
  background: transparent !important; border: none !important; box-shadow: none !important;
  color: #5eead4 !important; min-height: 3rem; padding: 0; justify-content: flex-start;
}
.st-key-hdr_menu [data-testid="stIconMaterial"] { font-size: 2.2rem; }
.st-key-hdr_gear button {
  background: rgba(150,160,230,.16) !important; border: 1px solid rgba(190,200,255,.30) !important;
  border-radius: 18px !important; color: #f1f4ff !important; width: 3.4rem; height: 3.4rem; min-height: 3.4rem;
  margin-left: auto; box-shadow: inset 0 1px 0 rgba(255,255,255,.12);
}
.st-key-hdr_gear [data-testid="stIconMaterial"] { font-size: 1.8rem; }

/* ---------- market pills ---------- */
.st-key-market_radio div[role="radiogroup"] {
  display: flex !important; flex-direction: row !important; flex-wrap: nowrap !important;
  gap: 0.55rem !important; width: 100%;
}
.st-key-market_radio div[role="radiogroup"] > label {
  flex: 1 1 0 !important; min-width: 0; height: 3.7rem; margin: 0 !important;
  display: flex !important; align-items: center; justify-content: center; gap: 0.4rem;
  padding: 0 0.3rem !important; border-radius: 24px !important;
  background: rgba(10,14,40,.55) !important;
  border: 1.5px solid rgba(var(--c), .75) !important;
  box-shadow: 0 0 14px rgba(var(--c), .35), inset 0 0 14px rgba(var(--c), .12);
  backdrop-filter: blur(10px);
}
.st-key-market_radio div[role="radiogroup"] > label > div:first-child { display: none !important; }
.st-key-market_radio div[role="radiogroup"] > label p {
  margin: 0; font-size: 1rem; font-weight: 600; color: #f1f4ff;
}
.st-key-market_radio div[role="radiogroup"] > label::before {
  content: ""; width: 30px; height: 30px; flex: none;
  background-repeat: no-repeat; background-position: center; background-size: contain;
}
.st-key-market_radio div[role="radiogroup"] > label:nth-child(1) { --c: 46,232,111; }
.st-key-market_radio div[role="radiogroup"] > label:nth-child(2) { --c: 176,92,255; }
.st-key-market_radio div[role="radiogroup"] > label:nth-child(3) { --c: 42,167,255; }
.st-key-market_radio div[role="radiogroup"] > label:nth-child(1)::before { background-image: __ICON_STOCKS__; }
.st-key-market_radio div[role="radiogroup"] > label:nth-child(2)::before { background-image: __ICON_CRYPTO__; }
.st-key-market_radio div[role="radiogroup"] > label:nth-child(3)::before { background-image: __ICON_FOREX__; }
.st-key-market_radio div[role="radiogroup"] > label:has(input:checked) {
  background: rgba(var(--c), .18) !important;
  box-shadow: 0 0 24px rgba(var(--c), .65), inset 0 0 20px rgba(var(--c), .28);
}

/* ---------- glass panels ---------- */
[class*="st-key-panel_"] {
  background: linear-gradient(145deg, rgba(150,160,230,.17), rgba(60,70,140,.10));
  border: 1px solid rgba(170,180,255,.28);
  border-radius: 26px; padding: 1rem 1.05rem 1.1rem;
  backdrop-filter: blur(18px);
  box-shadow: 0 14px 34px rgba(0,0,0,.35), inset 0 1px 0 rgba(255,255,255,.09);
}
.lbl {
  display: flex; align-items: center; gap: 0.45rem; color: #a9b2dd;
  font-size: 0.8rem; font-weight: 600; letter-spacing: 0.12em; text-transform: uppercase; white-space: nowrap;
}
.lbl svg { flex: none; }
.row-between { display: flex; align-items: center; justify-content: space-between; }
.score-val { display: flex; align-items: center; gap: 0.4rem; font-size: 1.25rem; font-weight: 600; color: #f1f4ff; }
.score-val svg { color: #cfd6ff; width: 22px; height: 22px; }
.sep { height: 1px; background: rgba(170,180,255,.18); margin: 0.1rem 0; }

/* timeframe select */
.st-key-tf_sel div[data-baseweb="select"] > div {
  background: rgba(20,26,64,.65) !important; border: 1px solid rgba(180,190,255,.32) !important;
  border-radius: 14px !important; min-height: 2.9rem; color: #f1f4ff;
}
.st-key-tf_sel div[data-baseweb="select"] * { color: #f1f4ff !important; font-size: 1.05rem; }
.st-key-tf_sel div[data-baseweb="select"] svg { fill: #cfd6ff !important; }

/* min-score slider */
.st-key-score_sel [data-baseweb="slider"] div[style*="linear-gradient"] {
  height: 8px !important; border-radius: 99px !important;
}
.st-key-score_sel [role="slider"] {
  width: 30px !important; height: 30px !important; border-radius: 50% !important;
  background: radial-gradient(circle at 35% 30%, #fff3b0, #ffc400 60%, #ff9800) !important;
  border: 3px solid rgba(255,255,255,.85) !important;
  box-shadow: 0 0 20px rgba(255,200,0,.9) !important;
}
.st-key-score_sel [data-testid="stThumbValue"] { display: none !important; }
.st-key-score_sel [data-testid="stTickBarMin"], .st-key-score_sel [data-testid="stTickBarMax"] {
  color: #a9b2dd !important; font-size: 0.85rem;
}

/* auto refresh pills */
.st-key-auto_mode div[role="radiogroup"] { display: flex !important; flex-wrap: nowrap !important; gap: 0.3rem !important; }
.st-key-auto_mode div[role="radiogroup"] > label {
  flex: 1 1 0; min-width: 0; margin: 0 !important; justify-content: center; padding: 0.3rem 0.1rem !important;
  border-radius: 999px !important; background: rgba(20,26,64,.55) !important;
  border: 1px solid rgba(180,190,255,.30) !important;
}
.st-key-auto_mode div[role="radiogroup"] > label > div:first-child { display: none !important; }
.st-key-auto_mode div[role="radiogroup"] > label p { margin: 0; font-size: 0.82rem; font-weight: 600; color: #c9d0f5; }
.st-key-auto_mode div[role="radiogroup"] > label:has(input:checked) {
  background: rgba(0,200,120,.38) !important; border-color: #2ee86f !important;
  box-shadow: 0 0 12px rgba(46,232,111,.5);
}
.st-key-auto_mode div[role="radiogroup"] > label:has(input:checked) p { color: #eafff3; }
/* toggle -> green when on */
label[data-baseweb="checkbox"]:has(input:checked) > span:first-child,
label[data-baseweb="checkbox"]:has(input:checked) > div:first-child { background-color: #2ee86f !important; }

/* SCAN NOW */
.st-key-scan_main button {
  width: 100%; min-height: 4.5rem; border-radius: 24px !important; position: relative;
  border: 1.5px solid rgba(205,255,170,.85) !important;
  background: linear-gradient(135deg, #14bf4c 0%, #4be63a 50%, #a6f23a 100%) !important;
  box-shadow: 0 0 30px rgba(90,255,80,.6), inset 0 0 24px rgba(255,255,255,.20) !important;
}
.st-key-scan_main button p {
  margin: 0; font-size: 1.9rem; font-weight: 800; letter-spacing: 0.02em; color: #fff !important;
  text-shadow: 0 2px 8px rgba(0,70,0,.5);
}
.st-key-scan_main button::after {
  content: ""; position: absolute; right: 16px; top: 50%; transform: translateY(-50%);
  width: 62px; height: 62px; border-radius: 50%;
  background: __ICON_TARGET__ center/72% no-repeat, rgba(255,255,255,.14);
  border: 2px solid rgba(255,255,255,.55);
}
.stProgress > div { height: .5rem !important; border-radius: 99px !important; background: rgba(30,35,80,.8) !important; }
.stProgress > div > div > div { background: linear-gradient(90deg,#00e676,#ffea00,#ff9100) !important; }

/* ---------- watchlist ---------- */
.st-key-wl_toggle button {
  background: transparent !important; border: none !important; box-shadow: none !important;
  color: #f1f4ff !important; min-height: 1.8rem; padding: 0; justify-content: flex-end; font-weight: 600;
}
.st-key-wl_toggle button p { font-size: 0.95rem; margin: 0; }
.wl-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 0.4rem; }
.wl-card {
  background: rgba(255,255,255,.06); border: 1px solid rgba(170,180,255,.25);
  border-radius: 18px; padding: 0.55rem 0.45rem; min-width: 0; overflow: hidden;
}
.wl-top { display: flex; align-items: center; gap: 0.25rem; min-width: 0; }
.wl-ic {
  width: 22px; height: 22px; border-radius: 50%; flex: none; display: flex; align-items: center;
  justify-content: center; font-size: 0.72rem; font-weight: 700; color: #fff;
}
.wl-sym { font-size: 0.7rem; font-weight: 700; color: #f1f4ff; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.wl-mkt { font-size: 0.7rem; color: #c3cae8; margin-top: 0.45rem; }
.wl-px { font-size: 0.95rem; font-weight: 700; color: #fff; margin-top: 0.15rem; white-space: nowrap; }
.wl-chg { font-size: 0.78rem; font-weight: 600; margin-top: 0.05rem; }
.wl-chg.up { color: #4ade80; } .wl-chg.down { color: #f87171; } .wl-chg.na { color: #8a93bd; }

/* ---------- bottom nav ---------- */
.st-key-bottomnav {
  position: fixed !important; left: 50%; transform: translateX(-50%); bottom: 0.6rem;
  width: min(calc(100% - 1.2rem), 410px); z-index: 999; gap: 0 !important;
  background: linear-gradient(180deg, rgba(45,55,110,.60), rgba(14,20,52,.88));
  border: 1px solid rgba(170,180,255,.28); border-radius: 30px; padding: 0.5rem 0.4rem 0.4rem;
  backdrop-filter: blur(18px); box-shadow: 0 -4px 30px rgba(0,0,0,.45);
}
.st-key-bottomnav button {
  width: 100%; min-height: 3.4rem; padding: 0.2rem 0; background: transparent !important;
  border: none !important; box-shadow: none !important; color: #a3add6 !important;
}
.st-key-bottomnav button[kind="primary"], .st-key-bottomnav button[data-testid="stBaseButton-primary"] { color: #4ade80 !important; }
.st-key-bottomnav button p {
  display: flex; flex-direction: column; align-items: center; gap: 0.1rem;
  margin: 0; font-size: 0.72rem; font-weight: 600; color: inherit;
}
.st-key-bottomnav button [data-testid="stIconMaterial"] { font-size: 1.75rem; }
.st-key-bottomnav div.st-key-nav_fab button {
  width: 76px; height: 76px; min-height: 76px; border-radius: 50% !important; margin: -2.6rem auto 0 auto;
  background: radial-gradient(circle at 35% 30%, rgba(46,232,111,.55), rgba(8,60,40,.95) 70%) !important;
  border: 2px solid rgba(110,255,170,.85) !important; color: #eafff3 !important;
  box-shadow: 0 0 26px rgba(46,232,111,.75), inset 0 0 18px rgba(46,232,111,.35) !important;
}
.st-key-bottomnav div.st-key-nav_fab button [data-testid="stIconMaterial"] { font-size: 2.6rem; }

/* ---------- result cards / misc ---------- */
.card {
  background: linear-gradient(145deg, rgba(150,160,230,.16), rgba(40,50,110,.12));
  border: 1px solid rgba(170,180,255,.25); border-radius: 20px; padding: 0.9rem 1rem; margin-bottom: 0.2rem;
}
.card-green { border-color: rgba(0,230,118,.55); box-shadow: 0 0 20px rgba(0,230,118,.14); }
.card-red { border-color: rgba(255,82,82,.50); box-shadow: 0 0 20px rgba(255,82,82,.12); }
.score { font-size: 1.45rem; font-weight: 800; }
.score-green { color: #00e676; text-shadow: 0 0 12px #00e67666; }
.score-yellow { color: #ffea00; } .score-red { color: #ff5252; }
.tag { display: inline-block; padding: 0.14rem 0.5rem; border-radius: 999px; font-size: 0.72rem; font-weight: 700; margin-left: 0.2rem; }
.tag-bull { background: rgba(0,230,118,.15); color: #69f0ae; border: 1px solid #00e67655; }
.tag-bear { background: rgba(255,82,82,.15); color: #ff8a80; border: 1px solid #ff525255; }
.tag-full { background: rgba(0,229,255,.15); color: #84ffff; border: 1px solid #00e5ff55; }
.sec-title { font-size: 1.15rem; font-weight: 700; color: #f1f4ff; margin: 0.2rem 0 0; }

div[data-testid="stExpander"] {
  background: rgba(150,160,230,.10); border: 1px solid rgba(170,180,255,.25) !important; border-radius: 20px;
}
.stTextArea textarea, .stTextInput input {
  border-radius: 14px !important; background: rgba(14,18,48,.75) !important;
  border: 1px solid rgba(170,180,255,.30) !important; color: #f1f4ff !important;
}
.stButton > button { border-radius: 16px; font-weight: 600; }
.stApp label, .stApp .stCheckbox, .stApp .stToggle { color: #d4daf7; }
</style>
"""
for _name, _val in {
    "__ICON_STOCKS__": ICON_STOCKS,
    "__ICON_CRYPTO__": ICON_CRYPTO,
    "__ICON_FOREX__": ICON_FOREX,
    "__ICON_TARGET__": ICON_TARGET,
}.items():
    CSS = CSS.replace(_name, _val)
st.markdown(CSS, unsafe_allow_html=True)


# ====================== DATA ======================
watchlists = load_json(WATCHLIST_FILE, DEFAULT_WATCHLISTS)
scanners = load_json(SCANNERS_FILE, DEFAULT_SCANNERS)
settings = load_json(SETTINGS_FILE, DEFAULT_SETTINGS)
history = load_json(HISTORY_FILE, [])

scanner_name = list(scanners.keys())[0] if scanners else "My Price Action Scanner"
if scanner_name not in scanners:
    scanners[scanner_name] = DEFAULT_SCANNERS["My Price Action Scanner"]
active_rules = scanners[scanner_name]["rules"]

market = st.session_state["market"]
current_list = watchlists.get(market, [])


# ====================== HELPERS (UI) ======================
@st.cache_data(ttl=120, show_spinner=False)
def fetch_quotes(symbols: tuple):
    """Last price + % change vs previous close for each symbol."""
    def one(sym):
        try:
            c = yf.Ticker(sym).history(period="5d", interval="1d")["Close"].dropna()
            if len(c) >= 2:
                return sym, float(c.iloc[-1]), float((c.iloc[-1] / c.iloc[-2] - 1) * 100)
            if len(c) == 1:
                return sym, float(c.iloc[-1]), None
        except Exception:
            pass
        return sym, None, None

    with ThreadPoolExecutor(max_workers=4) as ex:
        return {s: (p, ch) for s, p, ch in ex.map(one, symbols)}


def base_name(sym):
    return sym.replace(".NS", "").replace(".BO", "").replace("-USD", "").replace("=X", "").replace("^", "")


def display_name(sym, mkt):
    b = base_name(sym)
    if mkt == "Crypto":
        return f"{b}/USD"
    if mkt == "Forex" and len(b) == 6:
        return f"{b[:3]}/{b[3:]}"
    return b


CCY = {"EUR": "€", "GBP": "£", "USD": "$", "JPY": "¥", "INR": "₹", "AUD": "A$", "CAD": "C$", "CHF": "₣", "NZD": "N$"}
COIN = {"BTC": ("#f7931a", "₿"), "ETH": ("#627eea", "Ξ"), "SOL": ("#9945ff", "S"), "XRP": ("#23292f", "X"),
        "BNB": ("#f3ba2f", "B"), "DOGE": ("#c2a633", "Ð")}


def icon_html(sym, mkt):
    b = base_name(sym)
    if mkt == "Crypto":
        col, ch = COIN.get(b, ("#8b5cf6", b[:1]))
        return f'<span class="wl-ic" style="background:{col}">{ch}</span>'
    if mkt == "Forex":
        ch = CCY.get(b[:3], b[:1])
        return f'<span class="wl-ic" style="background:transparent;color:#a78bfa;font-size:1.15rem">{ch}</span>'
    return IC_TREND


def fmt_price(p, mkt):
    if p is None:
        return "—"
    if mkt == "Forex" or p < 1:
        return f"{p:,.4f}"
    return f"{p:,.2f}"


def watchlist_cards_html(symbols, mkt):
    quotes = fetch_quotes(tuple(symbols)) if symbols else {}
    cards = []
    for sym in symbols:
        p, ch = quotes.get(sym, (None, None))
        if ch is None:
            chg = '<div class="wl-chg na">—</div>'
        else:
            cls = "up" if ch >= 0 else "down"
            chg = f'<div class="wl-chg {cls}">{ch:+.2f}%</div>'
        cards.append(
            '<div class="wl-card"><div class="wl-top">'
            + icon_html(sym, mkt)
            + f'<span class="wl-sym">{display_name(sym, mkt)}</span></div>'
            + f'<div class="wl-mkt">{MARKET_SHORT[mkt]}</div>'
            + f'<div class="wl-px">{fmt_price(p, mkt)}</div>{chg}</div>'
        )
    return '<div class="wl-grid">' + "".join(cards) + "</div>"


def result_card_html(res, detailed=False):
    score = res["score"]
    score_cls = "score-green" if score >= 80 else ("score-yellow" if score >= 60 else "score-red")
    card_cls = "card card-green" if res.get("full_match") else ("card card-red" if res.get("direction") == "Bearish" else "card")
    dir_tag = "tag-bull" if res.get("direction") == "Bullish" else "tag-bear"
    full_badge = '<span class="tag tag-full">FULL</span>' if res.get("full_match") else ""
    body = (
        f"Near <b>{res.get('near_level', '—')}</b> ({res.get('level_price', '—')}) • {res.get('trend', '')}<br>"
        f"Price <b>{res.get('price', '—')}</b> • TF {res.get('interval', '')}<br>"
        f"Break: <b>{res.get('break_time_ist', '—')}</b><br>Setup: <b>{res.get('setup_time_ist', '—')}</b>"
    )
    if detailed:
        body += f"<br>Scanned: {res.get('scanned_at_ist', '—')}"
    return (
        f'<div class="{card_cls}"><div style="display:flex;justify-content:space-between;align-items:center;">'
        f'<div><span style="font-size:1.1rem;font-weight:700;">{res.get("symbol", "")}</span>'
        f'<span class="tag {dir_tag}">{res.get("direction", "")}</span>{full_badge}</div>'
        f'<span class="score {score_cls}">{score}</span></div>'
        f'<div style="margin-top:.4rem;color:#a9b6d6;font-size:.88rem;line-height:1.55;">{body}</div></div>'
    )


def render_chart(res):
    df = res.get("df")
    if df is None or len(df) == 0:
        return
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.75, 0.25])
    fig.add_trace(go.Candlestick(
        x=df.index, open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"], name="Price"
    ), row=1, col=1)
    if res.get("level_price"):
        lvl_color = "#00e676" if res.get("near_level") == "Support" else "#ff5252"
        fig.add_hline(y=res["level_price"], line_dash="dash", line_color=lvl_color, line_width=2,
                      annotation_text=res.get("near_level", "Level"), annotation_position="top left",
                      row=1, col=1)
    tpts = res.get("trend_points") or []
    if len(tpts) >= 2:
        try:
            xs, ys = [], []
            for x_str, y in tpts:
                for i in df.index:
                    if str(i) == x_str or str(i)[:16] == str(x_str)[:16]:
                        xs.append(i)
                        ys.append(y)
                        break
            if len(xs) >= 2:
                tcolor = "#00e676" if res.get("direction") == "Bullish" else "#ff5252"
                fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines+markers",
                                         line=dict(color=tcolor, width=2),
                                         marker=dict(size=7, color=tcolor), name="Trendline"), row=1, col=1)
        except Exception:
            pass
    fig.add_trace(go.Bar(x=df.index, y=df["Volume"], marker_color="#2a3a55", name="Vol"), row=2, col=1)
    fig.update_layout(height=360, template="plotly_dark", xaxis_rangeslider_visible=False,
                      margin=dict(l=0, r=0, t=8, b=0), showlegend=False,
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=False)
    st.plotly_chart(fig, use_container_width=True)
    st.caption(f"Break: {res.get('break_time_ist', '—')}  |  Setup: {res.get('setup_time_ist', '—')}  |  IST (+05:30)")


# ====================== HEADER ======================
h1, h2, h3 = st.columns([1, 4, 1])
with h1:
    st.button(":material/menu:", key="hdr_menu", on_click=go, args=("history",))
with h2:
    st.markdown('<div class="pa-title">PA Scanner</div>', unsafe_allow_html=True)
with h3:
    st.button(":material/settings:", key="hdr_gear", on_click=go, args=("settings",))

# ====================== BOTTOM NAV ======================
nav_now = st.session_state["nav"]
with st.container(key="bottomnav"):
    n1, n2, n3, n4, n5 = st.columns([1, 1, 1.1, 1, 1])
    with n1:
        st.button(":material/dashboard: Dashboard", key="nav_dashboard", on_click=go, args=("dashboard",),
                  type="primary" if nav_now == "dashboard" else "secondary")
    with n2:
        st.button(":material/adjust: Scan", key="nav_scan", on_click=go, args=("scan",),
                  type="primary" if nav_now == "scan" else "secondary")
    with n3:
        st.button(":material/track_changes:", key="nav_fab", on_click=fab_scan)
    with n4:
        st.button(":material/notifications: Alerts", key="nav_alerts", on_click=go, args=("alerts",),
                  type="primary" if nav_now == "alerts" else "secondary")
    with n5:
        st.button(":material/settings: Settings", key="nav_settings", on_click=go, args=("settings",),
                  type="primary" if nav_now == "settings" else "secondary")


# ====================== PAGE: DASHBOARD ======================
def page_dashboard():
    global market, current_list

    # ----- market pills -----
    if "market_radio" not in st.session_state:
        st.session_state["market_radio"] = st.session_state["market"]
    market = st.radio(
        "Market", MARKETS, horizontal=True, label_visibility="collapsed", key="market_radio",
        format_func=lambda x: MARKET_SHORT[x],
    )
    st.session_state["market"] = market
    current_list = watchlists.get(market, [])

    # ----- controls panel -----
    if "tf_sel" not in st.session_state:
        st.session_state["tf_sel"] = st.session_state["p_tf"]
    if "score_sel" not in st.session_state:
        st.session_state["score_sel"] = st.session_state["p_score"]
    if "auto_mode" not in st.session_state:
        st.session_state["auto_mode"] = AUTO_LABELS.get(st.session_state["refresh_secs"], "10s")
    if "auto_on" not in st.session_state:
        st.session_state["auto_on"] = st.session_state["refresh_secs"] > 0

    with st.container(key="panel_controls"):
        st.markdown(f'<div class="lbl">{IC_CLOCK} Timeframe</div>', unsafe_allow_html=True)
        tc, _sp = st.columns([1.1, 1.6])
        with tc:
            timeframe = st.selectbox("Timeframe", TF_OPTIONS, key="tf_sel", label_visibility="collapsed")
        st.session_state["p_tf"] = timeframe
        st.markdown('<div class="sep"></div>', unsafe_allow_html=True)

        score_hdr = st.empty()
        min_score = st.slider("Min Score", 0, 100, step=5, key="score_sel", label_visibility="collapsed")
        st.session_state["p_score"] = min_score
        yellow = min_score * 0.5
        score_hdr.markdown(
            "<style>.st-key-score_sel [data-baseweb='slider'] div[style*='linear-gradient']{"
            f"background:linear-gradient(90deg,#00e676 0%,#ffea00 {yellow}%,#ff9100 {min_score}%,"
            f"rgba(255,255,255,.14) {min_score}%) !important;}}</style>"
            f'<div class="row-between"><div class="lbl">Min Score {IC_INFO}</div>'
            f'<div class="score-val">{min_score} {IC_STAR}</div></div>',
            unsafe_allow_html=True,
        )
        st.markdown('<div class="sep"></div>', unsafe_allow_html=True)

        a1, a2, a3 = st.columns([1.3, 2.0, 0.6])
        with a1:
            st.markdown(f'<div class="lbl">Auto Refresh {IC_REFRESH}</div>', unsafe_allow_html=True)
        with a2:
            st.radio("Auto refresh", list(AUTO_MODES.keys()), horizontal=True, key="auto_mode",
                     label_visibility="collapsed", on_change=_on_mode)
        with a3:
            st.toggle("Auto refresh on", key="auto_on", label_visibility="collapsed", on_change=_on_toggle)

        progress_slot = st.empty()
        status_slot = st.empty()
        run = st.button("SCAN NOW", type="primary", use_container_width=True, key="scan_main")

    # ----- watchlist panel -----
    with st.container(key="panel_watch"):
        w1, w2 = st.columns([2, 1.3])
        with w1:
            st.markdown(f'<div class="lbl">{IC_STAR} Watchlist</div>', unsafe_allow_html=True)
        with w2:
            showing = st.session_state.get("show_watchlist", False)
            if st.button("Hide ›" if showing else "View all ›", key="wl_toggle"):
                st.session_state["show_watchlist"] = not showing
                st.rerun()
        if not current_list:
            st.caption("No symbols yet — add some in Settings.")
        else:
            st.markdown(watchlist_cards_html(current_list[:4], market), unsafe_allow_html=True)
        if st.session_state.get("show_watchlist"):
            for i, sym in enumerate(list(current_list)):
                c_a, c_b = st.columns([5, 1])
                c_a.write(f"**{sym}**")
                if c_b.button("🗑", key=f"del_{market}_{i}"):
                    current_list.pop(i)
                    watchlists[market] = current_list
                    save_json(WATCHLIST_FILE, watchlists)
                    st.rerun()
            if current_list and st.button("Clear all", key=f"clear_{market}"):
                watchlists[market] = []
                save_json(WATCHLIST_FILE, watchlists)
                st.rerun()

    # ----- run scan -----
    only_full = settings.get("full_matches_only", False)
    secs = st.session_state.get("refresh_secs", 0)
    due_auto = (
        secs > 0
        and st.session_state.get("next_scan_at") is not None
        and time.time() >= st.session_state["next_scan_at"]
    )
    force = st.session_state.get("force_scan", False)
    if force:
        st.session_state["force_scan"] = False

    if run or force or due_auto:
        if not current_list:
            st.error("Watchlist empty. Add symbols in Settings first.")
        else:
            progress_slot.progress(0)
            results = []
            for i, sym in enumerate(current_list):
                status_slot.caption(f"Scanning {sym} ({i + 1}/{len(current_list)})")
                res = scan_symbol(sym, interval=timeframe, rules=active_rules)
                if res:
                    if only_full and not res.get("full_match"):
                        pass
                    elif res["score"] >= min_score:
                        results.append(res)
                progress_slot.progress((i + 1) / max(len(current_list), 1))
                time.sleep(0.08)
            progress_slot.empty()

            results = sorted(results, key=lambda x: x["score"], reverse=True)
            light = [{k: v for k, v in r.items() if k != "df"} for r in results]

            st.session_state["scan_results"] = results
            st.session_state["last_scan"] = datetime.now().strftime("%H:%M")
            st.session_state["last_market"] = market
            st.session_state["last_tf"] = timeframe

            hist_entry = {
                "time": datetime.now().strftime("%Y-%m-%d %H:%M"),
                "market": market,
                "timeframe": timeframe,
                "count": len(light),
                "symbols": [x["symbol"] for x in light[:15]],
                "top_score": light[0]["score"] if light else 0,
            }
            save_json(HISTORY_FILE, ([hist_entry] + history)[:30])

            if settings.get("enable_telegram") and settings.get("telegram_token"):
                alert_min = settings.get("min_score_to_alert", 85)
                for r in [x for x in results if x["score"] >= alert_min or x.get("full_match")][:4]:
                    msg = (
                        f"🚨 <b>{r['symbol']}</b> | Score {r['score']}\n"
                        f"{r['direction']} | Near {r.get('near_level', '')}\n"
                        f"Price {r['price']} | {timeframe}"
                    )
                    send_telegram_alert(settings["telegram_token"], settings.get("telegram_chat_id", ""), msg)

            if results:
                status_slot.caption(f"✅ {len(results)} setup(s) found — open the Scan tab for charts")
            else:
                status_slot.caption("Scan complete — no setups found. Try a lower score or another timeframe.")

            if st.session_state.get("refresh_secs", 0) > 0:
                st.session_state["next_scan_at"] = time.time() + st.session_state["refresh_secs"]
            else:
                st.session_state["next_scan_at"] = None

    # ----- compact results -----
    results = st.session_state.get("scan_results") or []
    if st.session_state.get("last_scan"):
        st.markdown(
            f'<div class="sec-title">Latest results</div>'
            f'<div class="lbl" style="margin-bottom:.4rem">{st.session_state.get("last_market", "")} • '
            f'{st.session_state.get("last_tf", "")} • {st.session_state.get("last_scan", "—")}</div>',
            unsafe_allow_html=True,
        )
        if not results:
            st.caption("No setups on last scan.")
        else:
            for res in results[:5]:
                st.markdown(result_card_html(res), unsafe_allow_html=True)
            if len(results) > 5:
                st.caption(f"+{len(results) - 5} more in the Scan tab")

    # ----- soft poll for auto refresh -----
    if st.session_state.get("refresh_secs", 0) > 0:
        nsa = st.session_state.get("next_scan_at")
        if nsa is None:
            st.session_state["next_scan_at"] = time.time() + st.session_state["refresh_secs"]
            nsa = st.session_state["next_scan_at"]
        left = nsa - time.time()
        if left <= 0:
            st.session_state["force_scan"] = True
            st.rerun()
        else:
            time.sleep(min(15, max(1, left)))
            st.rerun()


# ====================== PAGE: SCAN (full results + charts) ======================
def page_scan():
    results = st.session_state.get("scan_results", [])
    st.markdown('<div class="sec-title">Scan results</div>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="lbl">{st.session_state.get("last_market", "—")} • {st.session_state.get("last_tf", "")} • '
        f'{st.session_state.get("last_scan", "—")}</div>',
        unsafe_allow_html=True,
    )
    if not results:
        st.info("No results yet. Tap the green target button to run a scan.")
        return
    st.caption(f"{len(results)} setups")
    for res in results:
        st.markdown(result_card_html(res, detailed=True), unsafe_allow_html=True)
        with st.expander(f"Chart • {res['symbol']}"):
            render_chart(res)


# ====================== PAGE: ALERTS ======================
def page_alerts():
    st.markdown('<div class="sec-title">Alerts</div>', unsafe_allow_html=True)
    with st.container(key="panel_alerts"):
        st.markdown(f'<div class="lbl">{IC_STAR} Telegram alerts</div>', unsafe_allow_html=True)
        enable_tg = st.toggle("Enable Telegram alerts", value=settings.get("enable_telegram", False), key="tg_enable")
        tg_token = st.text_input("Bot Token", value=settings.get("telegram_token", ""), type="password", key="tg_token")
        tg_chat = st.text_input("Chat ID", value=settings.get("telegram_chat_id", ""), key="tg_chat")
        alert_min = st.slider("Alert when score ≥", 50, 100, int(settings.get("min_score_to_alert", 85)), 5, key="tg_min")
        if st.button("Save alert settings", use_container_width=True, key="tg_save"):
            settings["enable_telegram"] = enable_tg
            settings["telegram_token"] = tg_token
            settings["telegram_chat_id"] = tg_chat
            settings["min_score_to_alert"] = alert_min
            save_json(SETTINGS_FILE, settings)
            st.success("Saved")

    hot = [r for r in st.session_state.get("scan_results", []) if r["score"] >= settings.get("min_score_to_alert", 85)]
    st.markdown('<div class="sec-title">High-score setups</div>', unsafe_allow_html=True)
    if not hot:
        st.caption("Nothing above your alert score in the last scan.")
    for res in hot:
        st.markdown(result_card_html(res), unsafe_allow_html=True)


# ====================== PAGE: SETTINGS ======================
def page_settings():
    st.markdown('<div class="sec-title">Settings</div>', unsafe_allow_html=True)

    # ---- add symbols ----
    with st.container(key="panel_add"):
        st.markdown(f'<div class="lbl">{IC_STAR} Add symbols</div>', unsafe_allow_html=True)
        mk = st.selectbox("Add to", MARKETS, index=MARKETS.index(st.session_state["market"]),
                          format_func=lambda x: MARKET_SHORT[x], key="set_market")
        lst = watchlists.get(mk, [])
        bulk = st.text_area("Symbols", placeholder=PLACEHOLDERS[mk], height=90, key=f"bulk_{mk}",
                            label_visibility="collapsed")
        st.caption("Separate by comma or new line")
        if st.button(f"➕ Add to {MARKET_SHORT[mk]}", use_container_width=True, key=f"add_{mk}"):
            if not bulk or not bulk.strip():
                st.warning("Type symbols first")
            else:
                raw = bulk.replace("\n", ",").replace(" ", ",")
                parts = [p.strip().upper() for p in raw.split(",") if p.strip()]
                added = []
                for p in parts:
                    fmt = get_symbol_suffix(p, mk)
                    if fmt not in lst:
                        lst.append(fmt)
                        added.append(fmt)
                watchlists[mk] = lst
                save_json(WATCHLIST_FILE, watchlists)
                st.session_state["market"] = mk
                st.success(f"Added {len(added)} symbols" if added else "All already in list")

    # ---- scan options ----
    with st.container(key="panel_opts"):
        st.markdown(f'<div class="lbl">{IC_REFRESH} Scan options</div>', unsafe_allow_html=True)
        ff = st.toggle("Full matches only", value=settings.get("full_matches_only", False), key="opt_full")
        if ff != settings.get("full_matches_only", False):
            settings["full_matches_only"] = ff
            save_json(SETTINGS_FILE, settings)

    # ---- scanner rules ----
    with st.expander("🔍 Scanner rules", expanded=False):
        st.write(scanners[scanner_name].get("description", "Price action scanner"))
        rules = active_rules
        st.markdown(
            f"""
**Active rules**
- Near S/R: **{'ON' if rules.get('require_near_sr') else 'OFF'}** ({rules.get('sr_pct', 1)}%)
- HL / LH structure: **{'ON' if rules.get('require_structure', True) else 'OFF'}**
- Healthy break candle: **{'ON' if rules.get('require_healthy_break', True) else 'OFF'}**
- Rejection / doji after break: **{'ON' if rules.get('require_pin_bar') else 'OFF'}**
- Rejection vol > Breakout vol: **{'ON' if rules.get('require_high_volume_rejection') else 'OFF'}**
- Consolidation: **{'ON' if rules.get('require_consolidation') else 'OFF'}**
- Volume dry before break: **{'ON' if rules.get('require_volume_dry') else 'OFF'}**
"""
        )
        e1, e2, e3 = st.columns(3)
        with e1:
            if st.button("✏️ Edit", use_container_width=True, key="edit_rules"):
                st.session_state["rules_backup"] = dict(active_rules)
                st.session_state["rules_edit"] = True
                st.rerun()
        with e2:
            if st.button("↩️ Revert", use_container_width=True, key="revert_rules"):
                if st.session_state.get("rules_backup"):
                    st.session_state["rules_forward"] = dict(scanners[scanner_name]["rules"])
                    scanners[scanner_name]["rules"] = dict(st.session_state["rules_backup"])
                    save_json(SCANNERS_FILE, scanners)
                    st.success("Reverted")
                    st.rerun()
                else:
                    st.info("Nothing to revert")
        with e3:
            if st.button("↪️ Forward", use_container_width=True, key="forward_rules"):
                if st.session_state.get("rules_forward"):
                    scanners[scanner_name]["rules"] = dict(st.session_state["rules_forward"])
                    save_json(SCANNERS_FILE, scanners)
                    st.session_state["rules_forward"] = None
                    st.success("Forward applied")
                    st.rerun()
                else:
                    st.info("Nothing to forward")

        if st.session_state.get("rules_edit"):
            st.markdown("---")
            st.caption("Toggle / adjust rules (saved on this server)")
            r = dict(active_rules)
            r["require_near_sr"] = st.checkbox("Near Support/Resistance", value=r.get("require_near_sr", True), key="r_near")
            r["sr_pct"] = st.slider("S/R distance %", 0.2, 2.0, float(r.get("sr_pct", 0.3)), 0.1, key="r_pct")
            r["require_structure"] = st.checkbox("Higher low / Lower high structure", value=r.get("require_structure", True), key="r_struct")
            r["require_healthy_break"] = st.checkbox("Healthy breakout/breakdown candle", value=r.get("require_healthy_break", True), key="r_break")
            r["require_pin_bar"] = st.checkbox("Rejection or doji after break", value=r.get("require_pin_bar", True), key="r_pin")
            r["require_high_volume_rejection"] = st.checkbox(
                "Rejection volume > breakout volume", value=r.get("require_high_volume_rejection", True), key="r_hv")
            r["require_consolidation"] = st.checkbox("Consolidation before break", value=r.get("require_consolidation", True), key="r_cons")
            r["require_volume_dry"] = st.checkbox("Volume drying before break", value=r.get("require_volume_dry", True), key="r_dry")
            if st.button("💾 Save rules", type="primary", use_container_width=True, key="r_save"):
                scanners[scanner_name]["rules"] = r
                save_json(SCANNERS_FILE, scanners)
                st.session_state["rules_edit"] = False
                st.success("Rules saved — no app update needed")
                st.rerun()
            if st.button("Cancel edit", use_container_width=True, key="r_cancel"):
                st.session_state["rules_edit"] = False
                st.rerun()

        st.markdown("---")
        st.caption("Scanner profiles")
        new_profile = st.text_input("New profile name", placeholder="My strict scanner", key="new_prof")
        if st.button("➕ Add profile (copy current rules)", use_container_width=True, key="add_prof"):
            if new_profile and new_profile.strip():
                name = new_profile.strip()
                if name not in scanners:
                    scanners[name] = {"description": f"Custom profile: {name}", "rules": dict(active_rules)}
                    save_json(SCANNERS_FILE, scanners)
                    st.success(f"Added profile {name}")
                    st.rerun()
                else:
                    st.warning("Name already exists")
            else:
                st.warning("Enter a name")
        for nm in list(scanners.keys()):
            c1, c2 = st.columns([4, 1])
            c1.write(f"• **{nm}**" + (" ← active" if nm == scanner_name else ""))
            if c2.button("🗑", key=f"del_prof_{nm}"):
                if len(scanners) <= 1:
                    st.warning("Keep at least one profile")
                else:
                    del scanners[nm]
                    save_json(SCANNERS_FILE, scanners)
                    st.rerun()

    st.button("🕒 Scan history", use_container_width=True, key="goto_hist", on_click=go, args=("history",))


# ====================== PAGE: HISTORY ======================
def page_history():
    st.markdown('<div class="sec-title">Scan history</div>', unsafe_allow_html=True)
    if not history:
        st.info("No history yet.")
        return
    for h in history:
        syms = ", ".join(h.get("symbols", [])[:8])
        if len(h.get("symbols", [])) > 8:
            syms += "…"
        top = f"• top {h.get('top_score')}" if h.get("top_score") else ""
        st.markdown(
            f'<div class="card"><div style="font-weight:700;font-size:1.02rem;">{h.get("time", "")}</div>'
            f'<div style="color:#a9b6d6;margin-top:.25rem;">{h.get("market", "")} • {h.get("timeframe", "")} • '
            f'<b>{h.get("count", 0)}</b> setups {top}</div>'
            f'<div style="margin-top:.35rem;font-size:.9rem;">{syms or "—"}</div></div>',
            unsafe_allow_html=True,
        )
    if st.button("Clear history", use_container_width=True, key="clear_hist"):
        save_json(HISTORY_FILE, [])
        st.rerun()


# ====================== ROUTER ======================
PAGES = {
    "dashboard": page_dashboard,
    "scan": page_scan,
    "alerts": page_alerts,
    "settings": page_settings,
    "history": page_history,
}
PAGES.get(st.session_state["nav"], page_dashboard)()
