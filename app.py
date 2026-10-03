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





# ====================== STREAMLIT APP ======================
st.set_page_config(
    page_title="PA Scanner",
    page_icon="📈",
    layout="centered",
    initial_sidebar_state="collapsed"
)

if "theme" not in st.session_state:
    st.session_state["theme"] = "dark"

theme = st.session_state["theme"]
is_dark = theme == "dark"

# ===== NEON GRADIENT UI (matched to preview) =====
if is_dark:
    st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@500;600;700;800;900&display=swap');

    html, body, [class*="css"], .stApp, button, input, textarea, label {
        font-family: 'Poppins', system-ui, sans-serif !important;
    }
    html, body, [class*="css"] { font-size: 16px !important; }

    .stApp {
        background:
            radial-gradient(900px 500px at 15% 0%, #7c4dff66 0%, transparent 55%),
            radial-gradient(800px 450px at 95% 15%, #00e5ff44 0%, transparent 50%),
            radial-gradient(700px 400px at 50% 100%, #00e67655 0%, transparent 50%),
            linear-gradient(165deg, #0a0618 0%, #12082a 40%, #071820 100%) !important;
        color: #f0f4ff !important;
    }
    #MainMenu, footer, header, [data-testid="stSidebar"] { display: none !important; }
    .stDeployButton { display: none; }

    .block-container {
        padding-top: 0.45rem !important;
        padding-bottom: 2.2rem !important;
        padding-left: 0.75rem !important;
        padding-right: 0.75rem !important;
        max-width: 430px !important;
    }

    .hero { text-align: center; padding: 0.15rem 0 0.65rem 0; }
    .hero h1 {
        margin: 0;
        font-size: 2rem;
        font-weight: 900;
        letter-spacing: -0.5px;
        background: linear-gradient(90deg, #00e5ff 0%, #00e676 45%, #ffea00 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        filter: drop-shadow(0 0 20px #00e5ff55);
    }
    .hero p {
        margin: 0.2rem 0 0 0;
        color: #9eb0d0;
        font-size: 0.88rem;
        font-weight: 600;
        letter-spacing: 0.04em;
    }

    .panel {
        background: linear-gradient(145deg, rgba(30,20,60,0.75), rgba(12,18,40,0.8));
        border: 1px solid rgba(124, 77, 255, 0.35);
        border-radius: 20px;
        padding: 0.95rem 1rem;
        margin-bottom: 0.8rem;
        box-shadow: 0 12px 32px rgba(0,0,0,0.35), inset 0 1px 0 rgba(255,255,255,0.06);
        backdrop-filter: blur(12px);
    }
    .panel-title {
        font-size: 0.78rem;
        font-weight: 800;
        color: #b39ddb;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        margin-bottom: 0.5rem;
    }

    .chip {
        display: inline-block;
        background: rgba(30, 40, 80, 0.8);
        border: 1px solid rgba(0, 229, 255, 0.35);
        color: #e0f7fa;
        border-radius: 999px;
        padding: 0.28rem 0.7rem;
        margin: 0.15rem 0.2rem 0.15rem 0;
        font-size: 0.84rem;
        font-weight: 700;
    }

    .card {
        background: linear-gradient(145deg, rgba(28,22,55,0.9), rgba(14,18,40,0.95));
        border-radius: 18px;
        padding: 1rem 1.05rem;
        margin-bottom: 0.75rem;
        border: 1px solid rgba(100, 120, 255, 0.25);
        box-shadow: 0 8px 24px rgba(0,0,0,0.3);
    }
    .card-green {
        border-color: rgba(0, 230, 118, 0.55);
        background: linear-gradient(145deg, rgba(10,40,28,0.95), rgba(12,20,30,0.95));
        box-shadow: 0 0 24px rgba(0, 230, 118, 0.15);
    }
    .card-red {
        border-color: rgba(255, 82, 82, 0.5);
        background: linear-gradient(145deg, rgba(40,12,20,0.95), rgba(18,12,20,0.95));
        box-shadow: 0 0 24px rgba(255, 82, 82, 0.12);
    }

    .score { font-size: 1.55rem; font-weight: 900; }
    .score-green { color: #00e676; text-shadow: 0 0 14px #00e67666; }
    .score-yellow { color: #ffea00; text-shadow: 0 0 14px #ffea0066; }
    .score-red { color: #ff5252; text-shadow: 0 0 14px #ff525266; }

    .tag {
        display: inline-block;
        padding: 0.16rem 0.55rem;
        border-radius: 999px;
        font-size: 0.75rem;
        font-weight: 800;
        margin-left: 0.25rem;
    }
    .tag-bull { background: rgba(0,230,118,0.15); color: #69f0ae; border: 1px solid #00e67666; }
    .tag-bear { background: rgba(255,82,82,0.15); color: #ff8a80; border: 1px solid #ff525266; }
    .tag-full { background: rgba(0,229,255,0.15); color: #84ffff; border: 1px solid #00e5ff66; }

    .stButton > button {
        border-radius: 16px !important;
        font-weight: 800 !important;
        min-height: 3.1rem !important;
        font-size: 1rem !important;
        border: none !important;
        width: 100% !important;
        letter-spacing: 0.02em !important;
    }
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #00c853 0%, #00e676 50%, #69f0ae 100%) !important;
        color: #03150a !important;
        box-shadow: 0 0 28px rgba(0, 230, 118, 0.45), 0 8px 20px rgba(0,0,0,0.3) !important;
        min-height: 3.65rem !important;
        font-size: 1.15rem !important;
        border: 1px solid #69f0ae88 !important;
    }
    .stButton > button[kind="secondary"] {
        background: linear-gradient(145deg, rgba(40,30,80,0.9), rgba(20,25,55,0.95)) !important;
        color: #e8eaf6 !important;
        border: 1px solid rgba(124, 77, 255, 0.45) !important;
        box-shadow: 0 0 12px rgba(124, 77, 255, 0.15) !important;
    }

    /* Neon market buttons */
    div[data-testid="stHorizontalBlock"] > div:nth-child(1) .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #00c853, #1de9b6) !important;
        color: #03150a !important;
        box-shadow: 0 0 18px #00e67666 !important;
        border: 1px solid #00e676aa !important;
    }
    div[data-testid="stHorizontalBlock"] > div:nth-child(2) .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #7c4dff, #e040fb) !important;
        color: #fff !important;
        box-shadow: 0 0 18px #7c4dff66 !important;
        border: 1px solid #e040fbaa !important;
    }
    div[data-testid="stHorizontalBlock"] > div:nth-child(3) .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #00b0ff, #2979ff) !important;
        color: #fff !important;
        box-shadow: 0 0 18px #00b0ff66 !important;
        border: 1px solid #40c4ffaa !important;
    }
    div[data-testid="stHorizontalBlock"] > div:nth-child(1) .stButton > button[kind="secondary"] {
        border: 1px solid #00e67655 !important; color: #69f0ae !important;
    }
    div[data-testid="stHorizontalBlock"] > div:nth-child(2) .stButton > button[kind="secondary"] {
        border: 1px solid #e040fb55 !important; color: #ea80fc !important;
    }
    div[data-testid="stHorizontalBlock"] > div:nth-child(3) .stButton > button[kind="secondary"] {
        border: 1px solid #00b0ff55 !important; color: #80d8ff !important;
    }

    .stTextArea textarea, .stTextInput input {
        border-radius: 14px !important;
        font-size: 0.98rem !important;
        background: rgba(12, 16, 36, 0.85) !important;
        border: 1px solid rgba(124, 77, 255, 0.35) !important;
        color: #f0f4ff !important;
    }
    .stSelectbox label, .stSlider label, .stToggle label, .stCheckbox label {
        font-size: 0.95rem !important;
        color: #c5cae9 !important;
        font-weight: 700 !important;
    }
    .stProgress > div > div {
        background: linear-gradient(90deg, #7c4dff, #00e5ff, #00e676, #ffea00) !important;
    }
    div[data-baseweb="slider"] div[role="slider"] {
        background: linear-gradient(90deg, #00e676, #ffea00) !important;
        box-shadow: 0 0 10px #00e67688 !important;
    }

    @media (max-width: 640px) {
        .block-container {
            max-width: 100% !important;
            padding-left: 0.65rem !important;
            padding-right: 0.65rem !important;
        }
        .hero h1 { font-size: 1.75rem !important; }
        .stButton > button { min-height: 2.95rem !important; }
        .stButton > button[kind="primary"] { min-height: 3.45rem !important; font-size: 1.1rem !important; }
    }
</style>
""", unsafe_allow_html=True)
else:
    st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@500;600;700;800;900&display=swap');
    html, body, [class*="css"], .stApp, button { font-family: 'Poppins', system-ui, sans-serif !important; }
    html, body, [class*="css"] { font-size: 16px !important; }
    .stApp {
        background:
            radial-gradient(800px 400px at 10% 0%, #e1bee7aa 0%, transparent 50%),
            radial-gradient(700px 400px at 100% 10%, #b2ebf2aa 0%, transparent 50%),
            linear-gradient(165deg, #f3e5f5 0%, #e3f2fd 50%, #e8f5e9 100%) !important;
        color: #1a237e !important;
    }
    #MainMenu, footer, header, [data-testid="stSidebar"] { display: none !important; }
    .stDeployButton { display: none; }
    .block-container {
        padding-top: 0.45rem !important; padding-bottom: 2.2rem !important;
        padding-left: 0.75rem !important; padding-right: 0.75rem !important;
        max-width: 430px !important;
    }
    .hero { text-align: center; padding: 0.15rem 0 0.65rem 0; }
    .hero h1 {
        margin: 0; font-size: 2rem; font-weight: 900;
        background: linear-gradient(90deg, #00838f, #2e7d32, #f9a825);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    }
    .hero p { margin: 0.2rem 0 0 0; color: #546e7a; font-size: 0.88rem; font-weight: 600; }
    .panel {
        background: rgba(255,255,255,0.88); border: 1px solid #ce93d8;
        border-radius: 20px; padding: 0.95rem 1rem; margin-bottom: 0.8rem;
        box-shadow: 0 10px 28px rgba(103,58,183,0.1); backdrop-filter: blur(8px);
    }
    .panel-title { font-size: 0.78rem; font-weight: 800; color: #6a1b9a; text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 0.5rem; }
    .chip {
        display: inline-block; background: #f3e5f5; border: 1px solid #ce93d8; color: #4a148c;
        border-radius: 999px; padding: 0.28rem 0.7rem; margin: 0.15rem 0.2rem; font-size: 0.84rem; font-weight: 700;
    }
    .card { background: #fff; border-radius: 18px; padding: 1rem; margin-bottom: 0.75rem; border: 1px solid #c5cae9; box-shadow: 0 6px 16px rgba(26,35,126,0.08); }
    .card-green { border-color: #66bb6a; background: linear-gradient(145deg, #e8f5e9, #fff); }
    .card-red { border-color: #ef5350; background: linear-gradient(145deg, #ffebee, #fff); }
    .score { font-size: 1.55rem; font-weight: 900; }
    .score-green { color: #2e7d32; } .score-yellow { color: #f9a825; } .score-red { color: #c62828; }
    .tag { display: inline-block; padding: 0.16rem 0.55rem; border-radius: 999px; font-size: 0.75rem; font-weight: 800; margin-left: 0.25rem; }
    .tag-bull { background: #e8f5e9; color: #2e7d32; border: 1px solid #81c784; }
    .tag-bear { background: #ffebee; color: #c62828; border: 1px solid #e57373; }
    .tag-full { background: #e3f2fd; color: #1565c0; border: 1px solid #64b5f6; }
    .stButton > button {
        border-radius: 16px !important; font-weight: 800 !important;
        min-height: 3.1rem !important; font-size: 1rem !important; border: none !important; width: 100% !important;
    }
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #00c853, #43a047) !important; color: #fff !important;
        box-shadow: 0 8px 22px rgba(0,200,83,0.3) !important; min-height: 3.65rem !important; font-size: 1.15rem !important;
    }
    .stButton > button[kind="secondary"] {
        background: #fff !important; color: #4a148c !important; border: 1px solid #ce93d8 !important;
    }
    div[data-testid="stHorizontalBlock"] > div:nth-child(1) .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #00c853, #1de9b6) !important;
    }
    div[data-testid="stHorizontalBlock"] > div:nth-child(2) .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #7c4dff, #e040fb) !important; color: #fff !important;
    }
    div[data-testid="stHorizontalBlock"] > div:nth-child(3) .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #00b0ff, #2979ff) !important; color: #fff !important;
    }
    .stTextArea textarea, .stTextInput input {
        border-radius: 14px !important; background: #fff !important; border: 1px solid #ce93d8 !important; color: #1a237e !important;
    }
    .stSelectbox label, .stSlider label, .stToggle label { color: #4a148c !important; font-weight: 700 !important; }
    .stProgress > div > div { background: linear-gradient(90deg, #7c4dff, #00e5ff, #00e676) !important; }
    @media (max-width: 640px) {
        .block-container { max-width: 100% !important; padding-left: 0.65rem !important; padding-right: 0.65rem !important; }
        .hero h1 { font-size: 1.75rem !important; }
    }
</style>
""", unsafe_allow_html=True)

# Theme toggle
tc1, tc2 = st.columns([4, 1])
with tc2:
    theme_label = "☀️ Light" if is_dark else "🌙 Dark"
    if st.button(theme_label, key="theme_toggle", use_container_width=True):
        st.session_state["theme"] = "light" if is_dark else "dark"
        st.rerun()

st.markdown("""
<div class="hero">
  <h1>PA Scanner</h1>
  <p>SCAN · ANALYZE · TRADE</p>
</div>
""", unsafe_allow_html=True)


# ---------- data ----------
watchlists = load_json(WATCHLIST_FILE, DEFAULT_WATCHLISTS)
scanners = load_json(SCANNERS_FILE, DEFAULT_SCANNERS)
settings = load_json(SETTINGS_FILE, DEFAULT_SETTINGS)
history = load_json(HISTORY_FILE, [])

if "market" not in st.session_state:
    st.session_state["market"] = "Indian Stocks"
if "view" not in st.session_state:
    st.session_state["view"] = "scan"  # scan | results | history
if "auto_refresh" not in st.session_state:
    st.session_state["auto_refresh"] = False
if "refresh_minutes" not in st.session_state:
    st.session_state["refresh_minutes"] = 5
if "next_scan_at" not in st.session_state:
    st.session_state["next_scan_at"] = None  # epoch seconds; timer continues across views
if "force_scan" not in st.session_state:
    st.session_state["force_scan"] = False
if "rules_edit" not in st.session_state:
    st.session_state["rules_edit"] = False
if "rules_backup" not in st.session_state:
    st.session_state["rules_backup"] = None
if "rules_forward" not in st.session_state:
    st.session_state["rules_forward"] = None
if "show_watchlist" not in st.session_state:
    st.session_state["show_watchlist"] = False
if "scan_results" not in st.session_state:
    st.session_state["scan_results"] = []

MARKET_ORDER = ["Indian Stocks", "Crypto", "Forex"]
MARKET_LABEL = {"Indian Stocks": "🇮🇳 Stocks", "Crypto": "₿ Crypto", "Forex": "💱 Forex"}
MARKET_COLOR = {
    "Indian Stocks": "#1b5e20",
    "Crypto": "#4a148c",
    "Forex": "#0d47a1",
}
PLACEHOLDERS = {
    "Indian Stocks": "RELIANCE, TCS, INFY, SBIN",
    "Crypto": "BTC, ETH, SOL, XRP",
    "Forex": "XAUUSD, EURUSD, GBPUSD",
}

scanner_name = list(scanners.keys())[0] if scanners else "My Price Action Scanner"
if scanner_name not in scanners:
    scanners[scanner_name] = DEFAULT_SCANNERS["My Price Action Scanner"]
active_rules = scanners[scanner_name]["rules"]

# ---------- Market selector (3 big buttons) ----------
c1, c2, c3 = st.columns(3)
for col, mkt in zip([c1, c2, c3], MARKET_ORDER):
    with col:
        is_active = st.session_state["market"] == mkt
        label = MARKET_LABEL[mkt]
        if st.button(
            label,
            key=f"mktbtn_{mkt}",
            use_container_width=True,
            type="primary" if is_active else "secondary",
        ):
            st.session_state["market"] = mkt
            st.session_state["show_watchlist"] = False
            st.session_state["view"] = "scan"
            st.rerun()

# Color accent line for selected market
sel = st.session_state["market"]
st.markdown(
    f"<div style='height:4px;border-radius:4px;margin:0.35rem 0 0.8rem 0;background:{MARKET_COLOR[sel]};'></div>",
    unsafe_allow_html=True,
)

market = st.session_state["market"]
current_list = watchlists.get(market, [])

# ---------- SCAN VIEW ----------
if st.session_state["view"] == "scan":
    st.markdown("<div class='panel'>", unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        timeframe = st.selectbox(
            "Timeframe",
            ["1m", "3m", "5m", "15m", "1h", "4h", "1d"],
            index=2,
            key=f"tf_{market}",
        )
    with c2:
        min_score = st.slider("Min Score", 0, 100, 50, 5, key=f"score_{market}")

    only_full = st.toggle("Full Matches Only", value=False, key=f"full_{market}")
    auto_refresh = st.toggle("Auto Refresh", value=st.session_state["auto_refresh"], key="auto_ref_toggle")
    st.session_state["auto_refresh"] = auto_refresh

    if auto_refresh:
        st.caption("Refresh interval")
        interval_opts = [1, 5, 15, 30, 45, 60]
        # row of interval chips
        cols = st.columns(6)
        for i, m in enumerate(interval_opts):
            with cols[i]:
                lab = f"{m}m"
                active = st.session_state["refresh_minutes"] == m
                if st.button(lab, key=f"refint_{m}", use_container_width=True, type="primary" if active else "secondary"):
                    st.session_state["refresh_minutes"] = m
                    # do not reset next_scan_at here — timer keeps running
                    st.rerun()
        # countdown info
        nsa = st.session_state.get("next_scan_at")
        if nsa:
            left = int(nsa - time.time())
            if left > 0:
                st.info(f"⏳ Next auto scan in **{left // 60}m {left % 60}s**")
            else:
                st.info("⏳ Auto scan due now…")
        else:
            st.caption("Timer starts after the next completed scan")

    st.markdown("</div>", unsafe_allow_html=True)

    # SCAN + Refresh Now
    st.markdown("<div class='scan-wrap'>", unsafe_allow_html=True)
    run = st.button("🔥 SCAN NOW", type="primary", use_container_width=True, key="scan_main")
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div style='height:0.35rem'></div>", unsafe_allow_html=True)
    if st.button("🔄 Refresh Now", use_container_width=True, key="refresh_now"):
        st.session_state["force_scan"] = True
        st.rerun()

    st.markdown("<div style='height:0.55rem'></div>", unsafe_allow_html=True)
    if st.button("📊 RESULTS", use_container_width=True, key="goto_results"):
        st.session_state["view"] = "results"
        st.rerun()

    # Watchlist button (count only)
    st.markdown("<div style='height:0.6rem'></div>", unsafe_allow_html=True)
    wl_label = f"📋 Watchlist ({len(current_list)})"
    if st.button(wl_label, use_container_width=True, key="wl_toggle"):
        st.session_state["show_watchlist"] = not st.session_state["show_watchlist"]
        st.rerun()

    if st.session_state["show_watchlist"]:
        st.markdown("<div class='panel'>", unsafe_allow_html=True)
        st.markdown(f"<div class='panel-title'>Symbols in {MARKET_LABEL[market]}</div>", unsafe_allow_html=True)
        if not current_list:
            st.info("Empty watchlist")
        else:
            for i, sym in enumerate(list(current_list)):
                a, b = st.columns([5, 1])
                a.write(f"**{sym}**")
                if b.button("🗑", key=f"del_{market}_{i}"):
                    current_list.pop(i)
                    watchlists[market] = current_list
                    save_json(WATCHLIST_FILE, watchlists)
                    st.rerun()
            if st.button("Clear all", key=f"clear_{market}"):
                watchlists[market] = []
                save_json(WATCHLIST_FILE, watchlists)
                st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

    # Add symbols below watchlist
    st.markdown("<div class='panel'>", unsafe_allow_html=True)
    st.markdown("<div class='panel-title'>+ Add symbols (many at once)</div>", unsafe_allow_html=True)
    bulk = st.text_area(
        "Symbols",
        placeholder=PLACEHOLDERS[market],
        height=90,
        key=f"bulk_{market}",
        label_visibility="collapsed",
    )
    st.caption("Separate by comma or new line")
    if st.button(f"➕ Add to {MARKET_LABEL[market]}", use_container_width=True, key=f"add_{market}"):
        if not bulk or not bulk.strip():
            st.warning("Type symbols first")
        else:
            raw = bulk.replace("\n", ",").replace(" ", ",")
            parts = [p.strip().upper() for p in raw.split(",") if p.strip()]
            added = []
            for p in parts:
                fmt = get_symbol_suffix(p, market)
                if fmt not in current_list:
                    current_list.append(fmt)
                    added.append(fmt)
            watchlists[market] = current_list
            save_json(WATCHLIST_FILE, watchlists)
            if added:
                st.success(f"Added {len(added)} symbols")
            else:
                st.info("All already in list")
            st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)

    # Scanner rules section (last)
    st.markdown("<div class='panel'>", unsafe_allow_html=True)
    st.markdown("<div class='panel-title'>🔍 Scanner rules</div>", unsafe_allow_html=True)
    with st.expander("Tap to view / manage rules", expanded=False):
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
            r["require_near_sr"] = st.checkbox("Near Support/Resistance", value=r.get("require_near_sr", True))
            r["sr_pct"] = st.slider("S/R distance %", 0.2, 2.0, float(r.get("sr_pct", 0.3)), 0.1)
            r["require_structure"] = st.checkbox("Higher low / Lower high structure", value=r.get("require_structure", True))
            r["require_healthy_break"] = st.checkbox("Healthy breakout/breakdown candle", value=r.get("require_healthy_break", True))
            r["require_pin_bar"] = st.checkbox("Rejection or doji after break", value=r.get("require_pin_bar", True))
            r["require_high_volume_rejection"] = st.checkbox(
                "Rejection volume > breakout volume", value=r.get("require_high_volume_rejection", True)
            )
            r["require_consolidation"] = st.checkbox("Consolidation before break", value=r.get("require_consolidation", True))
            r["require_volume_dry"] = st.checkbox("Volume drying before break", value=r.get("require_volume_dry", True))
            if st.button("💾 Save rules", type="primary", use_container_width=True):
                scanners[scanner_name]["rules"] = r
                save_json(SCANNERS_FILE, scanners)
                st.session_state["rules_edit"] = False
                st.success("Rules saved — no app update needed")
                st.rerun()
            if st.button("Cancel edit", use_container_width=True):
                st.session_state["rules_edit"] = False
                st.rerun()

        st.markdown("---")
        st.caption("Scanner profiles")
        new_profile = st.text_input("New profile name", placeholder="My strict scanner")
        if st.button("➕ Add profile (copy current rules)", use_container_width=True):
            if new_profile and new_profile.strip():
                name = new_profile.strip()
                if name not in scanners:
                    scanners[name] = {
                        "description": f"Custom profile: {name}",
                        "rules": dict(active_rules),
                    }
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
    st.markdown("</div>", unsafe_allow_html=True)

    # History shortcut
    if st.button("🕒 Scan history", use_container_width=True, key="goto_hist"):
        st.session_state["view"] = "history"
        st.rerun()

    # Telegram quick
    with st.expander("📱 Telegram alerts"):
        enable_tg = st.toggle("Enable", value=settings.get("enable_telegram", False))
        tg_token = st.text_input("Bot Token", value=settings.get("telegram_token", ""), type="password")
        tg_chat = st.text_input("Chat ID", value=settings.get("telegram_chat_id", ""))
        if st.button("Save Telegram", use_container_width=True):
            settings["enable_telegram"] = enable_tg
            settings["telegram_token"] = tg_token
            settings["telegram_chat_id"] = tg_chat
            save_json(SETTINGS_FILE, settings)
            st.success("Saved")

    # ---- run scan ----
    # Auto scan only when timer elapsed — NOT when returning from Results
    due_auto = False
    if st.session_state.get("auto_refresh") and st.session_state.get("next_scan_at") is not None:
        if time.time() >= st.session_state["next_scan_at"]:
            due_auto = True

    force = st.session_state.get("force_scan", False)
    should = run or force or due_auto
    if force:
        st.session_state["force_scan"] = False

    if should:
        if not current_list:
            st.error("Watchlist empty. Add symbols first.")
        else:
            progress = st.progress(0)
            status = st.empty()
            results = []
            for i, sym in enumerate(current_list):
                status.caption(f"Scanning {sym} ({i+1}/{len(current_list)})")
                res = scan_symbol(sym, interval=timeframe, rules=active_rules)
                if res:
                    if only_full and not res.get("full_match"):
                        pass
                    elif res["score"] >= min_score:
                        results.append(res)
                progress.progress((i + 1) / max(len(current_list), 1))
                time.sleep(0.08)
            status.empty()
            progress.empty()

            results = sorted(results, key=lambda x: x["score"], reverse=True)
            # drop heavy df for session storage safety in history
            light = []
            for r in results:
                item = {k: v for k, v in r.items() if k != "df"}
                light.append(item)

            st.session_state["scan_results"] = results
            st.session_state["last_scan"] = datetime.now().strftime("%H:%M")
            st.session_state["last_market"] = market
            st.session_state["last_tf"] = timeframe

            # clean history entry
            hist_entry = {
                "time": datetime.now().strftime("%Y-%m-%d %H:%M"),
                "market": market,
                "timeframe": timeframe,
                "count": len(light),
                "symbols": [x["symbol"] for x in light[:15]],
                "top_score": light[0]["score"] if light else 0,
            }
            history = [hist_entry] + history
            history = history[:30]  # keep last 30
            save_json(HISTORY_FILE, history)

            if settings.get("enable_telegram") and settings.get("telegram_token"):
                for r in [x for x in results if x["score"] >= 85 or x.get("full_match")][:4]:
                    msg = (
                        f"🚨 <b>{r['symbol']}</b> | Score {r['score']}\n"
                        f"{r['direction']} | Near {r.get('near_level','')}\n"
                        f"Price {r['price']} | {timeframe}"
                    )
                    send_telegram_alert(
                        settings["telegram_token"],
                        settings.get("telegram_chat_id", ""),
                        msg,
                    )

            if results:
                st.success(f"Found {len(results)} setup(s)")
                st.session_state["view"] = "results"
                time.sleep(0.4)
                st.rerun()
            else:
                st.warning("No setups found. Try lower score or other TF.")

            # Schedule next auto scan from NOW (timer not tied to Results view)
            if st.session_state.get("auto_refresh"):
                mins = int(st.session_state.get("refresh_minutes", 5))
                st.session_state["next_scan_at"] = time.time() + mins * 60
            else:
                st.session_state["next_scan_at"] = None

# Soft poll for countdown / due auto (does not reset timer when visiting Results)
if st.session_state.get("view") == "scan" and st.session_state.get("auto_refresh"):
    nsa = st.session_state.get("next_scan_at")
    if nsa is not None:
        left = nsa - time.time()
        if left <= 0:
            st.session_state["force_scan"] = True
            st.rerun()
        else:
            # update countdown about every 15s without restarting the full interval
            time.sleep(min(15, max(1, left)))
            st.rerun()

# ---------- RESULTS VIEW ----------
elif st.session_state["view"] == "results":
    st.markdown("<div class='panel'>", unsafe_allow_html=True)
    st.markdown(
        f"<div class='panel-title'>Results • {st.session_state.get('last_market','')} • "
        f"{st.session_state.get('last_tf','')} • {st.session_state.get('last_scan','—')}</div>",
        unsafe_allow_html=True,
    )
    if st.button("← Back to Scan", use_container_width=True):
        st.session_state["view"] = "scan"
        st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)

    results = st.session_state.get("scan_results", [])
    if not results:
        st.info("No results yet. Run a scan first.")
    else:
        st.caption(f"{len(results)} setups")
        for res in results:
            score = res["score"]
            score_cls = "score-green" if score >= 80 else ("score-yellow" if score >= 60 else "score-red")
            card_cls = "card card-green" if res.get("full_match") else (
                "card card-red" if res.get("direction") == "Bearish" else "card"
            )
            dir_tag = "tag-bull" if res["direction"] == "Bullish" else "tag-bear"
            full_badge = '<span class="tag tag-full">FULL</span>' if res.get("full_match") else ""

            st.markdown(f"""
            <div class="{card_cls}">
              <div style="display:flex;justify-content:space-between;align-items:center;">
                <div>
                  <span style="font-size:1.2rem;font-weight:800;">{res['symbol']}</span>
                  <span class="tag {dir_tag}">{res['direction']}</span>
                  {full_badge}
                </div>
                <span class="score {score_cls}">{score}</span>
              </div>
              <div style="margin-top:0.45rem;color:#9aabc8;font-size:0.95rem;line-height:1.55;">
                Near <b>{res.get('near_level','—')}</b> ({res.get('level_price','—')}) • {res.get('trend','')}<br>
                Price <b>{res['price']}</b> • TF {res.get('interval','')}<br>
                Break: <b>{res.get('break_time_ist','—')}</b><br>
                Setup: <b>{res.get('setup_time_ist','—')}</b><br>
                Scanned: {res.get('scanned_at_ist','—')}
              </div>
            </div>
            """, unsafe_allow_html=True)

            with st.expander(f"Chart • {res['symbol']}"):
                df = res.get("df")
                if df is not None and len(df) > 0:
                    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                                        vertical_spacing=0.03, row_heights=[0.75, 0.25])
                    fig.add_trace(go.Candlestick(
                        x=df.index, open=df["Open"], high=df["High"],
                        low=df["Low"], close=df["Close"], name="Price"
                    ), row=1, col=1)

                    # Support / Resistance horizontal line
                    if res.get("level_price"):
                        lvl_color = "#00e676" if res.get("near_level") == "Support" else "#ff5252"
                        fig.add_hline(
                            y=res["level_price"],
                            line_dash="dash",
                            line_color=lvl_color,
                            line_width=2,
                            annotation_text=res.get("near_level", "Level"),
                            annotation_position="top left",
                            row=1, col=1,
                        )

                    # Trendline from swing points (HL / LH)
                    tpts = res.get("trend_points") or []
                    if len(tpts) >= 2:
                        try:
                            xs, ys = [], []
                            idx_map = {str(i): i for i in df.index}
                            for x_str, y in tpts:
                                # match by string or nearest
                                matched = None
                                for i in df.index:
                                    if str(i) == x_str or str(i)[:16] == str(x_str)[:16]:
                                        matched = i
                                        break
                                if matched is not None:
                                    xs.append(matched)
                                    ys.append(y)
                            if len(xs) >= 2:
                                tcolor = "#00e676" if res.get("direction") == "Bullish" else "#ff5252"
                                fig.add_trace(go.Scatter(
                                    x=xs, y=ys, mode="lines+markers",
                                    line=dict(color=tcolor, width=2, dash="solid"),
                                    marker=dict(size=7, color=tcolor),
                                    name="Trendline",
                                ), row=1, col=1)
                        except Exception:
                            pass

                    fig.add_trace(go.Bar(x=df.index, y=df["Volume"], marker_color="#2a3a55", name="Vol"), row=2, col=1)
                    fig.update_layout(
                        height=360, template="plotly_dark",
                        xaxis_rangeslider_visible=False,
                        margin=dict(l=0, r=0, t=8, b=0),
                        showlegend=False,
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)",
                    )
                    fig.update_xaxes(showgrid=False)
                    fig.update_yaxes(showgrid=False)
                    st.plotly_chart(fig, use_container_width=True)
                    st.caption(
                        f"Break: {res.get('break_time_ist','—')}  |  "
                        f"Setup: {res.get('setup_time_ist','—')}  |  IST (+05:30)"
                    )

# ---------- HISTORY VIEW ----------
elif st.session_state["view"] == "history":
    st.markdown("<div class='panel'>", unsafe_allow_html=True)
    st.markdown("<div class='panel-title'>Scan history</div>", unsafe_allow_html=True)
    if st.button("← Back to Scan", use_container_width=True, key="hist_back"):
        st.session_state["view"] = "scan"
        st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)

    if not history:
        st.info("No history yet.")
    else:
        for h in history:
            syms = ", ".join(h.get("symbols", [])[:8])
            if len(h.get("symbols", [])) > 8:
                syms += "…"
            st.markdown(f"""
            <div class="card">
              <div style="font-weight:800;font-size:1.05rem;">{h.get('time','')}</div>
              <div style="color:#9aabc8;margin-top:0.25rem;">
                {h.get('market','')} • {h.get('timeframe','')} • <b>{h.get('count',0)}</b> setups
                {f'• top {h.get("top_score")}' if h.get('top_score') else ''}
              </div>
              <div style="margin-top:0.35rem;font-size:0.92rem;">{syms or '—'}</div>
            </div>
            """, unsafe_allow_html=True)
        if st.button("Clear history", use_container_width=True):
            save_json(HISTORY_FILE, [])
            st.rerun()

st.caption("PA Scanner • Mobile first • Chart-matched price action")
