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
# NOTE: everything ABOVE this line (scanner logic, rules, helpers) is unchanged.
# Only the look / layout below was redesigned.  Recommended: streamlit>=1.39
st.set_page_config(
    page_title="PA Scanner",
    page_icon="📈",
    layout="centered",
    initial_sidebar_state="collapsed"
)

if "theme" not in st.session_state:
    st.session_state["theme"] = "dark"
if "page" not in st.session_state:
    st.session_state["page"] = "dashboard"   # dashboard | rules | results | alerts | history

is_dark = st.session_state["theme"] == "dark"

try:
    _ST_VER = tuple(int(x) for x in st.__version__.split(".")[:2])
except Exception:
    _ST_VER = (1, 0)
MODERN = _ST_VER >= (1, 39)   # keyed containers + material icons


def kc(key):
    """Keyed container (lets CSS target it). Falls back safely on old Streamlit."""
    try:
        return st.container(key=key)
    except TypeError:
        return st.container()


def go_page(p):
    st.session_state["page"] = p


def toggle_theme():
    st.session_state["theme"] = "light" if st.session_state.get("theme") == "dark" else "dark"


def toggle_flag(name):
    st.session_state[name] = not st.session_state.get(name, False)


# ====================== STYLE ======================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@500;600;700;800;900&family=Rajdhani:wght@600;700&display=swap');

html, body, .stApp, button, input, label, textarea {
  font-family: 'Poppins', system-ui, sans-serif !important;
}
/* never break the icon font */
[data-testid="stIconMaterial"], .material-icons, span[class*="material"] {
  font-family: "Material Symbols Rounded", "Material Icons" !important;
}
html, body { font-size: 15.5px !important; }

.stApp {
  background:
    radial-gradient(900px 520px at 10% -5%, #7c4dff77 0%, transparent 55%),
    radial-gradient(800px 480px at 100% 10%, #00e5ff55 0%, transparent 50%),
    radial-gradient(700px 420px at 50% 110%, #00e67655 0%, transparent 55%),
    linear-gradient(165deg, #070414 0%, #12082a 45%, #061820 100%) !important;
  color: #f0f4ff !important;
}
#MainMenu, footer, header, [data-testid="stSidebar"], .stDeployButton { display: none !important; }

.block-container {
  padding-top: 0.6rem !important;
  padding-bottom: 14rem !important;      /* room so content never hides behind the nav */
  padding-left: 0.8rem !important;
  padding-right: 0.8rem !important;
  max-width: 430px !important;
}
[data-testid="stMain"] [data-testid="stVerticalBlock"] { gap: 0.8rem; }

/* ---------- keep these rows side-by-side on phones ---------- */
.st-key-hdr [data-testid="stHorizontalBlock"],
.st-key-tf_row [data-testid="stHorizontalBlock"],
.st-key-ar_row [data-testid="stHorizontalBlock"],
.st-key-wl_head [data-testid="stHorizontalBlock"],
.st-key-bottomnav [data-testid="stHorizontalBlock"] {
  flex-wrap: nowrap !important; align-items: center !important; gap: 0.5rem !important;
}
.st-key-hdr [data-testid="stColumn"], .st-key-hdr [data-testid="column"],
.st-key-tf_row [data-testid="stColumn"], .st-key-tf_row [data-testid="column"],
.st-key-ar_row [data-testid="stColumn"], .st-key-ar_row [data-testid="column"],
.st-key-wl_head [data-testid="stColumn"], .st-key-wl_head [data-testid="column"],
.st-key-bottomnav [data-testid="stColumn"], .st-key-bottomnav [data-testid="column"] {
  min-width: 0 !important;
}

/* ---------- header ---------- */
.app-title {
  text-align: center; margin: 0; font-size: 1.75rem; font-weight: 900; letter-spacing: -0.4px;
  background: linear-gradient(90deg, #00e5ff, #00e676, #ffea00);
  -webkit-background-clip: text; -webkit-text-fill-color: transparent;
  filter: drop-shadow(0 0 16px #00e5ff55); font-style: italic;
}
.st-key-hdr_menu button, .st-key-theme_toggle button {
  width: 3rem !important; min-height: 3rem !important; height: 3rem !important;
  padding: 0 !important; border-radius: 16px !important;
  background: rgba(40,40,110,0.55) !important;
  border: 1px solid rgba(150,130,255,0.45) !important;
  color: #80deea !important; font-size: 1.3rem !important;
}

/* ---------- glass panels ---------- */
[class*="st-key-panel_"] {
  box-sizing: border-box;
  background: linear-gradient(145deg, rgba(48,34,92,0.62), rgba(14,22,48,0.80));
  border: 1px solid rgba(150,120,255,0.32);
  border-radius: 26px;
  padding: 1rem 1.05rem !important;
  box-shadow: 0 14px 34px rgba(0,0,0,0.38), inset 0 1px 0 rgba(255,255,255,0.07);
  backdrop-filter: blur(14px);
}
.ptitle {
  font-family: 'Rajdhani', 'Poppins', sans-serif; font-weight: 700; font-size: 1.05rem;
  letter-spacing: 0.14em; text-transform: uppercase; color: #b39ddb;
}
.page-title {
  font-family: 'Rajdhani', 'Poppins', sans-serif; font-weight: 700; font-size: 1.5rem;
  letter-spacing: 0.06em;
  background: linear-gradient(90deg, #00e5ff, #00e676, #ffea00);
  -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}
.page-sub { color: #90a4c8; font-size: 0.8rem; margin-top: -0.1rem; }

/* ---------- market pills (Stocks / Crypto / Forex) ---------- */
.st-key-market_wrap div[role="radiogroup"] {
  display: flex !important; flex-direction: row !important; flex-wrap: nowrap !important;
  gap: 0.5rem !important; width: 100% !important;
}
.st-key-market_wrap div[role="radiogroup"] > label {
  flex: 1 1 0 !important; margin: 0 !important; justify-content: center !important;
  padding: 0.8rem 0.2rem !important; border-radius: 22px !important;
  background: rgba(18,14,40,0.85) !important; border: 1.5px solid rgba(255,255,255,0.15) !important;
  transition: all 0.25s ease;
}
/* hide the little radio circle */
.st-key-market_wrap div[role="radiogroup"] > label > div:first-child:not(:has([data-testid="stMarkdownContainer"])) {
  display: none !important;
}
.st-key-market_wrap div[role="radiogroup"] > label p {
  font-size: 1rem !important; font-weight: 800 !important; margin: 0 !important; text-align: center;
}
/* not selected -> faded + blurred */
.st-key-market_wrap div[role="radiogroup"] > label:not(:has(input:checked)) {
  opacity: 0.42; filter: blur(1.3px) saturate(0.55); transform: scale(0.96);
}
/* selected -> bright + glow */
.st-key-market_wrap div[role="radiogroup"] > label:has(input:checked) {
  opacity: 1; filter: none; transform: scale(1.04);
}
.st-key-market_wrap div[role="radiogroup"] > label:nth-child(1) { border-color: rgba(0,230,118,0.55) !important; }
.st-key-market_wrap div[role="radiogroup"] > label:nth-child(2) { border-color: rgba(224,64,251,0.55) !important; }
.st-key-market_wrap div[role="radiogroup"] > label:nth-child(3) { border-color: rgba(0,176,255,0.55) !important; }
.st-key-market_wrap div[role="radiogroup"] > label:nth-child(1):has(input:checked) {
  background: rgba(0,60,40,0.65) !important; border-color: #00e676 !important;
  box-shadow: 0 0 22px rgba(0,230,118,0.55), inset 0 0 14px rgba(0,230,118,0.18) !important;
}
.st-key-market_wrap div[role="radiogroup"] > label:nth-child(1):has(input:checked) p { color: #69f0ae !important; }
.st-key-market_wrap div[role="radiogroup"] > label:nth-child(2):has(input:checked) {
  background: rgba(70,12,80,0.65) !important; border-color: #e040fb !important;
  box-shadow: 0 0 22px rgba(224,64,251,0.55), inset 0 0 14px rgba(224,64,251,0.18) !important;
}
.st-key-market_wrap div[role="radiogroup"] > label:nth-child(2):has(input:checked) p { color: #ea80fc !important; }
.st-key-market_wrap div[role="radiogroup"] > label:nth-child(3):has(input:checked) {
  background: rgba(10,34,80,0.65) !important; border-color: #00b0ff !important;
  box-shadow: 0 0 22px rgba(0,176,255,0.55), inset 0 0 14px rgba(0,176,255,0.18) !important;
}
.st-key-market_wrap div[role="radiogroup"] > label:nth-child(3):has(input:checked) p { color: #80d8ff !important; }

/* ---------- timeframe / auto refresh labels ---------- */
.lbl-tf {
  font-family: 'Rajdhani', 'Poppins', sans-serif; font-weight: 700; font-size: 1.55rem;
  letter-spacing: 0.08em; text-transform: uppercase; white-space: nowrap; line-height: 1.1;
  background: linear-gradient(90deg, #00e5ff 0%, #00e676 55%, #ffea00 100%);
  -webkit-background-clip: text; -webkit-text-fill-color: transparent;
  filter: drop-shadow(0 0 10px rgba(0,229,255,0.35));
}
.lbl-ar {
  font-family: 'Rajdhani', 'Poppins', sans-serif; font-weight: 700; font-size: 1.1rem;
  letter-spacing: 0.07em; text-transform: uppercase; white-space: nowrap; line-height: 1.1;
  background: linear-gradient(90deg, #ea80fc 0%, #ff9100 100%);
  -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}
.ms-row { display: flex; justify-content: space-between; align-items: center; }
.lbl-ms {
  font-family: 'Rajdhani', 'Poppins', sans-serif; font-weight: 700; font-size: 1.25rem;
  letter-spacing: 0.08em; text-transform: uppercase; color: #c5b6ff;
}
.lbl-ms .info { color: #8a9bb8; font-size: 1rem; cursor: help; }
.ms-val { font-size: 1.5rem; font-weight: 800; color: #fff; }
.ms-val .star { color: #ffd54f; text-shadow: 0 0 10px rgba(255,213,79,0.7); }

/* ---------- dropdowns ---------- */
[data-testid="stSelectbox"] div[data-baseweb="select"] > div {
  background: rgba(20,16,50,0.92) !important;
  border: 1.5px solid rgba(0,229,255,0.55) !important;
  border-radius: 16px !important; min-height: 3.1rem;
  box-shadow: 0 0 14px rgba(0,229,255,0.18);
}
[data-testid="stSelectbox"] div[data-baseweb="select"] { font-size: 1.25rem; font-weight: 800; }
[data-testid="stSelectbox"] div[data-baseweb="select"] * { font-size: inherit; font-weight: inherit; }
[data-testid="stSelectbox"] div[data-baseweb="select"] div { color: #e0f7fa; }
/* auto refresh dropdown = same style, a bit smaller */
.st-key-ar_row [data-testid="stSelectbox"] div[data-baseweb="select"] > div {
  min-height: 2.4rem; border-radius: 13px !important; border-color: rgba(234,128,252,0.6) !important;
  box-shadow: 0 0 12px rgba(234,128,252,0.18);
}
.st-key-ar_row [data-testid="stSelectbox"] div[data-baseweb="select"] { font-size: 1rem; }

/* ---------- min-score slider ---------- */
[data-testid="stSlider"] [role="slider"] {
  width: 28px !important; height: 28px !important; border-radius: 50% !important;
  background: radial-gradient(circle at 35% 30%, #fff8d6 0%, #ffd54f 45%, #ff9100 100%) !important;
  border: 2px solid rgba(255,255,255,0.75) !important;
  box-shadow: 0 0 18px 5px rgba(255,193,7,0.6) !important;
}
[data-testid="stThumbValue"], [data-testid="stSliderThumbValue"] { display: none !important; }
[data-testid="stTickBarMin"], [data-testid="stTickBarMax"],
[data-testid="stSliderTickBarMin"], [data-testid="stSliderTickBarMax"] {
  color: #8a9bb8 !important; font-size: 0.85rem !important;
}
[data-testid="stSlider"] div[data-baseweb="slider"] > div:first-child > div:first-child,
[data-testid="stSlider"] div[data-baseweb="slider"] div:has(+ div[role="slider"]) {
  height: 12px !important; border-radius: 999px !important;
}

/* ---------- toggle ---------- */
.stToggle label, [data-testid="stToggle"] label { color: #c5cae9 !important; font-weight: 700 !important; }

/* ---------- progress line (same look as the score line) ---------- */
.stProgress { height: 1.2rem !important; }
.stProgress > div {
  height: 1.2rem !important; border-radius: 999px !important; overflow: visible !important;
  background: rgba(255,255,255,0.12) !important;
}
.stProgress > div > div {
  height: 1.2rem !important; border-radius: 999px !important; overflow: visible !important;
  background: linear-gradient(90deg, #00e676, #c6ff00, #ffea00, #ff9100) !important;
  position: relative;
}
.stProgress > div > div::after {
  content: ""; position: absolute; right: -4px; top: 50%; transform: translateY(-50%);
  width: 26px; height: 26px; border-radius: 50%;
  background: radial-gradient(circle at 35% 30%, #fff8d6 0%, #ffd54f 45%, #ff9100 100%);
  border: 2px solid rgba(255,255,255,0.75); box-shadow: 0 0 18px 5px rgba(255,193,7,0.6);
}

/* ---------- generic buttons ---------- */
.stButton > button {
  border-radius: 16px !important; font-weight: 800 !important;
  min-height: 2.9rem !important; font-size: 0.95rem !important; width: 100% !important;
  background: rgba(30,24,60,0.85) !important; color: #e8eaf6 !important;
  border: 1px solid rgba(140,100,255,0.4) !important;
}

/* ---------- SCAN NOW ---------- */
.st-key-scan_main.st-key-scan_main button {
  position: relative; min-height: 4.8rem !important; border-radius: 30px !important;
  background: linear-gradient(135deg, #00c853 0%, #64dd17 55%, #aeea00 100%) !important;
  border: 1.5px solid rgba(185,246,202,0.95) !important;
  box-shadow: 0 0 36px rgba(100,221,23,0.55), inset 0 2px 0 rgba(255,255,255,0.4),
              inset 0 -6px 14px rgba(0,120,40,0.25) !important;
  color: #ffffff !important;
}
.st-key-scan_main.st-key-scan_main button p {
  font-size: 2.1rem !important; font-weight: 900 !important; letter-spacing: 0.04em;
  color: #ffffff !important; text-shadow: 0 2px 8px rgba(0,60,20,0.45);
  padding-right: 3.4rem;
}
.st-key-scan_main.st-key-scan_main button::after {
  content: "🎯"; position: absolute; right: 18px; top: 50%; transform: translateY(-50%);
  width: 54px; height: 54px; border-radius: 50%;
  border: 2px solid rgba(255,255,255,0.6); background: rgba(255,255,255,0.18);
  display: flex; align-items: center; justify-content: center; font-size: 1.7rem;
}
.st-key-refresh_now button {
  min-height: 2.3rem !important; font-size: 0.82rem !important; border-radius: 14px !important;
}

/* ---------- watchlist ---------- */
.lbl-wl {
  font-family: 'Rajdhani', 'Poppins', sans-serif; font-weight: 700; font-size: 1.2rem;
  letter-spacing: 0.1em; text-transform: uppercase; color: #d1c4e9; white-space: nowrap;
}
.lbl-wl .star { color: #ffd54f; }
.lbl-wl .cnt { color: #8a9bb8; font-size: 0.85rem; letter-spacing: 0; }
.st-key-wl_add_btn button {
  width: 2.5rem !important; min-height: 2.5rem !important; height: 2.5rem !important;
  padding: 0 !important; border-radius: 50% !important; font-size: 1.5rem !important;
  background: linear-gradient(135deg, #00e676, #aeea00) !important; color: #03150a !important;
  border: none !important; box-shadow: 0 0 16px rgba(0,230,118,0.55) !important;
}
.st-key-wl_toggle button {
  background: transparent !important; border: none !important; color: #e0e6ff !important;
  min-height: 2.5rem !important; font-size: 0.9rem !important; white-space: nowrap;
}
.wl-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 0.4rem; }
.wl-tile {
  min-width: 0; background: rgba(20,24,60,0.75);
  border: 1px solid rgba(120,140,255,0.32); border-radius: 16px; padding: 0.55rem 0.4rem;
}
.wl-top { display: flex; align-items: center; gap: 0.2rem; }
.wl-ico { font-size: 0.85rem; font-weight: 900; }
.wl-sym { font-weight: 800; font-size: 0.7rem; color: #fff; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.wl-cat { font-size: 0.64rem; color: #90a4c8; margin: 0.25rem 0 0.15rem; }
.wl-price { font-weight: 800; font-size: 0.82rem; color: #fff; white-space: nowrap; }
.wl-chg { font-weight: 700; font-size: 0.72rem; }
.up { color: #4ade80; } .down { color: #ff6b6b; } .flat { color: #90a4c8; }

/* ---------- result cards ---------- */
.card {
  background: linear-gradient(145deg, rgba(28,22,55,0.92), rgba(14,18,40,0.95));
  border-radius: 18px; padding: 0.9rem 1rem; margin-bottom: 0.65rem;
  border: 1px solid rgba(100,120,255,0.25);
}
.card-green { border-color: rgba(0,230,118,0.5); box-shadow: 0 0 20px rgba(0,230,118,0.12); }
.card-red { border-color: rgba(255,82,82,0.45); box-shadow: 0 0 20px rgba(255,82,82,0.1); }
.score { font-size: 1.45rem; font-weight: 900; }
.score-green { color: #00e676; text-shadow: 0 0 12px #00e67666; }
.score-yellow { color: #ffea00; }
.score-red { color: #ff5252; }
.tag { display: inline-block; padding: 0.14rem 0.5rem; border-radius: 999px; font-size: 0.72rem; font-weight: 800; margin-left: 0.2rem; }
.tag-bull { background: rgba(0,230,118,0.15); color: #69f0ae; border: 1px solid #00e67655; }
.tag-bear { background: rgba(255,82,82,0.15); color: #ff8a80; border: 1px solid #ff525255; }
.tag-full { background: rgba(0,229,255,0.15); color: #84ffff; border: 1px solid #00e5ff55; }
.chip {
  display: inline-block; padding: 0.15rem 0.55rem; margin: 0.15rem 0.2rem 0 0; border-radius: 999px;
  font-size: 0.74rem; font-weight: 700; color: #b2ebf2;
  background: rgba(0,229,255,0.10); border: 1px solid rgba(0,229,255,0.3);
}

/* ---------- text inputs ---------- */
.stTextArea textarea, .stTextInput input {
  border-radius: 14px !important; background: rgba(12,16,36,0.85) !important;
  border: 1px solid rgba(124,77,255,0.35) !important; color: #f0f4ff !important;
}
.stSelectbox label, .stSlider label, .stCheckbox label { color: #c5cae9 !important; font-weight: 700 !important; }

/* ---------- BOTTOM NAV (floats above Streamlit's "Manage app" button) ---------- */
.st-key-bottomnav {
  position: fixed !important; left: 50%; transform: translateX(-50%);
  bottom: calc(66px + env(safe-area-inset-bottom, 0px));   /* <-- raise / lower the nav here */
  width: min(412px, calc(100vw - 18px)) !important; z-index: 1000;
  box-sizing: border-box;
  background: linear-gradient(180deg, rgba(24,32,78,0.92), rgba(10,14,36,0.97));
  border: 1px solid rgba(124,77,255,0.38); border-radius: 32px;
  padding: 0.4rem 0.35rem !important;
  backdrop-filter: blur(18px);
  box-shadow: 0 10px 30px rgba(0,0,0,0.55), 0 0 26px rgba(0,230,118,0.12);
}
.st-key-bottomnav .stButton > button, .st-key-bottomnav button {
  background: transparent !important; border: none !important; box-shadow: none !important;
  color: #8a9bb8 !important; min-height: 3.5rem !important; padding: 0.1rem 0 !important;
  border-radius: 18px !important;
}
.st-key-bottomnav button p {
  font-size: 0.68rem !important; font-weight: 700 !important; line-height: 1.2 !important;
  text-align: center; margin: 0 !important; color: inherit !important;
}
.st-key-bottomnav [data-testid="stIconMaterial"] { font-size: 1.55rem !important; }
.st-key-bottomnav .stButton { display: flex; justify-content: center; }
.st-key-nav_results button {
  width: 66px !important; height: 66px !important; min-height: 66px !important;
  border-radius: 50% !important; margin: -38px auto 0 auto !important;
  background: radial-gradient(circle at 30% 28%, #3ddc97 0%, #0a5d3b 72%) !important;
  border: 3px solid #69f0ae !important; color: #ffffff !important;
  box-shadow: 0 0 28px rgba(0,230,118,0.65), 0 6px 14px rgba(0,0,0,0.4) !important;
}
.st-key-nav_results [data-testid="stIconMaterial"] { font-size: 2rem !important; }

@media (max-width: 640px) {
  .block-container { max-width: 100% !important; }
}
</style>
""", unsafe_allow_html=True)

# ====================== SHARED UI HELPERS ======================
TF_OPTIONS = ["1m", "3m", "5m", "15m", "1h", "4h", "1d"]
AR_OPTIONS = ["Off"] + TF_OPTIONS
AR_MINUTES = {"1m": 1, "3m": 3, "5m": 5, "15m": 15, "1h": 60, "4h": 240, "1d": 1440}


@st.cache_data(ttl=300, show_spinner=False)
def get_quote(symbol):
    """Last close + % change vs previous close (only used for the watchlist tiles)."""
    try:
        h = yf.Ticker(symbol).history(period="5d", interval="1d").dropna()
        if len(h) >= 2:
            last = float(h["Close"].iloc[-1])
            prev = float(h["Close"].iloc[-2])
            return last, (last - prev) / prev * 100 if prev else None
        if len(h) == 1:
            return float(h["Close"].iloc[-1]), None
    except Exception:
        pass
    return None, None


def fmt_price(p):
    if p is None:
        return "—"
    return f"{p:,.4f}" if p < 10 else f"{p:,.2f}"


def page_title(title, sub=""):
    sub_html = f'<div class="page-sub">{sub}</div>' if sub else ""
    st.markdown(f'<div class="page-title">{title}</div>{sub_html}', unsafe_allow_html=True)


def result_card_html(res, detailed=True):
    score = res["score"]
    score_cls = "score-green" if score >= 80 else ("score-yellow" if score >= 60 else "score-red")
    card_cls = "card card-green" if res.get("full_match") else (
        "card card-red" if res.get("direction") == "Bearish" else "card"
    )
    dir_tag = "tag-bull" if res.get("direction") == "Bullish" else "tag-bear"
    full_badge = '<span class="tag tag-full">FULL</span>' if res.get("full_match") else ""
    if detailed:
        body = (
            f"Near <b>{res.get('near_level','—')}</b> ({res.get('level_price','—')}) • {res.get('trend','')}<br>"
            f"Price <b>{res.get('price','—')}</b> • TF {res.get('interval','')}<br>"
            f"Break: <b>{res.get('break_time_ist','—')}</b><br>"
            f"Setup: <b>{res.get('setup_time_ist','—')}</b><br>"
            f"Scanned: {res.get('scanned_at_ist','—')}"
        )
    else:
        body = (
            f"Near <b>{res.get('near_level','—')}</b> ({res.get('level_price','—')}) • {res.get('trend','')}<br>"
            f"Price <b>{res.get('price','—')}</b><br>"
            f"Break: <b>{res.get('break_time_ist','—')}</b> | Setup: <b>{res.get('setup_time_ist','—')}</b>"
        )
    return (
        f'<div class="{card_cls}">'
        f'<div style="display:flex;justify-content:space-between;align-items:center;">'
        f'<div><span style="font-size:1.15rem;font-weight:800;">{res.get("symbol","")}</span>'
        f'<span class="tag {dir_tag}">{res.get("direction","")}</span>{full_badge}</div>'
        f'<span class="score {score_cls}">{score}</span></div>'
        f'<div style="margin-top:0.45rem;color:#9aabc8;font-size:0.92rem;line-height:1.55;">{body}</div>'
        f'</div>'
    )


def render_chart(res):
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
                for x_str, y in tpts:
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


# ====================== DATA ======================
watchlists = load_json(WATCHLIST_FILE, DEFAULT_WATCHLISTS)
scanners = load_json(SCANNERS_FILE, DEFAULT_SCANNERS)
settings = load_json(SETTINGS_FILE, DEFAULT_SETTINGS)
history = load_json(HISTORY_FILE, [])

if "market" not in st.session_state:
    st.session_state["market"] = "Indian Stocks"
if "auto_refresh" not in st.session_state:
    st.session_state["auto_refresh"] = False
if "refresh_minutes" not in st.session_state:
    st.session_state["refresh_minutes"] = 5
if "next_scan_at" not in st.session_state:
    st.session_state["next_scan_at"] = None  # epoch seconds; timer continues across pages
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
if "show_add" not in st.session_state:
    st.session_state["show_add"] = False
if "scan_results" not in st.session_state:
    st.session_state["scan_results"] = []

MARKET_ORDER = ["Indian Stocks", "Crypto", "Forex"]
MARKET_LABEL = {"Indian Stocks": "🇮🇳 Stocks", "Crypto": "₿ Crypto", "Forex": "💱 Forex"}
MARKET_LABEL_SHORT = {"Indian Stocks": "Stocks", "Crypto": "Crypto", "Forex": "Forex"}
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

page = st.session_state["page"]
dyn_css = ""   # CSS that depends on current values (slider fill, active nav tab)

# ====================== HEADER ======================
with kc("hdr"):
    h1, h2, h3 = st.columns([1, 3.2, 1])
    with h1:
        st.button("☰", key="hdr_menu", use_container_width=True, on_click=go_page, args=("alerts",))
    with h2:
        st.markdown('<div class="app-title">PA Scanner</div>', unsafe_allow_html=True)
    with h3:
        st.button("☀️" if is_dark else "🌙", key="theme_toggle", use_container_width=True, on_click=toggle_theme)


# ====================================================================
#                              DASHBOARD
# ====================================================================
if page == "dashboard":

    # ---------- Market selector (selected one glows, other two fade/blur) ----------
    _mopts = ["Indian Stocks", "Crypto", "Forex"]
    _mlabels = {"Indian Stocks": "📈 Stocks", "Crypto": "₿ Crypto", "Forex": "💱 Forex"}
    try:
        cur_i = _mopts.index(st.session_state.get("market", "Indian Stocks"))
    except ValueError:
        cur_i = 0

    with kc("market_wrap"):
        picked = st.radio(
            "Market",
            options=_mopts,
            index=cur_i,
            format_func=lambda x: _mlabels.get(x, x),
            horizontal=True,
            label_visibility="collapsed",
            key="market_radio",
        )
    st.session_state["market"] = picked
    market = picked
    current_list = watchlists.get(market, [])

    # ---------- Scan panel ----------
    with kc("panel_scan"):
        # Timeframe  (big colourful label, dropdown next to it)
        with kc("tf_row"):
            c1, c2 = st.columns([1.2, 1])
            with c1:
                st.markdown('<div class="lbl-tf">⏱ Timeframe</div>', unsafe_allow_html=True)
            with c2:
                tf_default = st.session_state.get(f"ui_tf_{market}", "5m")
                if tf_default not in TF_OPTIONS:
                    tf_default = "5m"
                timeframe = st.selectbox(
                    "Timeframe",
                    TF_OPTIONS,
                    index=TF_OPTIONS.index(tf_default),
                    key=f"tf_{market}",
                    label_visibility="collapsed",
                )
                st.session_state[f"ui_tf_{market}"] = timeframe

        # Min score
        score_default = int(st.session_state.get(f"ui_score_{market}", 60))
        score_now = int(st.session_state.get(f"score_{market}", score_default))
        st.markdown(
            f'<div class="ms-row"><span class="lbl-ms">Min Score '
            f'<span class="info" title="Only setups with a score at or above this are shown">ⓘ</span></span>'
            f'<span class="ms-val">{score_now} <span class="star">★</span></span></div>',
            unsafe_allow_html=True,
        )
        min_score = st.slider("Min Score", 0, 100, score_default, 5, key=f"score_{market}", label_visibility="collapsed")
        st.session_state[f"ui_score_{market}"] = min_score
        _p = max(0, min(100, int(min_score)))
        dyn_css += (
            '[data-testid="stSlider"] div[data-baseweb="slider"] > div:first-child > div:first-child, '
            '[data-testid="stSlider"] div[data-baseweb="slider"] div:has(+ div[role="slider"]) {'
            f'background: linear-gradient(to right, #00e676 0%, #c6ff00 {_p*0.5:.0f}%, #ffea00 {_p*0.78:.0f}%, '
            f'#ff9100 {_p}%, rgba(255,255,255,0.14) {_p}%, rgba(255,255,255,0.14) 100%) !important; }}'
        )

        only_full = st.toggle(
            "Full Matches Only",
            value=bool(st.session_state.get(f"ui_full_{market}", False)),
            key=f"full_{market}",
        )
        st.session_state[f"ui_full_{market}"] = only_full

        # Auto refresh (same options as timeframe, slightly smaller)
        with kc("ar_row"):
            a1, a2 = st.columns([1.2, 1])
            with a1:
                st.markdown('<div class="lbl-ar">⟳ Auto Refresh</div>', unsafe_allow_html=True)
            with a2:
                ar_default = st.session_state.get("ui_ar", "Off")
                if ar_default not in AR_OPTIONS:
                    ar_default = "Off"
                ar_choice = st.selectbox(
                    "Auto refresh",
                    AR_OPTIONS,
                    index=AR_OPTIONS.index(ar_default),
                    key="ar_select",
                    label_visibility="collapsed",
                )
                st.session_state["ui_ar"] = ar_choice
        auto_refresh = ar_choice != "Off"
        st.session_state["auto_refresh"] = auto_refresh
        if auto_refresh:
            st.session_state["refresh_minutes"] = AR_MINUTES[ar_choice]
            nsa = st.session_state.get("next_scan_at")
            if nsa:
                left = int(nsa - time.time())
                if left > 0:
                    h_, m_, s_ = left // 3600, (left % 3600) // 60, left % 60
                    txt = f"{h_}h {m_}m" if h_ else f"{m_}m {s_}s"
                    st.caption(f"⏳ Next scan in {txt}")

        # Progress line ABOVE the scan button
        progress_slot = st.empty()
        status_slot = st.empty()
        progress_slot.progress(0)

        run = st.button("SCAN NOW", type="primary", use_container_width=True, key="scan_main")
        if st.button("🔄 Refresh Now", use_container_width=True, key="refresh_now"):
            st.session_state["force_scan"] = True
            st.rerun()

    # ---------- run scan (logic unchanged) ----------
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
            results = []
            for i, sym in enumerate(current_list):
                status_slot.caption(f"Scanning {sym} ({i+1}/{len(current_list)})")
                res = scan_symbol(sym, interval=timeframe, rules=active_rules)
                if res:
                    if only_full and not res.get("full_match"):
                        pass
                    elif res["score"] >= min_score:
                        results.append(res)
                progress_slot.progress((i + 1) / max(len(current_list), 1))
                time.sleep(0.08)
            status_slot.caption("Scan complete")

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

            if not results:
                st.warning("No setups found. Try lower score or other TF.")

            # Schedule next auto scan from NOW (timer not tied to Results view)
            if st.session_state.get("auto_refresh"):
                mins = int(st.session_state.get("refresh_minutes", 5))
                st.session_state["next_scan_at"] = time.time() + mins * 60
            else:
                st.session_state["next_scan_at"] = None

    # ---------- results: right below the scan button ----------
    results = st.session_state.get("scan_results") or []
    if st.session_state.get("last_scan"):
        with kc("panel_dash_results"):
            st.markdown(
                f'<div class="ptitle">Results • {st.session_state.get("last_market","")} • '
                f'{st.session_state.get("last_tf","")} • {st.session_state.get("last_scan","—")}</div>',
                unsafe_allow_html=True,
            )
            if not results:
                st.caption("No setups found on last scan.")
            else:
                st.caption(f"{len(results)} setup(s) found — tap the centre button for charts")
                for res in results:
                    st.markdown(result_card_html(res, detailed=False), unsafe_allow_html=True)

    # ---------- Watchlist ----------
    with kc("panel_watch"):
        with kc("wl_head"):
            w1, w2, w3 = st.columns([2.1, 0.7, 1.5])
            with w1:
                st.markdown(
                    f'<div class="lbl-wl"><span class="star">★</span> Watchlist '
                    f'<span class="cnt">{len(current_list)}</span></div>',
                    unsafe_allow_html=True,
                )
            with w2:
                st.button("+", key="wl_add_btn", on_click=toggle_flag, args=("show_add",), help="Add symbols")
            with w3:
                st.button("View all ›", key="wl_toggle", on_click=toggle_flag, args=("show_watchlist",))

        if not current_list:
            st.caption("No symbols yet — tap + to add")
        else:
            tiles = ""
            for sym in current_list[:4]:
                short = sym.replace(".NS", "").replace("-USD", "").replace("=X", "")
                price, chg = get_quote(sym)
                if chg is None:
                    cls, arrow, chg_txt = "flat", "•", "—"
                elif chg >= 0:
                    cls, arrow, chg_txt = "up", "↗", f"+{chg:.2f}%"
                else:
                    cls, arrow, chg_txt = "down", "↘", f"{chg:.2f}%"
                tiles += (
                    f'<div class="wl-tile"><div class="wl-top"><span class="wl-ico {cls}">{arrow}</span>'
                    f'<span class="wl-sym">{short}</span></div>'
                    f'<div class="wl-cat">{MARKET_LABEL_SHORT.get(market, market)}</div>'
                    f'<div class="wl-price">{fmt_price(price)}</div>'
                    f'<div class="wl-chg {cls}">{chg_txt}</div></div>'
                )
            st.markdown(f'<div class="wl-grid">{tiles}</div>', unsafe_allow_html=True)

        if st.session_state.get("show_watchlist"):
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

    # ---------- Add symbols (opens with the + button) ----------
    if st.session_state.get("show_add"):
        with kc("panel_add"):
            st.markdown('<div class="ptitle">+ Add symbols (many at once)</div>', unsafe_allow_html=True)
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


# ====================================================================
#                                RULES
# ====================================================================
elif page == "rules":
    with kc("panel_rules_head"):
        page_title("Scanner Rules", "View and change the rules the scanner uses")

    with kc("panel_rules_body"):
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
            if st.button("💾 Save rules", type="primary", use_container_width=True, key="save_rules"):
                scanners[scanner_name]["rules"] = r
                save_json(SCANNERS_FILE, scanners)
                st.session_state["rules_edit"] = False
                st.success("Rules saved — no app update needed")
                st.rerun()
            if st.button("Cancel edit", use_container_width=True):
                st.session_state["rules_edit"] = False
                st.rerun()

    with kc("panel_rules_profiles"):
        st.markdown('<div class="ptitle">Scanner profiles</div>', unsafe_allow_html=True)
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


# ====================================================================
#                               RESULTS
# ====================================================================
elif page == "results":
    with kc("panel_res_head"):
        page_title(
            "Scan Results",
            f"{st.session_state.get('last_market','')} • {st.session_state.get('last_tf','')} • "
            f"{st.session_state.get('last_scan','—')}",
        )
        st.button("← Back to Dashboard", use_container_width=True, key="res_back",
                  on_click=go_page, args=("dashboard",))

    results = st.session_state.get("scan_results", [])
    if not results:
        st.info("No results yet. Run a scan first.")
    else:
        st.caption(f"{len(results)} setups")
        for idx, res in enumerate(results):
            st.markdown(result_card_html(res, detailed=True), unsafe_allow_html=True)
            with st.expander(f"Chart • {res['symbol']}", expanded=(idx == 0)):
                render_chart(res)


# ====================================================================
#                                ALERTS
# ====================================================================
elif page == "alerts":
    with kc("panel_alerts_head"):
        page_title("Alerts", "Get setups sent to your Telegram")

    with kc("panel_alerts_body"):
        enable_tg = st.toggle("Enable", value=settings.get("enable_telegram", False))
        tg_token = st.text_input("Bot Token", value=settings.get("telegram_token", ""), type="password")
        tg_chat = st.text_input("Chat ID", value=settings.get("telegram_chat_id", ""))
        if st.button("Save Telegram", use_container_width=True):
            settings["enable_telegram"] = enable_tg
            settings["telegram_token"] = tg_token
            settings["telegram_chat_id"] = tg_chat
            save_json(SETTINGS_FILE, settings)
            st.success("Saved")


# ====================================================================
#                                HISTORY
# ====================================================================
elif page == "history":
    with kc("panel_hist_head"):
        page_title("Scan History", "Your last 30 scans, newest first")

    if not history:
        st.info("No history yet.")
    else:
        for h in history:
            chips = "".join(
                f'<span class="chip">{s.replace(".NS", "").replace("-USD", "").replace("=X", "")}</span>'
                for s in h.get("symbols", [])[:10]
            )
            if len(h.get("symbols", [])) > 10:
                chips += '<span class="chip">…</span>'
            top = f' • top score <b>{h.get("top_score")}</b>' if h.get("top_score") else ""
            st.markdown(
                f'<div class="card"><div style="font-weight:800;font-size:1.02rem;">{h.get("time","")}</div>'
                f'<div style="color:#9aabc8;margin-top:0.25rem;font-size:0.9rem;">'
                f'{h.get("market","")} • {h.get("timeframe","")} • <b>{h.get("count",0)}</b> setups{top}</div>'
                f'<div style="margin-top:0.3rem;">{chips or "—"}</div></div>',
                unsafe_allow_html=True,
            )
        if st.button("Clear history", use_container_width=True, key="clear_hist"):
            save_json(HISTORY_FILE, [])
            st.rerun()

st.caption("PA Scanner • Mobile first • Chart-matched price action")

# ====================================================================
#                   BOTTOM NAV  (sits above "Manage app")
# ====================================================================
if MODERN:
    NAV_ITEMS = [
        ("dashboard", ":material/grid_view:", "Dashboard"),
        ("rules", ":material/tune:", "Rules"),
        ("results", ":material/query_stats:", ""),
        ("alerts", ":material/notifications:", "Alerts"),
        ("history", ":material/history:", "History"),
    ]
else:
    NAV_ITEMS = [
        ("dashboard", "▦", "Dashboard"),
        ("rules", "📐", "Rules"),
        ("results", "📊", ""),
        ("alerts", "🔔", "Alerts"),
        ("history", "🕒", "History"),
    ]

with kc("bottomnav"):
    ncols = st.columns([1, 1, 1.05, 1, 1])
    for col, (pid, icon, text) in zip(ncols, NAV_ITEMS):
        with col:
            label = f"{icon}  \n{text}" if text else icon
            st.button(
                label,
                key=f"nav_{pid}",
                use_container_width=True,
                on_click=go_page,
                args=(pid,),
                help="Results" if pid == "results" else None,
            )

# highlight the active tab + live slider fill
dyn_css += f".st-key-nav_{page} button {{ color: #69f0ae !important; text-shadow: 0 0 10px rgba(105,240,174,0.6); }}"
st.markdown(f"<style>{dyn_css}</style>", unsafe_allow_html=True)

# Soft poll for countdown / due auto (does not reset timer when visiting other pages)
if page == "dashboard" and st.session_state.get("auto_refresh"):
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
