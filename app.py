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








# ====================== SCANNER 2 : HEAD & SHOULDERS / INVERSE H&S ======================
# Independent from Scanner 1 (scan_symbol is untouched).
# The inverse pattern is the exact mirror of the normal one: the price series is flipped
# (O,H,L,C -> -O,-L,-H,-C), the same top-pattern detector runs, and results are flipped back.
HNS_FILE = os.path.join(DATA_DIR, "hns_rules.json")

DEFAULT_HNS_RULES = {
    # which patterns
    "scan_bearish_hs": True,          # Head & Shoulders  (bearish, break BELOW neckline)
    "scan_inverse_hs": True,          # Inverse H&S       (bullish, break ABOVE neckline)
    # prior trend into the left shoulder
    "require_prior_trend": True,
    "min_trend_pct": 3.0,             # move into left shoulder >= 3% ...
    "trend_atr_mult": 3.0,            # ... or 3x ATR (whichever is smaller)
    "trend_lookback": 40,             # candles looked back from the left shoulder
    # shape
    "swing_strength": 3,              # candles each side to confirm a peak
    "min_head_prominence_pct": 1.0,   # head above both shoulders by >= 1% ...
    "head_atr_mult": 1.0,             # ... or 1x ATR (whichever is smaller)
    "max_shoulder_diff_pct": 15.0,    # shoulder heights (above neckline) within +-15%
    "max_time_asym_pct": 40.0,        # left/right time spans within +-40%
    "max_neckline_slope_deg": 5.0,    # neckline tilt (normalised to head height)
    # size
    "min_pattern_candles": 20,
    "max_pattern_candles": 80,
    "min_gap_candles": 5,             # min candles between LS-Head and Head-RS
    # breakout
    "min_break_pct": 0.25,            # close beyond neckline by >= 0.25%
    "require_healthy_break": True,    # strong body (>=45% of range) closing near its extreme
    "max_break_delay": 10,            # breakout within 10 candles of the right shoulder
    "invalid_if_above_rs": True,      # invalid if price closes beyond the right shoulder first
    "max_signal_age": 3,              # only show signals from the last N candles
    # optional
    "require_volume": False,
    "require_retest": False,
    "require_min_rr": False,
    "min_rr": 1.5,
}


def _hns_fetch(symbol, interval):
    """Download enough history for pattern work (3m / 4h are built by resampling)."""
    try:
        spec = {
            "1m": ("5d", "1m", None),
            "3m": ("5d", "1m", "3min"),
            "5m": ("30d", "5m", None),
            "15m": ("60d", "15m", None),
            "30m": ("60d", "30m", None),
            "1h": ("180d", "1h", None),
            "4h": ("365d", "1h", "4h"),
            "1d": ("2y", "1d", None),
        }
        period, yf_int, rs = spec.get(interval, ("60d", interval, None))
        df = yf.Ticker(symbol).history(period=period, interval=yf_int)
        if df is None or len(df) == 0:
            return None
        df = df.dropna(subset=["Open", "High", "Low", "Close"])
        if rs:
            df = df.resample(rs).agg(
                {"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"}
            ).dropna(subset=["Open", "High", "Low", "Close"])
        return df
    except Exception:
        return None


def _hns_swings(H, k):
    """Indices of swing highs. Right side may be shorter than k for the newest candles
    (so a fresh right shoulder can be used right away)."""
    n = len(H)
    out = []
    for i in range(k, n - 1):
        r = min(k, n - 1 - i)
        win = H[i - k:i + r + 1]
        if H[i] == win.max() and int(np.argmax(win)) == k:
            out.append(i)
    return out


def _hns_detect(O, H, L, C, V, r):
    """Find the best *top-type* head & shoulders (head = highest peak, break DOWN).
    Returns a dict (values in this orientation) or None."""
    n = len(C)
    k = max(1, int(r["swing_strength"]))
    prev_c = np.roll(C, 1)
    prev_c[0] = C[0]
    tr = np.maximum(H - L, np.maximum(np.abs(H - prev_c), np.abs(L - prev_c)))
    atr = np.nan_to_num(pd.Series(tr).rolling(14, min_periods=5).mean().values, nan=0.0)

    sh = _hns_swings(H, k)
    if len(sh) < 3:
        return None

    min_w = int(r["min_pattern_candles"])
    max_w = int(r["max_pattern_candles"])
    gap = int(r["min_gap_candles"])
    delay = int(r["max_break_delay"])
    age = int(r["max_signal_age"])
    rs_min = n - 1 - age - delay - (15 if r.get("require_retest") else 0)

    max_sh = max(float(r["max_shoulder_diff_pct"]), 0.01)
    max_t = max(float(r["max_time_asym_pct"]), 0.01)
    max_ang = max(float(r["max_neckline_slope_deg"]), 0.01)
    min_brk = max(float(r["min_break_pct"]), 0.001)

    best = None
    for rs in sh:
        if rs < rs_min:
            continue
        for h in sh:
            if h > rs - gap:
                break
            for ls in sh:
                if ls > h - gap:
                    break
                if not (min_w <= rs - ls <= max_w):
                    continue
                Pls, Ph, Prs = H[ls], H[h], H[rs]
                if H[ls:rs + 1].max() > Ph:            # head must be the highest point
                    continue

                # troughs + neckline (through the two troughs)
                t1 = ls + int(np.argmin(L[ls:h + 1]))
                t2 = h + int(np.argmin(L[h:rs + 1]))
                if t2 <= t1:
                    continue
                n1, n2 = float(L[t1]), float(L[t2])

                def neck(x, n1=n1, n2=n2, t1=t1, t2=t2):
                    return n1 + (n2 - n1) * (x - t1) / (t2 - t1)

                hh = Ph - neck(h)
                hl = Pls - neck(ls)
                hr = Prs - neck(rs)
                if hh <= 0 or hl <= 0 or hr <= 0:
                    continue

                # 3. head clearly above both shoulders
                a = atr[rs]
                need = r["min_head_prominence_pct"] / 100.0 * abs(Ph)
                if a > 0 and r["head_atr_mult"] > 0:
                    need = min(need, r["head_atr_mult"] * a)
                if Ph - max(Pls, Prs) < need:
                    continue

                # 4. symmetry
                sh_diff = abs(hl - hr) / max(hl, hr) * 100
                if sh_diff > max_sh:
                    continue
                lt, rt = h - ls, rs - h
                t_diff = abs(lt - rt) / max(lt, rt) * 100
                if t_diff > max_t:
                    continue

                # 5. neckline slope (tilt relative to head height, 1 unit horizontal)
                angle = float(np.degrees(np.arctan(abs(n2 - n1) / hh)))
                if angle > max_ang:
                    continue

                # 1. prior trend into the left shoulder
                lb = max(0, ls - int(r["trend_lookback"]))
                move = float(Pls - L[lb:ls + 1].min())
                t_need = r["min_trend_pct"] / 100.0 * abs(Pls)
                if atr[ls] > 0 and r["trend_atr_mult"] > 0:
                    t_need = min(t_need, r["trend_atr_mult"] * atr[ls])
                trend_ok = move >= t_need
                if r["require_prior_trend"] and not trend_ok:
                    continue

                # 7/8/9. breakout candle
                b = None
                for j in range(rs + 1, min(n, rs + delay + 1)):
                    if r["invalid_if_above_rs"] and C[j] > Prs:
                        break
                    nk = neck(j)
                    if C[j] < nk - abs(nk) * min_brk / 100.0:
                        if r["require_healthy_break"]:
                            candle = {"Open": O[j], "High": H[j], "Low": L[j], "Close": C[j]}
                            if not is_healthy_break_candle(candle, "down"):
                                continue
                        b = j
                        break
                if b is None:
                    continue
                if r["invalid_if_above_rs"] and np.any(C[b + 1:] > Prs):   # pattern failed after break
                    continue

                # optional retest: price returns to the neckline from below and is rejected
                sig, retest = b, False
                for j in range(b + 2, n):
                    nk = neck(j)
                    if H[j] >= nk - abs(nk) * 0.001 and C[j] < nk:
                        retest, sig = True, j
                        break
                if r["require_retest"] and not retest:
                    continue
                if (n - 1) - sig > age:                      # not fresh any more
                    continue

                # reward : risk  (stop beyond right shoulder, target = measured move)
                entry = float(C[sig])
                stop = float(Prs)
                target = float(neck(sig) - hh)
                risk, reward = stop - entry, entry - target
                if risk <= 0:
                    continue
                rr = reward / risk if reward > 0 else 0.0
                if r["require_min_rr"] and rr < float(r["min_rr"]):
                    continue

                # optional volume confirmation
                has_vol = V is not None and float(np.nansum(V[ls:sig + 1])) > 0
                cond_a = cond_b = None
                vol_ok = None
                if has_vol:
                    def pv(i):
                        return float(np.nanmean(V[max(0, i - 1):i + 2]))
                    cond_a = max(pv(ls), pv(h)) > pv(rs)                   # fades into right shoulder
                    pre = V[max(0, b - 10):b]
                    cond_b = bool(len(pre) and V[b] > np.nanmean(pre))      # breakout on higher volume
                    vol_ok = bool(cond_a and cond_b)
                    if r["require_volume"] and not vol_ok:
                        continue

                # score (100): symmetry 25, breakout 25, volume 20, prior trend 15, neckline 15
                sym = 15 * max(0.0, 1 - sh_diff / max_sh) + 10 * max(0.0, 1 - t_diff / max_t)
                nkb = neck(b)
                bdist = (nkb - C[b]) / abs(nkb) * 100
                rng = H[b] - L[b]
                body_ratio = abs(C[b] - O[b]) / rng if rng > 0 else 0.0
                brk = 15 * min(1.0, bdist / (3 * min_brk)) + 10 * min(1.0, max(0.0, (body_ratio - 0.45) / 0.4))
                if has_vol:
                    volp = (8 if cond_a else 0) + (12 if cond_b else 0)
                else:
                    volp = 10   # no volume data (e.g. forex) -> neutral
                trp = 15 * min(1.0, move / (2 * t_need)) if t_need > 0 else 7.5
                flat = 15 * max(0.0, 1 - angle / max_ang)
                score = int(round(min(100.0, sym + brk + volp + trp + flat)))

                cand = {
                    "score": score, "ls": ls, "h": h, "rs": rs, "t1": t1, "t2": t2, "b": b, "sig": sig,
                    "n1": n1, "n2": n2, "neck_sig": float(neck(sig)), "neck_b": float(nkb),
                    "Pls": float(Pls), "Ph": float(Ph), "Prs": float(Prs),
                    "entry": entry, "stop": stop, "target": target, "rr": rr,
                    "retest": retest, "vol_ok": vol_ok, "trend_ok": trend_ok,
                    "sh_diff": sh_diff, "t_diff": t_diff, "angle": angle,
                    "parts": {"symmetry": round(sym), "breakout": round(brk), "volume": round(volp),
                              "prior_trend": round(trp), "neckline": round(flat)},
                }
                if best is None or (cand["score"], cand["sig"]) > (best["score"], best["sig"]):
                    best = cand
    return best


def scan_hns(symbol, interval="15m", rules=None):
    """Scanner 2 - Head & Shoulders (bearish) and Inverse Head & Shoulders (bullish)."""
    try:
        r = dict(DEFAULT_HNS_RULES)
        if rules:
            r.update(rules)
        df = _hns_fetch(symbol, interval)
        if df is None or len(df) < 60:
            return None
        win = (int(r["max_pattern_candles"]) + int(r["max_break_delay"]) + int(r["trend_lookback"])
               + int(r["max_signal_age"]) + 40)
        df = df.tail(max(win, 120))

        O = df["Open"].values.astype(float)
        H = df["High"].values.astype(float)
        L = df["Low"].values.astype(float)
        C = df["Close"].values.astype(float)
        V = df["Volume"].values.astype(float) if "Volume" in df.columns else None

        found = []
        if r["scan_bearish_hs"]:
            p = _hns_detect(O, H, L, C, V, r)
            if p:
                p["sign"] = 1
                found.append(p)
        if r["scan_inverse_hs"]:
            p = _hns_detect(-O, -L, -H, -C, V, r)      # mirror image
            if p:
                p["sign"] = -1
                found.append(p)
        if not found:
            return None
        p = max(found, key=lambda x: (x["score"], x["sig"]))

        sg = p["sign"]
        def pr(v):
            return round(float(sg * v), 5)

        idx = df.index
        name = "Head & Shoulders" if sg == 1 else "Inverse H&S"
        direction = "Bearish" if sg == 1 else "Bullish"
        full = bool(p["vol_ok"] is True or (p["vol_ok"] is None and p["score"] >= 80))   # FULL = volume confirmed
        start = max(0, p["ls"] - 15)

        return {
            "symbol": symbol,
            "score": int(p["score"]),
            "full_match": full,
            "trend": name,
            "pattern": name,
            "direction": direction,
            "price": pr(p["entry"]),
            "interval": interval,
            "near_level": "Neckline",
            "level_price": pr(p["neck_b"]),
            "break_level": pr(p["neck_b"]),
            "target": pr(p["target"]),
            "stop": pr(p["stop"]),
            "rr": round(float(p["rr"]), 2),
            "break_time_ist": to_ist_str(idx[p["b"]]),
            "setup_time_ist": to_ist_str(idx[p["sig"]]),
            "scanned_at_ist": datetime.now(IST).strftime("%d-%b-%Y %H:%M IST"),
            "scores": p["parts"],
            "details": {
                "prior_trend_ok": bool(p["trend_ok"]),
                "volume_ok": p["vol_ok"],
                "retest": bool(p["retest"]),
                "shoulder_diff_pct": round(p["sh_diff"], 1),
                "time_diff_pct": round(p["t_diff"], 1),
                "neckline_angle": round(p["angle"], 1),
            },
            # neckline drawn from the first trough to the signal candle
            "trend_points": [
                [str(idx[p["t1"]]), pr(p["n1"])],
                [str(idx[p["sig"]]), pr(p["neck_sig"])],
            ],
            "pattern_points": [
                ["LS", str(idx[p["ls"]]), pr(p["Pls"])],
                ["H", str(idx[p["h"]]), pr(p["Ph"])],
                ["RS", str(idx[p["rs"]]), pr(p["Prs"])],
            ],
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "df": df.iloc[start:],
        }
    except Exception:
        return None




# ====================== TELEGRAM: CHART IMAGE + FULL DETAILS ======================
# send_telegram_alert (text only) above is unchanged and is still used as the fallback.
import io as _io
import html as _html
import threading as _th
_CHART_LOCK = _th.Lock()


def _pos_of(df, x_str):
    """Row position of a timestamp string in df (matches to the minute)."""
    key = str(x_str)[:16]
    for n_, ts in enumerate(df.index):
        if str(ts)[:16] == key:
            return n_
    return None


def make_alert_chart_png(res):
    """Draw the setup (candles, S/R or neckline, trendline, pattern points, target/stop) as a PNG.
    Returns (png_bytes or None, error_text)."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle
    except Exception:
        return None, "matplotlib is not installed (add 'matplotlib' to requirements.txt)"

    try:
        df = res.get("df")
        if df is None or len(df) < 5:
            return None, "no candle data"
        n = len(df)
        O = df["Open"].values.astype(float)
        H = df["High"].values.astype(float)
        L = df["Low"].values.astype(float)
        C = df["Close"].values.astype(float)
        V = df["Volume"].values.astype(float) if "Volume" in df.columns else None
        has_vol = V is not None and np.nansum(V) > 0

        bg, panel, grid = "#0b1020", "#0f1630", "#1d2750"
        up, dn, amber = "#26d07c", "#ff5252", "#ffb300"
        txt = "#dfe6ff"

        if has_vol:
            fig, (ax, axv) = plt.subplots(2, 1, figsize=(10, 6.4), dpi=110, sharex=True,
                                          gridspec_kw={"height_ratios": [4, 1], "hspace": 0.04})
        else:
            fig, ax = plt.subplots(1, 1, figsize=(10, 5.6), dpi=110)
            axv = None
        fig.patch.set_facecolor(bg)
        for a in [ax] + ([axv] if axv is not None else []):
            a.set_facecolor(panel)
            a.grid(True, color=grid, linewidth=0.6, alpha=0.7)
            a.tick_params(colors=txt, labelsize=8)
            for sp in a.spines.values():
                sp.set_color(grid)

        # candles
        for i in range(n):
            col = up if C[i] >= O[i] else dn
            ax.plot([i, i], [L[i], H[i]], color=col, linewidth=1.0, zorder=2)
            body_lo, body_hi = min(O[i], C[i]), max(O[i], C[i])
            ax.add_patch(Rectangle((i - 0.33, body_lo), 0.66, max(body_hi - body_lo, (H.max() - L.min()) * 0.0015),
                                   facecolor=col, edgecolor=col, zorder=3))
        if axv is not None:
            axv.bar(range(n), V, color=[up if C[i] >= O[i] else dn for i in range(n)], width=0.7, alpha=0.8)
            axv.set_ylabel("Vol", color=txt, fontsize=8)

        is_pat = bool(res.get("pattern"))
        levels_for_ylim = [L.min(), H.max()]

        # support / resistance line (Scanner 1)
        if res.get("level_price") and not is_pat:
            lvl = float(res["level_price"])
            lc = up if res.get("near_level") == "Support" else dn
            ax.axhline(lvl, color=lc, linestyle="--", linewidth=1.8, zorder=1)
            ax.text(0.5, lvl, f" {res.get('near_level','Level')} {lvl:g}", color=lc, fontsize=9,
                    va="bottom", fontweight="bold")
            levels_for_ylim.append(lvl)

        # trendline / neckline
        tp = res.get("trend_points") or []
        pts = []
        for x_str, y in tp:
            p_ = _pos_of(df, x_str)
            if p_ is not None:
                pts.append((p_, float(y)))
        if len(pts) >= 2:
            tcol = amber if is_pat else (up if res.get("direction") == "Bullish" else dn)
            (x1, y1), (x2, y2) = pts[0], pts[-1]
            ax.plot([x1, x2], [y1, y2], color=tcol, linewidth=2.2, marker="o", markersize=5, zorder=4)
            if x2 > x1 and not is_pat:      # extend the trendline to the latest candle
                slope = (y2 - y1) / (x2 - x1)
                ax.plot([x2, n - 1], [y2, y2 + slope * (n - 1 - x2)], color=tcol, linewidth=1.4,
                        linestyle=":", zorder=4)
            if is_pat:
                ax.text(x1, y1, " Neckline", color=tcol, fontsize=9, va="top", fontweight="bold")

        # head & shoulders markers, target and stop
        if is_pat:
            for lab, x_str, y in (res.get("pattern_points") or []):
                p_ = _pos_of(df, x_str)
                if p_ is not None:
                    inv = res.get("direction") == "Bullish"
                    ax.scatter([p_], [y], color=amber, s=46, zorder=5)
                    ax.annotate(lab, (p_, y), textcoords="offset points",
                                xytext=(0, -14 if inv else 8), ha="center", color="#ffd54f",
                                fontsize=10, fontweight="bold")
            if res.get("target") is not None:
                t_ = float(res["target"])
                ax.axhline(t_, color=up, linestyle=":", linewidth=1.5)
                ax.text(n - 1, t_, f"Target {t_:g} ", color=up, fontsize=9, ha="right", va="bottom", fontweight="bold")
                levels_for_ylim.append(t_)
            if res.get("stop") is not None:
                s_ = float(res["stop"])
                ax.axhline(s_, color=dn, linestyle=":", linewidth=1.5)
                ax.text(n - 1, s_, f"Stop {s_:g} ", color=dn, fontsize=9, ha="right", va="bottom", fontweight="bold")
                levels_for_ylim.append(s_)

        # mark break + setup candles
        labels_ist = [to_ist_str(ts) for ts in df.index]
        span = max(levels_for_ylim) - min(levels_for_ylim)
        bt, st_ = res.get("break_time_ist"), res.get("setup_time_ist")
        marks = []
        if bt in labels_ist and st_ in labels_ist and bt == st_:
            marks.append((labels_ist.index(bt), "Break / Setup", "#40c4ff"))
        else:
            if bt in labels_ist:
                marks.append((labels_ist.index(bt), "Break", "#40c4ff"))
            if st_ in labels_ist:
                marks.append((labels_ist.index(st_), "Setup", "#ea80fc"))
        for p_, name, colr in marks:
            ax.axvline(p_, color=colr, linestyle="--", linewidth=1.0, alpha=0.8)
            ax.text(p_, max(levels_for_ylim) + span * 0.02, name, color=colr, fontsize=8,
                    ha=("left" if name == "Setup" else "right"), va="bottom", fontweight="bold")

        pad = span * 0.08 if span > 0 else 1
        ax.set_ylim(min(levels_for_ylim) - pad, max(levels_for_ylim) + pad * 1.6)
        ax.set_xlim(-1, n)

        # x labels
        step = max(1, n // 7)
        ticks = list(range(0, n, step))
        tick_txt = [f"{labels_ist[i][:6]}\n{labels_ist[i][12:17]}" for i in ticks]
        (axv if axv is not None else ax).set_xticks(ticks)
        (axv if axv is not None else ax).set_xticklabels(tick_txt, color=txt, fontsize=8)
        if axv is not None:
            ax.tick_params(labelbottom=False)

        name = res.get("pattern") or f"{res.get('direction','')} setup"
        fig.suptitle(f"{res.get('symbol','')}  |  {res.get('interval','')}  |  {name}  |  Score {res.get('score','')}",
                     color="#ffffff", fontsize=13, fontweight="bold", x=0.02, ha="left", y=0.985)
        fig.text(0.98, 0.012, "PA Scanner", color="#6b7bb5", fontsize=8, ha="right")
        fig.subplots_adjust(left=0.06, right=0.985, top=0.93, bottom=0.10 if axv is not None else 0.12)

        buf = _io.BytesIO()
        fig.savefig(buf, format="png", facecolor=bg)
        plt.close(fig)
        return buf.getvalue(), ""
    except Exception as e:
        try:
            plt.close("all")
        except Exception:
            pass
        return None, f"chart error: {e}"


_S1_RULES = [
    ("require_near_sr", False, lambda r: f"Near support/resistance (within {r.get('sr_pct', 1)}%)"),
    ("require_structure", True, lambda r: "Higher-low / lower-high structure"),
    ("require_healthy_break", True, lambda r: "Healthy breakout/breakdown candle"),
    ("require_pin_bar", False, lambda r: "Rejection / doji after the break"),
    ("require_high_volume_rejection", False, lambda r: "Rejection volume > breakout volume"),
    ("require_consolidation", False, lambda r: "Consolidation before the break"),
    ("require_volume_dry", False, lambda r: "Volume drying before the break"),
]


def build_alert_text(res, scanner_id, market, timeframe, rules):
    """Full alert text (HTML for Telegram). Returns (full_text, short_headline)."""
    e = _html.escape
    rules = rules or {}
    bull = res.get("direction") == "Bullish"
    dot = "🟢" if bull else "🔴"
    sym = e(str(res.get("symbol", "")))
    sc_name = "Scanner 2 · Head &amp; Shoulders" if scanner_id == 2 else "Scanner 1 · Price Action"

    head = (f"🚨 <b>{sym}</b>  {dot} {e(str(res.get('pattern') or res.get('direction','')))}"
            f"{' (' + e(str(res.get('direction',''))) + ')' if res.get('pattern') else ''}\n"
            f"Score <b>{res.get('score','—')}</b> | {e(str(timeframe))} | Price <b>{res.get('price','—')}</b>")

    lines = [head, f"<i>{sc_name} • {e(str(market))}</i>", ""]

    if res.get("pattern"):
        lines.append(f"Neckline: <b>{res.get('level_price','—')}</b>")
        lines.append(f"Target: <b>{res.get('target','—')}</b>   Stop: <b>{res.get('stop','—')}</b>   R:R <b>{res.get('rr','—')}</b>")
    else:
        lines.append(f"{e(str(res.get('near_level','Level')))}: <b>{res.get('level_price','—')}</b>  •  Trend: {e(str(res.get('trend','')))}")
    lines.append(f"Break candle: {e(str(res.get('break_time_ist','—')))}")
    lines.append(f"Setup candle: {e(str(res.get('setup_time_ist','—')))}")
    lines.append(f"Scanned: {e(str(res.get('scanned_at_ist','—')))}")
    lines.append("")

    if res.get("pattern"):
        d = res.get("details") or {}
        sc = res.get("scores") or {}
        lines.append("<b>Score breakdown</b>")
        lines.append(
            f"Symmetry {sc.get('symmetry','–')}/25 • Breakout {sc.get('breakout','–')}/25 • "
            f"Volume {sc.get('volume','–')}/20 • Trend {sc.get('prior_trend','–')}/15 • Neckline {sc.get('neckline','–')}/15"
        )
        lines.append("<b>Checks</b>")
        lines.append(f"✅ Shoulder height diff {d.get('shoulder_diff_pct','–')}% (max {rules.get('max_shoulder_diff_pct','–')}%)")
        lines.append(f"✅ Left/right time diff {d.get('time_diff_pct','–')}% (max {rules.get('max_time_asym_pct','–')}%)")
        lines.append(f"✅ Neckline slope {d.get('neckline_angle','–')}° (max {rules.get('max_neckline_slope_deg','–')}°)")
        lines.append(f"{'✅' if d.get('prior_trend_ok') else '➖'} Prior trend into left shoulder")
        v = d.get("volume_ok")
        lines.append(f"{'✅' if v else ('❌' if v is False else '➖')} Volume confirmation" + (" (no volume data)" if v is None else ""))
        lines.append(f"{'✅' if d.get('retest') else '➖'} Neckline retest")
        lines.append("✅ Strong breakout candle closed beyond neckline" if rules.get("require_healthy_break", True)
                     else "✅ Closed beyond neckline")
    else:
        lines.append("<b>Rules passed</b>")
        shown = 0
        for key, default, label in _S1_RULES:
            if rules.get(key, default):
                lines.append(f"✅ {e(label(rules))}")
                shown += 1
        if not shown:
            lines.append("✅ All enabled rules")

    full = "\n".join(lines)
    short = head
    return full, short


def send_telegram_photo(token, chat_id, photo_bytes, caption):
    """Send a PNG with caption. Returns (ok, error_text)."""
    if not token or not chat_id:
        return False, "missing token / chat id"
    try:
        url = f"https://api.telegram.org/bot{token}/sendPhoto"
        r = requests.post(
            url,
            data={"chat_id": chat_id, "caption": caption, "parse_mode": "HTML"},
            files={"photo": ("chart.png", photo_bytes, "image/png")},
            timeout=30,
        )
        if r.status_code == 200:
            return True, ""
        return False, f"Telegram said {r.status_code}: {r.text[:120]}"
    except Exception as e:
        return False, str(e)[:120]


def send_full_alert(token, chat_id, res, scanner_id, market, timeframe, rules, with_chart=True):
    """Chart image + full details. Falls back to text-only if the image can't be made/sent.
    Returns (ok, mode, note)."""
    full, short = build_alert_text(res, scanner_id, market, timeframe, rules)
    note = ""
    if with_chart:
        with _CHART_LOCK:
            png, err = make_alert_chart_png(res)
        if png:
            if len(full) <= 1000:
                ok, err2 = send_telegram_photo(token, chat_id, png, full)
            else:
                ok, err2 = send_telegram_photo(token, chat_id, png, short)
                if ok:
                    send_telegram_alert(token, chat_id, full)
            if ok:
                return True, "chart", ""
            note = err2
        else:
            note = err
    ok = send_telegram_alert(token, chat_id, full)
    return ok, "text", note




# ====================== AUTO-SCAN ENGINE (all three sections, runs in the background) ======================
# No Streamlit calls in here on purpose: a background thread uses this, so it keeps scanning
# every section whose Auto Refresh is ON - whichever page / section you are looking at.
import threading

AUTO_CFG_FILE = os.path.join(DATA_DIR, "auto_config.json")
SENT_FILE = os.path.join(DATA_DIR, "alerts_sent.json")
AUTO_MARKETS = ["Indian Stocks", "Crypto", "Forex"]
INTERVAL_MINUTES = {"1m": 1, "3m": 3, "5m": 5, "15m": 15, "1h": 60, "4h": 240, "1d": 1440}
SCANNER_LABELS = {1: "Scanner 1 · Price Action", 2: "Scanner 2 · Head & Shoulders"}

# defaults: timeframe 15m, auto-refresh interval 15m, auto refresh OFF until the user turns it on
DEFAULT_MARKET_CFG = {
    "timeframe": "15m",
    "min_score": 60,
    "full_only": False,
    "auto_on": False,
    "interval": "15m",
    "scanner": 1,
}

_DISPATCH_LOCK = threading.Lock()
_CHART_LOCK = threading.Lock()


def save_json_atomic(path, data):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp, path)


def settings_snapshot():
    s = dict(DEFAULT_SETTINGS)
    s.update(load_json(SETTINGS_FILE, {}))
    return s


class AutoState:
    """Shared by every browser session (created once per server process)."""

    def __init__(self):
        self.lock = threading.RLock()
        self.cfg = {}
        raw = load_json(AUTO_CFG_FILE, {})
        for m in AUTO_MARKETS:
            c = dict(DEFAULT_MARKET_CFG)
            part = raw.get(m, {}) if isinstance(raw, dict) else {}
            if isinstance(part, dict):
                c.update({k: v for k, v in part.items() if k in DEFAULT_MARKET_CFG})
            self.cfg[m] = c
        self.results = {}     # market -> {"results": [...], "time", "tf", "scanner", "source", "at"}
        self.next_run = {}    # market -> epoch seconds
        self.last_run = {}
        self.busy = {}
        self.errors = {}
        self.thread = None
        now = time.time()
        for m in AUTO_MARKETS:
            if self.cfg[m]["auto_on"]:
                self.next_run[m] = now + self._secs(m)

    def _secs(self, m):
        return INTERVAL_MINUTES.get(self.cfg[m].get("interval", "15m"), 15) * 60

    def get(self, m):
        with self.lock:
            return dict(self.cfg.get(m, DEFAULT_MARKET_CFG))

    def update(self, m, **kw):
        """Change settings of one section. Starts / stops its timer when needed."""
        with self.lock:
            c = self.cfg[m]
            changed = {}
            for k, v in kw.items():
                if k in DEFAULT_MARKET_CFG and c.get(k) != v:
                    c[k] = v
                    changed[k] = v
            if not changed:
                return
            if "auto_on" in changed or "interval" in changed:
                if c["auto_on"]:
                    self.next_run[m] = time.time() + self._secs(m)      # timer starts now
                else:
                    self.next_run.pop(m, None)
            try:
                save_json_atomic(AUTO_CFG_FILE, self.cfg)
            except Exception:
                pass

    def seconds_left(self, m):
        with self.lock:
            nr = self.next_run.get(m)
        return None if nr is None else int(nr - time.time())

    def set_results(self, m, results, tf, scanner, source):
        with self.lock:
            self.results[m] = {
                "results": results, "time": datetime.now().strftime("%H:%M"), "tf": tf,
                "scanner": scanner, "source": source, "at": time.time(),
            }

    def get_results(self, m):
        with self.lock:
            return self.results.get(m)

    def latest_any(self):
        with self.lock:
            if not self.results:
                return None, None
            m = max(self.results, key=lambda k: self.results[k]["at"])
            return m, self.results[m]


def core_scan(market, cfg, progress=None):
    """The scan loop (same rules / scoring as always). Used by SCAN NOW and by the auto engine.
    Returns (results, rules_used)."""
    wl = load_json(WATCHLIST_FILE, DEFAULT_WATCHLISTS).get(market, [])
    sc = load_json(SCANNERS_FILE, DEFAULT_SCANNERS)
    nm = list(sc.keys())[0] if sc else "My Price Action Scanner"
    rules1 = sc.get(nm, DEFAULT_SCANNERS["My Price Action Scanner"])["rules"]
    hns = dict(DEFAULT_HNS_RULES)
    hns.update(load_json(HNS_FILE, {}))

    tf = cfg["timeframe"]
    results = []
    n = len(wl)
    for i, sym in enumerate(wl):
        if progress:
            progress("start", i, n, sym)
        if cfg.get("scanner") == 2:
            res = scan_hns(sym, interval=tf, rules=hns)
        else:
            res = scan_symbol(sym, interval=tf, rules=rules1)
        if res:
            if cfg.get("full_only") and not res.get("full_match"):
                pass
            elif res["score"] >= cfg.get("min_score", 60):
                results.append(res)
        if progress:
            progress("done", i, n, sym)
        time.sleep(0.08)
    results = sorted(results, key=lambda x: x["score"], reverse=True)
    return results, (hns if cfg.get("scanner") == 2 else rules1)


def record_history(market, tf, scanner, results):
    try:
        hist = load_json(HISTORY_FILE, [])
        entry = {
            "time": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "market": market,
            "timeframe": tf,
            "scanner": SCANNER_LABELS.get(scanner, ""),
            "count": len(results),
            "symbols": [x["symbol"] for x in results[:15]],
            "top_score": results[0]["score"] if results else 0,
        }
        save_json_atomic(HISTORY_FILE, ([entry] + hist)[:30])
    except Exception:
        pass


def dispatch_alerts(results, scanner_id, market, timeframe, rules):
    """Telegram alert (chart + full details) for every NEW result. A setup that was already
    alerted (same symbol / timeframe / signal candle) is not sent again.
    Returns (sent, already_sent, notes)."""
    s = settings_snapshot()
    if not (s.get("enable_telegram") and s.get("telegram_token") and s.get("telegram_chat_id")):
        return 0, 0, ["telegram alerts are off"]
    with _DISPATCH_LOCK:
        log = load_json(SENT_FILE, {})
        cutoff = time.time() - 3 * 86400
        log = {k: v for k, v in log.items() if v >= cutoff}
        min_sc = int(s.get("telegram_min_score", 0))
        cap = int(s.get("telegram_max_alerts", 4))
        sent = skipped = fails = 0
        notes = []
        for r in results:
            if r.get("score", 0) < min_sc:
                continue
            key = f"{scanner_id}|{market}|{r.get('symbol')}|{timeframe}|{r.get('setup_time_ist')}"
            if key in log:
                skipped += 1
                continue
            if sent >= cap or fails >= 3:
                break
            ok, mode, note = send_full_alert(
                s["telegram_token"], s["telegram_chat_id"], r, scanner_id, market, timeframe, rules,
                with_chart=bool(s.get("telegram_send_chart", True)),
            )
            if ok:
                log[key] = time.time()
                sent += 1
                fails = 0
            else:
                fails += 1
            if note:
                notes.append(note)
            time.sleep(1.1)          # Telegram: about 1 message / second per chat
        try:
            save_json_atomic(SENT_FILE, log)
        except Exception:
            pass
    return sent, skipped, notes


def run_market_job(state, market):
    """One full automatic scan of a section + automatic Telegram alerts."""
    cfg = state.get(market)
    results, rules = core_scan(market, cfg)
    state.set_results(market, results, cfg["timeframe"], cfg["scanner"], "auto")
    if results:
        record_history(market, cfg["timeframe"], cfg["scanner"], results)
    dispatch_alerts(results, cfg["scanner"], market, cfg["timeframe"], rules)


def scheduler_loop(state):
    """Runs for the life of the server: scans every section whose timer is due."""
    while True:
        try:
            for m in AUTO_MARKETS:
                with state.lock:
                    nr = state.next_run.get(m)
                    due = bool(state.cfg[m]["auto_on"] and nr is not None
                               and time.time() >= nr and not state.busy.get(m))
                if not due:
                    continue
                state.busy[m] = True
                try:
                    run_market_job(state, m)
                    state.errors.pop(m, None)
                except Exception as e:
                    state.errors[m] = str(e)[:120]
                finally:
                    state.busy[m] = False
                    with state.lock:
                        state.last_run[m] = time.time()
                        if state.cfg[m]["auto_on"]:
                            state.next_run[m] = time.time() + state._secs(m)
        except Exception:
            pass
        time.sleep(3)


def start_scheduler(state):
    """Start the background thread once (safe to call on every rerun)."""
    with state.lock:
        if state.thread is None or not state.thread.is_alive():
            t = threading.Thread(target=scheduler_loop, args=(state,), daemon=True, name="pa-scheduler")
            state.thread = t
            t.start()


def dispatch_in_background(results, scanner_id, market, timeframe, rules):
    """Used by SCAN NOW so the screen does not wait for Telegram."""
    t = threading.Thread(target=dispatch_alerts, args=(list(results), scanner_id, market, timeframe, rules),
                         daemon=True, name="pa-alerts")
    t.start()


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
import streamlit.components.v1 as components


@st.cache_resource
def get_auto_state():
    return AutoState()          # one shared engine for every browser session


AUTO = get_auto_state()
start_scheduler(AUTO)           # background auto-scan thread (Stocks + Crypto + Forex)

if "market" not in st.session_state:
    st.session_state["market"] = "Indian Stocks"
if "scanner" not in st.session_state:           # 1 = Price Action, 2 = Head & Shoulders
    st.session_state["scanner"] = AUTO.get(st.session_state["market"])["scanner"]

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
    """Open a page. Opening the Dashboard always returns to the clean home screen."""
    st.session_state["page"] = p
    st.session_state["_goto_n"] = st.session_state.get("_goto_n", 0) + 1     # triggers scroll-to-top
    if p == "dashboard":
        st.session_state["show_add"] = False
        st.session_state["show_watchlist"] = False
        st.session_state["rules_edit"] = False
        st.session_state["scanner"] = AUTO.get(st.session_state.get("market", "Indian Stocks"))["scanner"]


def set_scanner(n):
    """Dashboard picker: remembered for the selected section (also used by its auto scan)."""
    st.session_state["scanner"] = n
    AUTO.update(st.session_state.get("market", "Indian Stocks"), scanner=n)


def set_scanner_view(n):
    """Rules page picker: only chooses which scanner's rules you are looking at."""
    st.session_state["scanner"] = n


def set_market(m):
    st.session_state["market"] = m
    st.session_state["scanner"] = AUTO.get(m)["scanner"]


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
.st-key-bottomnav [data-testid="stHorizontalBlock"] { gap: 0.1rem !important; }
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

/* ---------- market pills (Stocks / Crypto / Forex) : full width, big ---------- */
.st-key-market_wrap [data-testid="stHorizontalBlock"] {
  flex-wrap: nowrap !important; gap: 0.55rem !important; width: 100% !important;
}
.st-key-market_wrap [data-testid="stColumn"], .st-key-market_wrap [data-testid="column"] { min-width: 0 !important; }
.st-key-market_wrap.st-key-market_wrap button {
  min-height: 4.3rem !important; border-radius: 24px !important; padding: 0 0.2rem !important;
  background: rgba(18,14,40,0.85) !important; border: 2px solid rgba(255,255,255,0.18) !important;
  transition: all 0.25s ease;
  opacity: 0.42; filter: blur(1.3px) saturate(0.55); transform: scale(0.96);   /* not selected = faded */
}
.st-key-market_wrap.st-key-market_wrap button p {
  font-size: 1.3rem !important; font-weight: 800 !important; margin: 0 !important; white-space: nowrap;
}
.st-key-mk_stocks button { border-color: rgba(0,230,118,0.6) !important; color: #69f0ae !important; }
.st-key-mk_crypto button { border-color: rgba(224,64,251,0.6) !important; color: #ea80fc !important; }
.st-key-mk_forex  button { border-color: rgba(0,176,255,0.6) !important; color: #80d8ff !important; }

/* ---------- scanner picker (Scanner 1 / Scanner 2) ---------- */
.st-key-scanner_wrap [data-testid="stHorizontalBlock"] {
  flex-wrap: nowrap !important; gap: 0.55rem !important; width: 100% !important;
}
.st-key-scanner_wrap [data-testid="stColumn"], .st-key-scanner_wrap [data-testid="column"] { min-width: 0 !important; }
.st-key-scanner_wrap.st-key-scanner_wrap button {
  min-height: 4rem !important; border-radius: 22px !important; padding: 0.2rem 0.3rem !important;
  background: rgba(18,14,40,0.85) !important; border: 2px solid rgba(255,255,255,0.18) !important;
  transition: all 0.25s ease;
  opacity: 0.42; filter: blur(1.1px) saturate(0.55); transform: scale(0.97);
}
.st-key-scanner_wrap.st-key-scanner_wrap button p {
  font-size: 1.05rem !important; font-weight: 800 !important; line-height: 1.3 !important;
  margin: 0 !important; white-space: nowrap; text-align: center;
}
.st-key-sc_1.st-key-sc_1 button { border-color: rgba(0,229,255,0.65) !important; color: #80deea !important; }
.st-key-sc_2.st-key-sc_2 button { border-color: rgba(255,179,0,0.70) !important; color: #ffd54f !important; }

/* ---------- clickable title + extra side-by-side rows ---------- */
.st-key-hdr_title.st-key-hdr_title button {
  background: transparent !important; border: none !important; box-shadow: none !important;
  min-height: 3rem !important; padding: 0 !important;
}
.st-key-hdr_title.st-key-hdr_title button p {
  margin: 0 !important; font-size: 1.75rem !important; font-weight: 900 !important; font-style: italic;
  letter-spacing: -0.4px; line-height: 1.2 !important;
  background: linear-gradient(90deg, #00e5ff, #00e676, #ffea00);
  -webkit-background-clip: text; -webkit-text-fill-color: transparent;
  filter: drop-shadow(0 0 16px #00e5ff55);
}
.st-key-ar_sw_row [data-testid="stHorizontalBlock"], .st-key-panel_auto_all [data-testid="stHorizontalBlock"] {
  flex-wrap: nowrap !important; align-items: center !important; gap: 0.5rem !important;
}
.st-key-ar_sw_row [data-testid="stColumn"], .st-key-ar_sw_row [data-testid="column"],
.st-key-panel_auto_all [data-testid="stColumn"], .st-key-panel_auto_all [data-testid="column"] { min-width: 0 !important; }

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

/* ---------- progress line (single bar, same look as the score line) ---------- */
.pbar-wrap { padding: 0.55rem 14px 0.2rem 14px; }
.pbar { height: 12px; border-radius: 999px; background: rgba(255,255,255,0.14); position: relative; }
.pfill {
  height: 100%; border-radius: 999px; position: relative; transition: width 0.2s ease;
  background: linear-gradient(90deg, #00e676 0%, #c6ff00 50%, #ffea00 78%, #ff9100 100%);
}
.pknob {
  position: absolute; right: -14px; top: 50%; transform: translateY(-50%);
  width: 28px; height: 28px; border-radius: 50%;
  background: radial-gradient(circle at 35% 30%, #fff8d6 0%, #ffd54f 45%, #ff9100 100%);
  border: 2px solid rgba(255,255,255,0.75); box-shadow: 0 0 18px 5px rgba(255,193,7,0.6);
}

/* ---------- generic buttons ---------- */
:where(.stButton) > button {
  border-radius: 16px !important; font-weight: 800 !important;
  min-height: 2.9rem !important; font-size: 0.95rem !important; width: 100% !important;
  background: rgba(30,24,60,0.85) !important; color: #e8eaf6 !important;
  border: 1px solid rgba(140,100,255,0.4) !important;
}

/* ---------- SCAN NOW (elegant emerald / teal) ---------- */
.st-key-scan_main.st-key-scan_main button {
  position: relative; min-height: 4.8rem !important; border-radius: 28px !important;
  background: linear-gradient(135deg, #0b5d4e 0%, #0f8a72 50%, #1ab394 100%) !important;
  border: 1px solid rgba(160,255,225,0.40) !important;
  box-shadow: 0 10px 28px rgba(15,138,114,0.38), inset 0 1px 0 rgba(255,255,255,0.22),
              inset 0 -8px 16px rgba(0,40,30,0.25) !important;
  color: #ffffff !important;
}
.st-key-scan_main.st-key-scan_main button p {
  font-size: 1.9rem !important; font-weight: 700 !important; letter-spacing: 0.14em;
  color: #ffffff !important; text-shadow: 0 1px 6px rgba(0,30,20,0.35);
  padding-right: 3.4rem; margin: 0 !important;
}
.st-key-scan_main.st-key-scan_main button::after {
  content: ""; position: absolute; right: 18px; top: 50%; transform: translateY(-50%);
  width: 52px; height: 52px; border-radius: 50%;
  border: 1.5px solid rgba(255,255,255,0.45);
  background: rgba(255,255,255,0.10) url("data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='white' stroke-width='1.6' stroke-linecap='round'><circle cx='12' cy='12' r='9'/><circle cx='12' cy='12' r='5'/><circle cx='12' cy='12' r='1.3' fill='white'/></svg>") center / 58% no-repeat;
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
  width: min(430px, calc(100vw - 12px)) !important; z-index: 1000;
  box-sizing: border-box;
  background: linear-gradient(180deg, rgba(24,32,78,0.92), rgba(10,14,36,0.97));
  border: 1px solid rgba(124,77,255,0.38); border-radius: 32px;
  padding: 0.45rem 0.25rem !important;
  backdrop-filter: blur(18px);
  box-shadow: 0 10px 30px rgba(0,0,0,0.55), 0 0 26px rgba(0,230,118,0.12);
}
.st-key-bottomnav .stButton > button, .st-key-bottomnav button {
  background: transparent !important; border: none !important; box-shadow: none !important;
  color: #a9b8d6 !important; min-height: 4.3rem !important; padding: 0.1rem 0 !important;
  border-radius: 18px !important;
}
.st-key-bottomnav button p {
  font-size: 0.9rem !important; font-weight: 700 !important; line-height: 1.25 !important;
  text-align: center; margin: 0 !important; color: inherit !important; white-space: nowrap;
}
.st-key-bottomnav [data-testid="stIconMaterial"] { font-size: 2.2rem !important; }
.st-key-bottomnav .stButton { display: flex; justify-content: center; }
.st-key-nav_results button {
  width: 76px !important; height: 76px !important; min-height: 76px !important;
  border-radius: 50% !important; margin: -44px auto 0 auto !important;
  background: radial-gradient(circle at 30% 28%, #3ddc97 0%, #0a5d3b 72%) !important;
  border: 3px solid #69f0ae !important; color: #ffffff !important;
  box-shadow: 0 0 28px rgba(0,230,118,0.65), 0 6px 14px rgba(0,0,0,0.4) !important;
}
.st-key-nav_results [data-testid="stIconMaterial"] { font-size: 2.6rem !important; }

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


SCANNER_BTNS = [
    (1, "Scanner 1  \nPrice Action", "sc_1", "rgba(0,60,70,0.7)", "#00e5ff", "rgba(0,229,255,0.5)"),
    (2, "Scanner 2  \nHead & Shoulders", "sc_2", "rgba(80,50,0,0.7)", "#ffb300", "rgba(255,179,0,0.5)"),
]


def scanner_picker(persist=True):
    """Two big buttons (same style as Stocks/Crypto/Forex). Returns the CSS that highlights the active one."""
    cur = st.session_state.get("scanner", 1)
    with kc("scanner_wrap"):
        cols = st.columns(2)
        for col, (sid, label, key, _bg, _bd, _gl) in zip(cols, SCANNER_BTNS):
            with col:
                st.button(label, key=key, use_container_width=True,
                          on_click=(set_scanner if persist else set_scanner_view), args=(sid,))
    _, _, key, bg, bd, gl = [b for b in SCANNER_BTNS if b[0] == cur][0]
    return (
        f".st-key-{key}.st-key-{key} button {{ opacity: 1 !important; filter: none !important; "
        f"transform: scale(1.03) !important; background: {bg} !important; border-color: {bd} !important; "
        f"box-shadow: 0 0 22px {gl}, inset 0 0 12px {gl} !important; }}"
    )


MARKET_BTNS = [
    ("Indian Stocks", "📈 Stocks", "mk_stocks"),
    ("Crypto", "₿ Crypto", "mk_crypto"),
    ("Forex", "💱 Forex", "mk_forex"),
]
_MARKET_SEL = {
    "Indian Stocks": ("mk_stocks", "rgba(0,60,40,0.7)", "#00e676", "rgba(0,230,118,0.55)"),
    "Crypto": ("mk_crypto", "rgba(70,12,80,0.7)", "#e040fb", "rgba(224,64,251,0.55)"),
    "Forex": ("mk_forex", "rgba(10,34,80,0.7)", "#00b0ff", "rgba(0,176,255,0.55)"),
}


def market_picker():
    """Stocks / Crypto / Forex big buttons. Returns the CSS that highlights the selected one."""
    market = st.session_state.get("market", "Indian Stocks")
    if market not in _MARKET_SEL:
        market = "Indian Stocks"
    with kc("market_wrap"):
        mcols = st.columns(3)
        for col, (m_name, m_label, m_key) in zip(mcols, MARKET_BTNS):
            with col:
                st.button(m_label, key=m_key, use_container_width=True, on_click=set_market, args=(m_name,))
    k, bg, bd, gl = _MARKET_SEL[market]
    return (
        f".st-key-{k} button {{ opacity: 1 !important; filter: none !important; transform: scale(1.04) !important; "
        f"background: {bg} !important; border-color: {bd} !important; "
        f"box-shadow: 0 0 24px {gl}, inset 0 0 14px {gl} !important; }}"
    )


_TIMER_JS = """<div style="font-family:system-ui,sans-serif;font-weight:700;font-size:15px;color:#69f0ae;
white-space:nowrap;padding-top:4px;">&#9203; Next scan in <span id="t">--:--</span></div>
<script>var T=__T__;function f(){var s=Math.max(0,Math.round((T-Date.now())/1000));
var h=Math.floor(s/3600),m=Math.floor((s%3600)/60),x=s%60;
document.getElementById("t").textContent=(h?h+"h ":"")+String(m).padStart(2,"0")+":"+String(x).padStart(2,"0");}
f();setInterval(f,1000);</script>"""


def _auto_timer(market):
    """Live countdown. Only runs while Auto refresh is ON for this section."""
    c = AUTO.get(market)
    left = AUTO.seconds_left(market)
    if not c["auto_on"] or left is None:
        st.caption("⏱ Timer off")
    elif AUTO.busy.get(market):
        st.caption("⏳ Scanning now…")
    else:
        components.html(_TIMER_JS.replace("__T__", str(int((time.time() + max(left, 0)) * 1000))), height=32)
    err = AUTO.errors.get(market)
    if err:
        st.caption(f"⚠ last auto scan failed: {err}")


def _dash_results(market):
    """Latest results of this section (manual or automatic scan). Refreshes by itself."""
    entry = AUTO.get_results(market)
    if not entry:
        return
    results = entry["results"]
    with kc("panel_dash_results"):
        st.markdown(
            f'<div class="ptitle">Results • {SCANNER_LABELS.get(entry["scanner"], "")[:9]} • '
            f'{MARKET_LABEL_SHORT.get(market, market)} • {entry["tf"]} • {entry["time"]} • {entry["source"]}</div>',
            unsafe_allow_html=True,
        )
        if not results:
            st.caption("No setups found on last scan.")
        else:
            st.caption(f"{len(results)} setup(s) found — tap the centre button for charts")
            for res in results:
                st.markdown(result_card_html(res, detailed=False), unsafe_allow_html=True)


def _wrap_fragment(fn, every):
    frag = getattr(st, "fragment", None)
    if frag is None:
        return fn
    try:
        return frag(run_every=every)(fn)
    except Exception:
        return fn


auto_timer = _wrap_fragment(_auto_timer, 10)
dash_results = _wrap_fragment(_dash_results, 10)


def _save_tg(key, wkey):
    """Alert settings are saved the moment you change them and stay until you change them again."""
    snap = settings_snapshot()
    v = st.session_state.get(wkey)
    if key in ("telegram_max_alerts", "telegram_min_score"):
        v = int(v)
    elif key in ("enable_telegram", "telegram_send_chart"):
        v = bool(v)
    elif isinstance(v, str):
        v = v.strip()
    snap[key] = v
    save_json_atomic(SETTINGS_FILE, snap)


def _set_auto_on(m):
    AUTO.update(m, auto_on=bool(st.session_state.get(f"all_on_{m}")))


def _all_auto(flag):
    for m in AUTO_MARKETS:
        AUTO.update(m, auto_on=flag)
        st.session_state.pop(f"all_on_{m}", None)


def progress_html(frac):
    pct = int(max(0.0, min(1.0, float(frac))) * 100)
    return (f'<div class="pbar-wrap"><div class="pbar"><div class="pfill" style="width:{pct}%">'
            f'<span class="pknob"></span></div></div></div>')


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
    if res.get("pattern"):
        body += (f"<br>Target <b>{res.get('target','—')}</b> • Stop <b>{res.get('stop','—')}</b> "
                 f"• R:R <b>{res.get('rr','—')}</b>")
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
        if res.get("level_price") and not res.get("pattern"):
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
                    tcolor = "#ffb300" if res.get("pattern") else ("#00e676" if res.get("direction") == "Bullish" else "#ff5252")
                    fig.add_trace(go.Scatter(
                        x=xs, y=ys, mode="lines+markers",
                        line=dict(color=tcolor, width=2, dash="solid"),
                        marker=dict(size=7, color=tcolor),
                        name="Trendline",
                    ), row=1, col=1)
            except Exception:
                pass

        # Scanner 2: label Left shoulder / Head / Right shoulder and draw target + stop
        if res.get("pattern"):
            try:
                pp = res.get("pattern_points") or []
                px, py, pt = [], [], []
                for lab, x_str, y in pp:
                    for i in df.index:
                        if str(i) == x_str or str(i)[:16] == str(x_str)[:16]:
                            px.append(i); py.append(y); pt.append(lab)
                            break
                if px:
                    fig.add_trace(go.Scatter(
                        x=px, y=py, mode="markers+text", text=pt, textposition="top center",
                        marker=dict(size=9, color="#ffb300"), textfont=dict(color="#ffd54f", size=12),
                        name="Pattern"), row=1, col=1)
                if res.get("target") is not None:
                    fig.add_hline(y=res["target"], line_dash="dot", line_color="#00e676", line_width=1.5,
                                  annotation_text="Target", annotation_position="bottom left", row=1, col=1)
                if res.get("stop") is not None:
                    fig.add_hline(y=res["stop"], line_dash="dot", line_color="#ff5252", line_width=1.5,
                                  annotation_text="Stop", annotation_position="top left", row=1, col=1)
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


# ---- Scanner 2 rule editor definition: (key, kind, label, min, max, step, help)
HNS_FIELDS = [
    ("Patterns to scan", [
        ("scan_bearish_hs", "bool", "Head & Shoulders (bearish)", 0, 0, 0, "Break BELOW the neckline."),
        ("scan_inverse_hs", "bool", "Inverse Head & Shoulders (bullish)", 0, 0, 0, "Break ABOVE the neckline."),
    ]),
    ("Prior trend", [
        ("require_prior_trend", "bool", "Require a trend into the left shoulder", 0, 0, 0, None),
        ("min_trend_pct", "float", "Min move into left shoulder (%)", 0.1, 20, 0.5, "Uses the smaller of this % and the ATR multiple below."),
        ("trend_atr_mult", "float", "…or ATR multiple", 0, 10, 0.5, "0 = use % only."),
        ("trend_lookback", "int", "Trend lookback (candles)", 10, 150, 5, None),
    ]),
    ("Pattern shape", [
        ("swing_strength", "int", "Peak sensitivity (candles each side)", 2, 8, 1, "Higher = only bigger, cleaner peaks."),
        ("min_head_prominence_pct", "float", "Head above shoulders (%)", 0.1, 10, 0.1, "Smaller of this % and the ATR multiple below."),
        ("head_atr_mult", "float", "…or ATR multiple", 0, 5, 0.25, "0 = use % only."),
        ("max_shoulder_diff_pct", "float", "Max shoulder height difference (%)", 5, 60, 1, "Measured from the neckline."),
        ("max_time_asym_pct", "float", "Max left/right time difference (%)", 10, 90, 5, None),
        ("max_neckline_slope_deg", "float", "Max neckline slope (°)", 0.5, 30, 0.5, "Relative to head height. Raise to allow tilted necklines."),
    ]),
    ("Pattern size (candles)", [
        ("min_pattern_candles", "int", "Min, left shoulder → right shoulder", 8, 100, 1, None),
        ("max_pattern_candles", "int", "Max, left shoulder → right shoulder", 20, 300, 5, None),
        ("min_gap_candles", "int", "Min candles between peaks", 2, 20, 1, None),
    ]),
    ("Breakout", [
        ("min_break_pct", "float", "Close beyond neckline by (%)", 0.05, 3, 0.05, None),
        ("require_healthy_break", "bool", "Strong breakout candle (body ≥ 45%)", 0, 0, 0, None),
        ("max_break_delay", "int", "Breakout within N candles of right shoulder", 1, 40, 1, None),
        ("invalid_if_above_rs", "bool", "Invalid if price passes the right shoulder first", 0, 0, 0, None),
        ("max_signal_age", "int", "Only show signals from the last N candles", 0, 30, 1, "Keeps results fresh."),
    ]),
    ("Optional filters", [
        ("require_volume", "bool", "Require volume confirmation", 0, 0, 0, "High at left shoulder/head, lower at right shoulder, higher on breakout. Skipped when a market has no volume data."),
        ("require_retest", "bool", "Require a neckline retest", 0, 0, 0, "Signal is then the retest candle."),
        ("require_min_rr", "bool", "Require minimum reward : risk", 0, 0, 0, "Target = neckline − head height. Stop = right shoulder."),
        ("min_rr", "float", "Minimum reward : risk", 0.5, 10, 0.1, None),
    ]),
]


def reset_hns_rules():
    save_json(HNS_FILE, dict(DEFAULT_HNS_RULES))
    for k in [k for k in st.session_state.keys() if str(k).startswith("hns_")]:
        del st.session_state[k]


# ====================== DATA ======================
watchlists = load_json(WATCHLIST_FILE, DEFAULT_WATCHLISTS)
scanners = load_json(SCANNERS_FILE, DEFAULT_SCANNERS)
_raw_settings = load_json(SETTINGS_FILE, {})
settings = dict(DEFAULT_SETTINGS)
settings.update(_raw_settings)
# Optional: keep Telegram details in Streamlit "Secrets" so they survive an app reboot
_changed = False
for _k, _sk in (("telegram_token", "TELEGRAM_TOKEN"), ("telegram_chat_id", "TELEGRAM_CHAT_ID")):
    try:
        _v = st.secrets.get(_sk)
    except Exception:
        _v = None
    if _v and not settings.get(_k):
        settings[_k] = str(_v)
        _changed = True
try:
    _en = st.secrets.get("TELEGRAM_ENABLED")
except Exception:
    _en = None
if _en is not None and "enable_telegram" not in _raw_settings:
    settings["enable_telegram"] = bool(_en)
    _changed = True
if _changed:
    try:
        save_json_atomic(SETTINGS_FILE, settings)
    except Exception:
        pass
history = load_json(HISTORY_FILE, [])
hns_rules = dict(DEFAULT_HNS_RULES)
hns_rules.update(load_json(HNS_FILE, {}))
scanner_id = st.session_state.get("scanner", 1)
SCANNER_NAMES = {1: "Scanner 1 · Price Action", 2: "Scanner 2 · Head & Shoulders"}

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
        st.button("PA Scanner", key="hdr_title", use_container_width=True, on_click=go_page, args=("dashboard",))
    with h3:
        st.button("☀️" if is_dark else "🌙", key="theme_toggle", use_container_width=True, on_click=toggle_theme)


# ====================================================================
#                              DASHBOARD
# ====================================================================
if page == "dashboard":

    # ---------- Market selector (selected glows, others fade) + Scanner picker ----------
    dyn_css += market_picker()
    market = st.session_state.get("market", "Indian Stocks")
    if market not in MARKET_ORDER:
        market = "Indian Stocks"
    dyn_css += scanner_picker()
    scanner_id = st.session_state.get("scanner", 1)
    cfg = AUTO.get(market)                 # remembered settings of this section (defaults: 15m / 15m)
    current_list = watchlists.get(market, [])

    # ---------- Scan panel ----------
    with kc("panel_scan"):
        # Timeframe  (big colourful label, dropdown next to it) - default 15m
        with kc("tf_row"):
            c1, c2 = st.columns([1.2, 1])
            with c1:
                st.markdown('<div class="lbl-tf">⏱ Timeframe</div>', unsafe_allow_html=True)
            with c2:
                tf_default = cfg["timeframe"] if cfg["timeframe"] in TF_OPTIONS else "15m"
                timeframe = st.selectbox(
                    "Timeframe",
                    TF_OPTIONS,
                    index=TF_OPTIONS.index(tf_default),
                    key=f"tf_{market}",
                    label_visibility="collapsed",
                )

        if scanner_id == 2 and timeframe in ("1m", "3m", "5m"):
            st.caption("⚠ Head & Shoulders is mostly noise on small timeframes — 15m or higher works best.")

        # Min score
        score_default = int(cfg["min_score"])
        score_now = int(st.session_state.get(f"score_{market}", score_default))
        st.markdown(
            f'<div class="ms-row"><span class="lbl-ms">Min Score '
            f'<span class="info" title="Only setups with a score at or above this are shown">ⓘ</span></span>'
            f'<span class="ms-val">{score_now} <span class="star">★</span></span></div>',
            unsafe_allow_html=True,
        )
        min_score = st.slider("Min Score", 0, 100, score_default, 5, key=f"score_{market}", label_visibility="collapsed")
        _p = max(0, min(100, int(min_score)))
        dyn_css += (
            '[data-testid="stSlider"] div[data-baseweb="slider"] > div:first-child > div:first-child, '
            '[data-testid="stSlider"] div[data-baseweb="slider"] div:has(+ div[role="slider"]) {'
            f'background: linear-gradient(to right, #00e676 0%, #c6ff00 {_p*0.5:.0f}%, #ffea00 {_p*0.78:.0f}%, '
            f'#ff9100 {_p}%, rgba(255,255,255,0.14) {_p}%, rgba(255,255,255,0.14) 100%) !important; }}'
        )

        only_full = st.toggle("Full Matches Only", value=bool(cfg["full_only"]), key=f"full_{market}")

        # Auto refresh: interval (default 15m) + ON/OFF button + live timer
        with kc("ar_row"):
            a1, a2 = st.columns([1.2, 1])
            with a1:
                st.markdown('<div class="lbl-ar">⟳ Auto Refresh</div>', unsafe_allow_html=True)
            with a2:
                ar_default = cfg["interval"] if cfg["interval"] in TF_OPTIONS else "15m"
                ar_interval = st.selectbox(
                    "Auto refresh every",
                    TF_OPTIONS,
                    index=TF_OPTIONS.index(ar_default),
                    key=f"ar_int_{market}",
                    label_visibility="collapsed",
                )
        with kc("ar_sw_row"):
            s1, s2 = st.columns([1, 1.15])
            with s1:
                ar_on = st.toggle(
                    f"Auto refresh {'ON' if cfg['auto_on'] else 'OFF'}",
                    value=bool(cfg["auto_on"]),
                    key=f"ar_on_{market}",
                )
            # remember everything for this section (the background engine uses these settings)
            AUTO.update(market, timeframe=timeframe, min_score=int(min_score), full_only=bool(only_full),
                        scanner=scanner_id, interval=ar_interval, auto_on=bool(ar_on))
            with s2:
                auto_timer(market)

        _tg_on = bool(settings.get("enable_telegram") and settings.get("telegram_token"))
        st.caption("📨 Telegram alerts ON — sent automatically after every scan" if _tg_on
                   else "📨 Telegram alerts OFF — turn them on in the Alerts page")

        # Progress line ABOVE the scan button
        progress_slot = st.empty()
        status_slot = st.empty()
        progress_slot.markdown(progress_html(0), unsafe_allow_html=True)
        st.markdown('<div style="height:0.6rem"></div>', unsafe_allow_html=True)

        run = st.button("SCAN NOW", type="primary", use_container_width=True, key="scan_main")
        if st.button("🔄 Refresh Now", use_container_width=True, key="refresh_now"):
            st.session_state["force_scan"] = True
            st.rerun()

    # ---------- run scan (same scan loop the auto engine uses) ----------
    force = st.session_state.get("force_scan", False)
    should = run or force
    if force:
        st.session_state["force_scan"] = False

    if should:
        if not current_list:
            st.error("Watchlist empty. Add symbols first.")
        else:
            cfg_now = {"timeframe": timeframe, "min_score": int(min_score),
                       "full_only": bool(only_full), "scanner": scanner_id}

            def _prog(stage, i, n, sym):
                if stage == "start":
                    status_slot.caption(f"Scanning {sym} ({i+1}/{n})")
                else:
                    progress_slot.markdown(progress_html((i + 1) / max(n, 1)), unsafe_allow_html=True)

            results, _rules_used = core_scan(market, cfg_now, progress=_prog)
            status_slot.caption("Scan complete")

            AUTO.set_results(market, results, timeframe, scanner_id, "manual")
            record_history(market, timeframe, scanner_id, results)

            # automatic Telegram alerts as soon as the scan is complete (runs in the background)
            if _tg_on:
                dispatch_in_background(results, scanner_id, market, timeframe, _rules_used)
                status_slot.caption("Scan complete • 📨 sending Telegram alerts…")

            if not results:
                st.warning("No setups found. Try lower score or other TF.")

    # ---------- results: right below the scan button (also shows automatic scans) ----------
    dash_results(market)

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
elif page == "rules" and scanner_id == 2:
    with kc("panel_rules_head"):
        page_title("Scanner 2 Rules", "Head & Shoulders / Inverse H&S — change any number, then Save")
    dyn_css += scanner_picker(persist=False)

    new_rules = dict(hns_rules)
    for gi, (g_title, g_fields) in enumerate(HNS_FIELDS):
        with kc(f"panel_hns_{gi}"):
            st.markdown(f'<div class="ptitle">{g_title}</div>', unsafe_allow_html=True)
            for name, kind, label, lo, hi, step, hlp in g_fields:
                wk = f"hns_{name}"
                if kind == "bool":
                    new_rules[name] = st.checkbox(label, value=bool(hns_rules[name]), key=wk, help=hlp)
                elif kind == "int":
                    new_rules[name] = int(st.number_input(label, min_value=int(lo), max_value=int(hi),
                                                          value=int(hns_rules[name]), step=int(step), key=wk, help=hlp))
                else:
                    new_rules[name] = float(st.number_input(label, min_value=float(lo), max_value=float(hi),
                                                            value=float(hns_rules[name]), step=float(step),
                                                            format="%.2f", key=wk, help=hlp))

    with kc("panel_hns_save"):
        if st.button("💾 Save Scanner 2 rules", type="primary", use_container_width=True, key="save_hns"):
            save_json(HNS_FILE, new_rules)
            st.success("Saved — applies to the next scan")
        st.button("↩️ Reset to recommended defaults", use_container_width=True, key="reset_hns",
                  on_click=reset_hns_rules)

elif page == "rules":
    with kc("panel_rules_head"):
        page_title("Scanner Rules", "View and change the rules the scanner uses")
    dyn_css += scanner_picker(persist=False)

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
    market = st.session_state.get("market", "Indian Stocks")
    if market not in MARKET_ORDER:
        market = "Indian Stocks"
    entry = AUTO.get_results(market)
    with kc("panel_res_head"):
        if entry:
            sub = (f"{SCANNER_LABELS.get(entry['scanner'], '')} • {market} • {entry['tf']} • "
                   f"{entry['time']} • {entry['source']}")
        else:
            sub = f"{market} • no scan yet"
        page_title("Scan Results", sub)
        st.button("← Back to Dashboard", use_container_width=True, key="res_back",
                  on_click=go_page, args=("dashboard",))
    dyn_css += market_picker()

    results = entry["results"] if entry else []
    if not results:
        st.info("No results for this section yet. Run a scan, or turn Auto refresh ON.")
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
        st.toggle("Telegram alerts ON / OFF", value=bool(settings.get("enable_telegram", False)),
                  key="tg_enable", on_change=_save_tg, args=("enable_telegram", "tg_enable"))
        tg_token = st.text_input("Bot Token", value=settings.get("telegram_token", ""), type="password",
                                 key="tg_token", on_change=_save_tg, args=("telegram_token", "tg_token"))
        tg_chat = st.text_input("Chat ID", value=settings.get("telegram_chat_id", ""),
                                key="tg_chat", on_change=_save_tg, args=("telegram_chat_id", "tg_chat"))
        tg_chart = st.toggle("Send chart image with every alert", value=bool(settings.get("telegram_send_chart", True)),
                             key="tg_chart", on_change=_save_tg, args=("telegram_send_chart", "tg_chart"))
        st.number_input("Max alerts per scan", min_value=1, max_value=10,
                        value=int(settings.get("telegram_max_alerts", 4)), step=1,
                        key="tg_max", on_change=_save_tg, args=("telegram_max_alerts", "tg_max"))
        st.number_input("Only alert if score is at least", min_value=0, max_value=100,
                        value=int(settings.get("telegram_min_score", 0)), step=5,
                        key="tg_min", on_change=_save_tg, args=("telegram_min_score", "tg_min"))
        st.caption("Everything here saves the moment you change it and stays that way until you change it again. "
                   "Alerts go out automatically after every scan (SCAN NOW or Auto refresh) with the chart and full "
                   "details. The same setup is never sent twice.")

        if st.button("📨 Send latest result as a test", use_container_width=True, key="tg_test"):
            _m, _entry = AUTO.latest_any()
            if not (tg_token and tg_chat):
                st.warning("Enter the bot token and chat ID first.")
            elif not _entry or not _entry["results"]:
                st.warning("Run a scan first - the test sends your latest result.")
            else:
                _sid = _entry["scanner"]
                ok, mode, note = send_full_alert(
                    tg_token, tg_chat, _entry["results"][0], _sid, _m, _entry["tf"],
                    hns_rules if _sid == 2 else active_rules, with_chart=bool(tg_chart),
                )
                if ok and mode == "chart":
                    st.success("Sent with chart image ✅")
                elif ok:
                    st.warning(f"Sent as text only. Reason: {note or 'chart off'}")
                else:
                    st.error(f"Could not send. {note}")

    with kc("panel_auto_all"):
        st.markdown('<div class="ptitle">Auto scan · all sections</div>', unsafe_allow_html=True)
        for m in MARKET_ORDER:
            c = AUTO.get(m)
            e = AUTO.get_results(m)
            left = AUTO.seconds_left(m)
            r1, r2 = st.columns([1.6, 1])
            with r1:
                st.markdown(f"**{MARKET_LABEL[m]}**")
            with r2:
                st.toggle(f"Auto {m}", value=bool(c["auto_on"]), key=f"all_on_{m}",
                          label_visibility="collapsed", on_change=_set_auto_on, args=(m,))
            bits = [f"{c['timeframe']} candles", f"every {c['interval']}", SCANNER_LABELS[c["scanner"]][:9]]
            if c["auto_on"] and left is not None:
                bits.append("scanning now" if AUTO.busy.get(m) else f"next in {max(left, 0) // 60}m {max(left, 0) % 60:02d}s")
            if e:
                bits.append(f"last {e['time']} ({e['source']}) · {len(e['results'])} found")
            st.caption(" • ".join(bits))
        b1, b2 = st.columns(2)
        with b1:
            st.button("▶ All ON", use_container_width=True, key="all_auto_on", on_click=_all_auto, args=(True,))
        with b2:
            st.button("■ All OFF", use_container_width=True, key="all_auto_off", on_click=_all_auto, args=(False,))
        st.caption("Each section uses the timeframe, min score and scanner you last set for it on the Dashboard. "
                   "Auto scans keep running in the background while the app is awake - even if you are on another "
                   "page or section.")


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
                f'{h.get("scanner","Scanner 1 · Price Action")}<br>{h.get("market","")} • {h.get("timeframe","")} • <b>{h.get("count",0)}</b> setups{top}</div>'
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
    ncols = st.columns([1.05, 1, 0.95, 1, 1])
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

# Return to the top of the screen whenever a page / the title / Dashboard is tapped
if st.session_state.get("_goto_n", 0) != st.session_state.get("_goto_seen", 0):
    st.session_state["_goto_seen"] = st.session_state.get("_goto_n", 0)
    components.html(
        "<script>try{var d=window.parent.document;"
        "var m=d.querySelector('[data-testid=\"stMain\"]')||d.querySelector('section.main');"
        "if(m){m.scrollTo({top:0,behavior:'smooth'});}window.parent.scrollTo(0,0);}catch(e){}</script>",
        height=0,
    )
