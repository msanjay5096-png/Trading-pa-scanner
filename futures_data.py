"""
futures_data.py - real NIFTY / BANKNIFTY *futures* candles (with true volume) via Angel One SmartAPI (free data).
Setup once:  pip install smartapi-python pyotp logzero   (add to requirements.txt)
Streamlit -> Settings -> Secrets (or environment variables):
    ANGEL_API_KEY      = "..."   # from smartapi.angelone.in -> create app
    ANGEL_CLIENT_CODE  = "A123456"
    ANGEL_PIN          = "1234"  # your Angel login PIN/MPIN
    ANGEL_TOTP_SECRET  = "..."   # the TOTP secret text shown when you enable TOTP on SmartAPI
pa_engine.fetch() uses this automatically for ^NSEI and ^NSEBANK when the secrets exist; otherwise it falls back to Yahoo.
"""
import os, time, requests
from datetime import datetime, timedelta, timezone
import pandas as pd

IST = timezone(timedelta(hours=5, minutes=30))
FUT_MAP = {"^NSEI": "NIFTY", "^NSEBANK": "BANKNIFTY", "NIFTY": "NIFTY", "BANKNIFTY": "BANKNIFTY"}
MASTER_URL = "https://margincalculator.angelbroking.com/OpenAPI_File/files/OpenAPIScripMaster.json"
# yahoo-style interval -> (Angel interval, safe days per request)
INTERVALS = {"1m": ("ONE_MINUTE", 20), "5m": ("FIVE_MINUTE", 60), "15m": ("FIFTEEN_MINUTE", 90),
             "30m": ("THIRTY_MINUTE", 90), "1h": ("ONE_HOUR", 150), "1d": ("ONE_DAY", 400)}
_CACHE = {"sess": None, "t": 0.0, "master": None, "mt": 0.0}


def is_fut_symbol(symbol):
    return symbol.upper() in FUT_MAP


def _cfg(k):
    v = os.environ.get(k)
    if v:
        return v
    try:
        import streamlit as st
        return st.secrets.get(k)
    except Exception:
        return None


def configured():
    return all(_cfg(k) for k in ("ANGEL_API_KEY", "ANGEL_CLIENT_CODE", "ANGEL_PIN", "ANGEL_TOTP_SECRET"))


def _session():
    if _CACHE["sess"] is not None and time.time() - _CACHE["t"] < 6 * 3600:
        return _CACHE["sess"]
    from SmartApi import SmartConnect
    import pyotp
    api = SmartConnect(api_key=_cfg("ANGEL_API_KEY"))
    r = api.generateSession(_cfg("ANGEL_CLIENT_CODE"), _cfg("ANGEL_PIN"), pyotp.TOTP(_cfg("ANGEL_TOTP_SECRET")).now())
    if not r or not r.get("status"):
        raise RuntimeError(f"Angel login failed: {r.get('message') if r else 'no response'}")
    _CACHE.update(sess=api, t=time.time())
    return api


def _master():
    if _CACHE["master"] is None or time.time() - _CACHE["mt"] > 12 * 3600:
        _CACHE["master"] = requests.get(MASTER_URL, timeout=60).json()
        _CACHE["mt"] = time.time()
    return _CACHE["master"]


def pick_contract(master, name, today=None, roll_days=2):
    """Nearest running monthly future; within `roll_days` of expiry -> next month (volume has moved there)."""
    today = today or datetime.now(IST).date()
    c = []
    for r in master:
        if r.get("exch_seg") == "NFO" and r.get("instrumenttype") == "FUTIDX" and r.get("name") == name:
            try:
                c.append((datetime.strptime(r["expiry"], "%d%b%Y").date(), r))
            except Exception:
                pass
    c = sorted([x for x in c if x[0] >= today], key=lambda x: x[0])
    if not c:
        return None
    if (c[0][0] - today).days <= roll_days and len(c) > 1:
        return c[1][1]
    return c[0][1]


def fetch_futures(symbol, interval="15m", days=30):
    """Returns OHLCV DataFrame (IST index) of the current futures contract, or None."""
    name = FUT_MAP.get(symbol.upper())
    if not name or interval not in INTERVALS or not configured():
        return None
    ang, chunk = INTERVALS[interval]
    contract = pick_contract(_master(), name)
    if not contract:
        return None
    api = _session()
    end = datetime.now(IST).replace(tzinfo=None)
    start = end - timedelta(days=int(days))
    rows, cur = [], start
    while cur < end:
        nxt = min(cur + timedelta(days=chunk), end)
        res = api.getCandleData({"exchange": "NFO", "symboltoken": contract["token"], "interval": ang,
                                 "fromdate": cur.strftime("%Y-%m-%d %H:%M"), "todate": nxt.strftime("%Y-%m-%d %H:%M")})
        if res and res.get("status") and res.get("data"):
            rows += res["data"]
        time.sleep(0.4)                      # stay under the rate limit
        cur = nxt
    if not rows:
        return None
    df = pd.DataFrame(rows, columns=["t", "Open", "High", "Low", "Close", "Volume"])
    ts = pd.to_datetime(df["t"].astype(str).str.replace(r"\+05:30$", "", regex=True))
    df.index = ts.dt.tz_localize(IST)
    df = df.drop(columns="t").astype(float)
    df = df[~df.index.duplicated()].sort_index()
    df.attrs["contract"] = contract.get("symbol")
    return df
